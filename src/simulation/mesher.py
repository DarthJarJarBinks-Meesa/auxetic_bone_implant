"""
src/simulation/mesher.py
=========================
gmsh meshing module for the auxetic plate pipeline.

This module takes an extruded 3D CadQuery solid, applies version-1 meshing
settings from ``config/meshing.yaml``, generates a 3D tetrahedral volume
mesh, and returns a ``MeshResult`` with metadata and file paths.

PIPELINE POSITION:
    3D solid (cq.Workplane)  →  [THIS MODULE]  →  model.msh  →  solver_exporter.py

ARCHITECTURAL DECISION — CadQuery → gmsh handoff via temporary STEP:
    CadQuery and gmsh use different CAD kernel representations (CadQuery
    wraps OCC; gmsh embeds its own OCC interface).  There is no direct
    in-memory bridge between them in version 1.  The robust, practical
    handoff is:
        1. CadQuery writes the solid to a temporary STEP file.
        2. gmsh imports that STEP using its OCC geometry kernel.
        3. gmsh synchronises the model and meshes the imported volume.
    This approach is well-tested, widely used, and agnostic to CadQuery
    internal representation details.  The temporary file is deleted after
    import.  If a future version provides a direct OCC/BRep bridge, this
    handoff can be replaced without changing the public API of this module.

ARCHITECTURAL DECISION — preset-driven meshing; local refinement is scaffolded:
    Version 1 uses global characteristic-length presets (coarse / default /
    refined) as the primary mesh-size control.  Feature refinement rules
    from meshing.yaml are loaded and stored, and a simplified global
    reduction is applied when refinement is enabled (smaller global bounds
    are used to approximate the effect).  True local field-based refinement
    (gmsh Distance + Threshold fields on identified thin surfaces) requires
    reliable geometry tagging that is not yet stable across all three
    designs in version 1.  The scaffold is in place and documented;
    full local refinement is a future-version improvement.

ARCHITECTURAL DECISION — MeshResult soft-fails by default:
    ``generate_volume_mesh()`` returns a ``MeshResult`` with ``success=False``
    and an ``error_message`` on failure, rather than raising immediately.
    This allows the case runner to log and continue without aborting the
    sweep.  Callers that cannot proceed without a valid mesh should use
    ``require_successful_mesh()``, which raises ``MeshingError``.

ARCHITECTURAL DECISION — gmsh session lifecycle (init/finalize per call):
    gmsh is a global singleton in its Python API.  Initialising and
    finalising per ``generate_volume_mesh`` call avoids state leakage
    between cases, at the cost of small startup overhead per call.
    For version-1 sweep runs where cases run sequentially, this is the
    correct trade-off.  A future parallel runner may need a per-process
    gmsh session instead.

UNITS: mm throughout.  gmsh default unit is mm when geometry is imported
from a STEP file whose internal units are mm.
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cadquery as cq
import gmsh

logger = logging.getLogger(__name__)

# Version 1 meshing target dimension (3D volume mesh)
_TARGET_DIMENSION: int = 3


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class MeshingError(Exception):
    """
    Raised when meshing fails in an unrecoverable way.

    Soft failures (e.g. per-case meshing errors in a sweep) are represented
    by ``MeshResult.success = False`` and should not raise this exception.
    Use ``require_successful_mesh()`` to promote a soft failure to a hard one.
    """


# ---------------------------------------------------------------------------
# Typed dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MeshingPreset:
    """
    Parsed meshing preset from ``config/meshing.yaml [presets]``.

    Attributes:
        name:                        Preset name (``"coarse"``, ``"default"``, ``"refined"``).
        characteristic_length_min_mm: Minimum element characteristic length [mm].
        characteristic_length_max_mm: Maximum element characteristic length [mm].
        mesh_order:                  Element order (1 = linear TET4; 2 = quadratic TET10).
        smoothing_steps:             Number of Laplacian smoothing iterations.
        optimize_mesh:               Whether to run Netgen/Laplacian optimisation.
        notes:                       Optional human-readable notes from config.
    """

    name: str
    characteristic_length_min_mm: float
    characteristic_length_max_mm: float
    mesh_order: int
    smoothing_steps: int
    optimize_mesh: bool
    notes: str | None = None

    def validate(self) -> None:
        """
        Validate preset numeric values.

        Raises:
            MeshingError: if any value is out of range.
        """
        if self.characteristic_length_min_mm <= 0.0:
            raise MeshingError(
                f"Preset '{self.name}': characteristic_length_min_mm must be "
                f"positive, got {self.characteristic_length_min_mm}."
            )
        if self.characteristic_length_max_mm <= 0.0:
            raise MeshingError(
                f"Preset '{self.name}': characteristic_length_max_mm must be "
                f"positive, got {self.characteristic_length_max_mm}."
            )
        if self.characteristic_length_max_mm < self.characteristic_length_min_mm:
            raise MeshingError(
                f"Preset '{self.name}': characteristic_length_max_mm "
                f"({self.characteristic_length_max_mm}) must be ≥ "
                f"characteristic_length_min_mm ({self.characteristic_length_min_mm})."
            )
        if self.mesh_order not in (1, 2):
            raise MeshingError(
                f"Preset '{self.name}': mesh_order must be 1 or 2, "
                f"got {self.mesh_order}."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name":                          self.name,
            "characteristic_length_min_mm":  self.characteristic_length_min_mm,
            "characteristic_length_max_mm":  self.characteristic_length_max_mm,
            "mesh_order":                    self.mesh_order,
            "smoothing_steps":               self.smoothing_steps,
            "optimize_mesh":                 self.optimize_mesh,
            "notes":                         self.notes,
        }


@dataclass
class FeatureRefinementRule:
    """
    A single feature-aware refinement rule from ``config/meshing.yaml``.

    Version 1 uses these as configuration-level awareness rather than
    as full geometric field-based refinement.  See module docstring.

    Attributes:
        name:                         Rule name (e.g. ``"hinges"``).
        enabled:                      Whether the rule is active.
        target_element_size_mm:       Target element size at the feature [mm].
        trigger_dimension_threshold_mm: Feature width below which rule applies.
        notes:                        Optional notes.
    """

    name: str
    enabled: bool
    target_element_size_mm: float | None = None
    trigger_dimension_threshold_mm: float | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name":                            self.name,
            "enabled":                         self.enabled,
            "target_element_size_mm":          self.target_element_size_mm,
            "trigger_dimension_threshold_mm":  self.trigger_dimension_threshold_mm,
            "notes":                           self.notes,
        }


@dataclass
class MeshResult:
    """
    Result of one meshing operation.

    Attributes:
        success:              True if meshing completed without errors.
        preset_name:          The name of the preset used.
        mesh_file:            Path to the written .msh file (if written).
        vtk_file:             Path to the written .vtk file (if written).
        element_count:        Total number of volume elements (if available).
        node_count:           Total number of nodes (if available).
        bbox_dimensions_mm:   (xsize, ysize, zsize) bounding box of the solid [mm].
        warnings:             List of non-fatal warning strings.
        metadata:             Additional metadata for logging and reporting.
        error_message:        Error description if success is False.
    """

    success: bool
    preset_name: str
    mesh_file: str | None = None
    vtk_file: str | None = None
    element_count: int | None = None
    node_count: int | None = None
    bbox_dimensions_mm: tuple[float, float, float] | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success":             self.success,
            "preset_name":         self.preset_name,
            "mesh_file":           self.mesh_file,
            "vtk_file":            self.vtk_file,
            "element_count":       self.element_count,
            "node_count":          self.node_count,
            "bbox_dimensions_mm":  list(self.bbox_dimensions_mm)
                                   if self.bbox_dimensions_mm else None,
            "warnings":            self.warnings,
            "metadata":            self.metadata,
            "error_message":       self.error_message,
        }


# ---------------------------------------------------------------------------
# Meshing config parsing helpers
# ---------------------------------------------------------------------------

def load_meshing_config(
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """
    Load and return the raw meshing configuration dict from
    ``config/meshing.yaml`` via ``config_loader.py``.

    Args:
        project_root: Project root for config resolution (optional).

    Returns:
        Raw meshing config dict (the full ``meshing_config`` section).

    Raises:
        MeshingError: if config cannot be loaded.
    """
    try:
        from utils.config_loader import load_pipeline_config
        cfg = load_pipeline_config(project_root)
        return cfg.meshing_config
    except Exception as exc:
        raise MeshingError(
            f"Failed to load meshing config: {exc}.  "
            f"Ensure the project config files are present and valid."
        ) from exc


def get_meshing_preset(
    name: str,
    meshing_config: dict[str, Any] | None = None,
    project_root: str | Path | None = None,
) -> MeshingPreset:
    """
    Return a parsed ``MeshingPreset`` for the given preset name.

    Args:
        name:            Preset name — one of ``"coarse"``, ``"default"``,
                         ``"refined"``.
        meshing_config:  Already-loaded meshing config dict (optional;
                         loaded from config if not provided).
        project_root:    Project root for config resolution (optional).

    Returns:
        ``MeshingPreset`` instance.

    Raises:
        MeshingError: if the preset name is not found or values are invalid.
    """
    if meshing_config is None:
        meshing_config = load_meshing_config(project_root)

    presets = meshing_config.get("presets", {})
    if name not in presets:
        available = list(presets.keys())
        raise MeshingError(
            f"Meshing preset '{name}' not found in meshing.yaml.  "
            f"Available presets: {available}."
        )

    raw = presets[name]
    try:
        preset = MeshingPreset(
            name=name,
            characteristic_length_min_mm=float(
                raw.get("characteristic_length_min_mm",
                        meshing_config.get("global_defaults", {}).get(
                            "characteristic_length_min_mm", 0.15
                        ))
            ),
            characteristic_length_max_mm=float(
                raw.get("characteristic_length_max_mm",
                        meshing_config.get("global_defaults", {}).get(
                            "characteristic_length_max_mm", 1.0
                        ))
            ),
            mesh_order=int(raw.get("mesh_order", 1)),
            smoothing_steps=int(raw.get("smoothing_steps", 5)),
            optimize_mesh=bool(raw.get("optimize_mesh", True)),
            notes=raw.get("notes"),
        )
    except (TypeError, ValueError) as exc:
        raise MeshingError(
            f"Failed to parse meshing preset '{name}': {exc}"
        ) from exc

    preset.validate()
    return preset


def get_feature_refinement_rules(
    meshing_config: dict[str, Any] | None = None,
    project_root: str | Path | None = None,
) -> list[FeatureRefinementRule]:
    """
    Parse and return the list of feature refinement rules from the meshing config.

    Rules with ``enabled: false`` are included but flagged as disabled so the
    caller knows which rules exist even if not all are active.

    Args:
        meshing_config:  Already-loaded meshing config dict (optional).
        project_root:    Project root for config resolution (optional).

    Returns:
        List of ``FeatureRefinementRule`` instances.
    """
    if meshing_config is None:
        meshing_config = load_meshing_config(project_root)

    refinement_section = meshing_config.get("feature_refinement", {})
    targets_raw = refinement_section.get("targets", {})

    rules: list[FeatureRefinementRule] = []
    for rule_name, rule_data in targets_raw.items():
        if not isinstance(rule_data, dict):
            continue
        rule = FeatureRefinementRule(
            name=rule_name,
            enabled=bool(rule_data.get("enabled", False)),
            target_element_size_mm=(
                float(rule_data["target_element_size_mm"])
                if "target_element_size_mm" in rule_data else None
            ),
            trigger_dimension_threshold_mm=(
                float(rule_data["trigger_dimension_threshold_mm"])
                if "trigger_dimension_threshold_mm" in rule_data else None
            ),
            notes=rule_data.get("notes"),
        )
        rules.append(rule)

    return rules


# ---------------------------------------------------------------------------
# gmsh session helpers
# ---------------------------------------------------------------------------

def _initialize_gmsh(model_name: str = "auxetic_plate") -> None:
    """
    Initialise a clean gmsh session.

    ARCHITECTURAL DECISION — init/finalize per call:
        See module docstring.  Always call ``_finalize_gmsh()`` after use,
        even on failure.  Use the ``_gmsh_session`` context manager for
        automatic cleanup.
    """
    gmsh.initialize()
    gmsh.model.add(model_name)
    gmsh.option.setNumber("General.Terminal", 0)   # suppress console output
    logger.debug("gmsh session initialised: model='%s'.", model_name)


def _finalize_gmsh() -> None:
    """Finalize and clean up the gmsh session."""
    try:
        gmsh.finalize()
        logger.debug("gmsh session finalised.")
    except Exception:
        pass  # ignore errors during cleanup


@contextlib.contextmanager
def _gmsh_session(model_name: str = "auxetic_plate"):
    """
    Context manager that initialises gmsh on entry and finalises on exit
    (even if an exception occurs).

    Usage::

        with _gmsh_session("my_model"):
            # ... gmsh operations ...
    """
    _initialize_gmsh(model_name)
    try:
        yield
    finally:
        _finalize_gmsh()


# ---------------------------------------------------------------------------
# Geometry handoff helpers
# ---------------------------------------------------------------------------

def _export_cadquery_solid_to_temp_step(solid: cq.Workplane) -> Path:
    """
    Export a CadQuery solid to a temporary STEP file and return the path.

    ARCHITECTURAL DECISION — STEP export as the CadQuery→gmsh bridge:
        See module docstring.  The temporary file is created in the system
        temp directory and must be deleted by the caller after import.

    Args:
        solid: CadQuery Workplane holding the 3D solid.

    Returns:
        Path to the temporary STEP file.

    Raises:
        MeshingError: if the export fails.
    """
    try:
        tmp_fd, tmp_path_str = tempfile.mkstemp(suffix=".step", prefix="auxetic_solid_")
        os.close(tmp_fd)
        tmp_path = Path(tmp_path_str)
        cq.exporters.export(solid, str(tmp_path))
        logger.debug("CadQuery solid exported to temporary STEP: %s", tmp_path)
        return tmp_path
    except Exception as exc:
        raise MeshingError(
            f"Failed to export CadQuery solid to STEP for gmsh handoff: {exc}.  "
            f"Ensure the solid is non-degenerate and CadQuery exporters are "
            f"available."
        ) from exc


def _load_step_into_gmsh(step_path: Path) -> None:
    """
    Import a STEP file into the current gmsh session using the OCC kernel.

    The OCC kernel is required for STEP import and is consistent with the
    geometry kernel used by CadQuery (both wrap OpenCASCADE).

    Args:
        step_path: Path to the STEP file to import.

    Raises:
        MeshingError: if the import fails.
    """
    if not step_path.exists():
        raise MeshingError(
            f"STEP file not found for gmsh import: {step_path}"
        )
    try:
        gmsh.model.occ.importShapes(str(step_path))
        gmsh.model.occ.synchronize()
        logger.debug("STEP file imported and gmsh OCC model synchronised.")
    except Exception as exc:
        raise MeshingError(
            f"gmsh failed to import STEP file {step_path}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Apply meshing preset
# ---------------------------------------------------------------------------

def _apply_meshing_preset(
    preset: MeshingPreset,
    global_defaults: dict[str, Any] | None = None,
) -> None:
    """
    Apply a ``MeshingPreset`` as global gmsh meshing options.

    Sets characteristic length bounds, mesh order, element size optimisation,
    and algorithm selections from the preset and global defaults.

    ARCHITECTURAL DECISION — global options only in version 1:
        gmsh supports per-entity (surface/curve) size fields for local
        refinement.  Version 1 applies only global options here.  The
        feature refinement scaffold applies a modest global size reduction
        when refinement is enabled (see ``_apply_feature_refinement_hint``).
        True per-entity fields are reserved for a future version.

    Args:
        preset:          ``MeshingPreset`` to apply.
        global_defaults: Raw ``global_defaults`` dict from meshing.yaml
                         (used for algorithm selection fallback).
    """
    gd = global_defaults or {}

    # --- Characteristic length bounds ---
    gmsh.option.setNumber(
        "Mesh.CharacteristicLengthMin", preset.characteristic_length_min_mm
    )
    gmsh.option.setNumber(
        "Mesh.CharacteristicLengthMax", preset.characteristic_length_max_mm
    )

    # --- Mesh order (element type) ---
    # mesh_order=1 → TET4 (linear); mesh_order=2 → TET10 (quadratic)
    gmsh.option.setNumber("Mesh.ElementOrder", preset.mesh_order)

    # --- Smoothing ---
    gmsh.option.setNumber("Mesh.Smoothing", preset.smoothing_steps)

    # --- Optimisation ---
    # gmsh Optimize flag: 0=off, 1=Laplacian, 2=Netgen
    gmsh.option.setNumber("Mesh.Optimize", 1 if preset.optimize_mesh else 0)

    # --- Algorithm selection from global_defaults ---
    algo_2d = int(gd.get("mesh_algorithm_2d", 6))   # 6 = Frontal-Delaunay
    algo_3d = int(gd.get("mesh_algorithm_3d", 4))   # 4 = Frontal
    gmsh.option.setNumber("Mesh.Algorithm", algo_2d)
    gmsh.option.setNumber("Mesh.Algorithm3D", algo_3d)

    # --- Recombination (must stay off for tetrahedral-only workflow) ---
    gmsh.option.setNumber("Mesh.RecombineAll", 0)

    logger.debug(
        "Applied meshing preset '%s': len_min=%.4f, len_max=%.4f, "
        "order=%d, smooth=%d, optimize=%s.",
        preset.name,
        preset.characteristic_length_min_mm,
        preset.characteristic_length_max_mm,
        preset.mesh_order,
        preset.smoothing_steps,
        preset.optimize_mesh,
    )


def _apply_feature_refinement_hint(
    rules: list[FeatureRefinementRule],
    preset: MeshingPreset,
) -> str | None:
    """
    Apply a simplified global size hint when feature refinement is enabled.

    VERSION 1 SIMPLIFICATION:
        True local field-based refinement (gmsh Distance + Threshold fields
        attached to specific thin surfaces) requires reliable identification
        of those surfaces from the imported OCC geometry.  This is not yet
        stable across all three designs.  Instead, version 1 detects whether
        any refinement rule is active and reduces the global characteristic
        length maximum slightly to approximate the intent.

        This will be replaced with true per-entity fields in a future version
        when geometry tagging is more reliable.

    Args:
        rules:   List of ``FeatureRefinementRule`` instances.
        preset:  Currently applied ``MeshingPreset``.

    Returns:
        A warning string describing the simplification if any rule is active,
        or None if no rules are enabled.
    """
    enabled_rules = [r for r in rules if r.enabled]
    if not enabled_rules:
        return None

    # Find the smallest target element size among enabled rules.
    target_sizes = [
        r.target_element_size_mm for r in enabled_rules
        if r.target_element_size_mm is not None
    ]
    if target_sizes:
        min_target = min(target_sizes)
        # Reduce global max to the smallest feature target, but not below global min.
        reduced_max = max(
            min_target,
            preset.characteristic_length_min_mm,
        )
        if reduced_max < preset.characteristic_length_max_mm:
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", reduced_max)
            logger.debug(
                "Feature refinement hint applied: global max reduced from "
                "%.4f to %.4f mm (smallest enabled target).",
                preset.characteristic_length_max_mm,
                reduced_max,
            )

    rule_names = [r.name for r in enabled_rules]
    return (
        f"VERSION 1 SIMPLIFICATION — Feature refinement rules ({rule_names}) are "
        f"enabled but applied as a global size reduction only.  "
        f"True per-entity field-based refinement is a future-version improvement."
    )


# ---------------------------------------------------------------------------
# Mesh extraction helpers
# ---------------------------------------------------------------------------

def _extract_mesh_counts() -> tuple[int | None, int | None]:
    """
    Extract node and element counts from the current gmsh model.

    Returns:
        Tuple ``(element_count, node_count)`` or ``(None, None)`` on failure.
    """
    try:
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        node_count = len(node_tags)

        # Count only 3D (tetrahedra) element types
        elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim=3)
        element_count = sum(len(tags) for tags in elem_tags)

        return element_count, node_count
    except Exception as exc:
        logger.warning("Could not extract gmsh mesh counts: %s", exc)
        return None, None


def _solid_bbox_dimensions(
    solid: cq.Workplane,
) -> tuple[float, float, float] | None:
    """
    Return the (xsize, ysize, zsize) bounding-box dimensions of the solid.

    Returns ``None`` on failure rather than raising.
    """
    try:
        bb = solid.val().BoundingBox()
        return (
            bb.xmax - bb.xmin,
            bb.ymax - bb.ymin,
            bb.zmax - bb.zmin,
        )
    except Exception as exc:
        logger.warning("Could not compute solid bounding box: %s", exc)
        return None


def _quality_warnings_from_result(
    result: MeshResult,
    quality_config: dict[str, Any],
) -> list[str]:
    """
    Check mesh counts against quality thresholds from meshing config and
    return a list of warning strings.

    Args:
        result:         Partially-populated ``MeshResult`` (counts filled in).
        quality_config: Raw ``quality_controls`` dict from meshing.yaml.

    Returns:
        List of human-readable warning strings (may be empty).
    """
    warnings: list[str] = []

    min_count = quality_config.get("minimum_element_count_warning", 500)
    max_count = quality_config.get("maximum_element_count_warning", 500_000)

    if result.element_count is not None:
        if result.element_count < min_count:
            warnings.append(
                f"Element count ({result.element_count}) is below the minimum "
                f"warning threshold ({min_count}).  "
                f"The mesh may be too coarse or the geometry may be degenerate."
            )
        if result.element_count > max_count:
            warnings.append(
                f"Element count ({result.element_count}) exceeds the maximum "
                f"warning threshold ({max_count}).  "
                f"Consider switching to the coarse or default preset."
            )

    if result.element_count == 0:
        warnings.append(
            "Zero volume elements generated.  "
            "The geometry may be degenerate or too thin to mesh at this preset."
        )

    return warnings


# ---------------------------------------------------------------------------
# Main meshing function
# ---------------------------------------------------------------------------

def generate_volume_mesh(
    solid_geometry: cq.Workplane,
    output_msh_path: str | Path,
    preset_name: str = "default",
    write_vtk: bool = True,
    apply_refinement_rules: bool = False,
    project_root: str | Path | None = None,
) -> MeshResult:
    """
    Generate a 3D tetrahedral volume mesh from a CadQuery solid.

    This is the primary meshing entry point for the pipeline.

    Workflow:
      1. Validate solid geometry.
      2. Load meshing config and parse the requested preset.
      3. Export the solid to a temporary STEP file.
      4. Import STEP into a fresh gmsh session.
      5. Apply preset global meshing options.
      6. Optionally apply feature refinement hint.
      7. Generate the 3D mesh.
      8. Write ``.msh`` (and optionally ``.vtk``).
      9. Extract counts and bounding-box metadata.
      10. Return ``MeshResult``.

    Args:
        solid_geometry:       CadQuery Workplane holding the 3D solid.
        output_msh_path:      Path to write the ``.msh`` output file.
        preset_name:          Meshing preset to use (``"coarse"``,
                              ``"default"``, or ``"refined"``).
        write_vtk:            If True, also write a ``.vtk`` file alongside
                              the ``.msh`` file.
        apply_refinement_rules: If True, apply the feature refinement hint
                              (global size reduction from active rules).
        project_root:         Project root for config resolution (optional).

    Returns:
        ``MeshResult`` — ``success=False`` with ``error_message`` if any
        step fails.

    Example::

        result = generate_volume_mesh(
            solid, "runs/case_000001/mesh/model.msh", preset_name="default"
        )
        if not result.success:
            logger.error(result.error_message)
    """
    output_msh_path = Path(output_msh_path)
    result = MeshResult(success=False, preset_name=preset_name)

    # --- Validate solid geometry ---
    if solid_geometry is None:
        result.error_message = "solid_geometry is None."
        return result
    if not isinstance(solid_geometry, cq.Workplane):
        result.error_message = (
            f"solid_geometry must be cq.Workplane, "
            f"got {type(solid_geometry).__name__}."
        )
        return result

    bbox = _solid_bbox_dimensions(solid_geometry)
    result.bbox_dimensions_mm = bbox
    if bbox is not None:
        result.metadata["bbox_x_mm"] = round(bbox[0], 4)
        result.metadata["bbox_y_mm"] = round(bbox[1], 4)
        result.metadata["bbox_z_mm"] = round(bbox[2], 4)

    # --- Load config ---
    try:
        meshing_config = load_meshing_config(project_root)
    except MeshingError as exc:
        result.error_message = str(exc)
        return result

    try:
        preset = get_meshing_preset(preset_name, meshing_config)
    except MeshingError as exc:
        result.error_message = str(exc)
        return result

    global_defaults = meshing_config.get("global_defaults", {})
    quality_config = meshing_config.get("quality_controls", {})
    export_config = meshing_config.get("export", {})
    gmsh_format = str(export_config.get("gmsh_format_version", "2.2"))
    result.metadata["preset"] = preset.to_dict()

    # --- Export to temporary STEP ---
    tmp_step: Path | None = None
    try:
        tmp_step = _export_cadquery_solid_to_temp_step(solid_geometry)
    except MeshingError as exc:
        result.error_message = str(exc)
        return result

    # --- Ensure output directory exists ---
    output_msh_path.parent.mkdir(parents=True, exist_ok=True)

    # --- gmsh session ---
    try:
        with _gmsh_session("auxetic_plate"):

            # Import geometry
            try:
                _load_step_into_gmsh(tmp_step)
            except MeshingError as exc:
                result.error_message = str(exc)
                return result

            # Apply preset
            _apply_meshing_preset(preset, global_defaults)

            # Feature refinement hint (if requested)
            if apply_refinement_rules:
                rules = get_feature_refinement_rules(meshing_config)
                hint = _apply_feature_refinement_hint(rules, preset)
                if hint:
                    result.warnings.append(hint)
                    result.metadata["refinement_rules"] = [
                        r.to_dict() for r in rules if r.enabled
                    ]

            # Generate mesh
            logger.info(
                "Running gmsh meshing (preset='%s', len_min=%.4f, len_max=%.4f mm).",
                preset.name,
                preset.characteristic_length_min_mm,
                preset.characteristic_length_max_mm,
            )
            try:
                gmsh.model.mesh.generate(_TARGET_DIMENSION)
            except Exception as exc:
                result.error_message = (
                    f"gmsh mesh generation failed: {exc}"
                )
                return result

            # Optimise if requested
            if preset.optimize_mesh:
                try:
                    gmsh.model.mesh.optimize("Netgen")
                except Exception as exc:
                    result.warnings.append(
                        f"Mesh optimisation (Netgen) failed and was skipped: {exc}"
                    )

            # Write .msh
            try:
                gmsh.option.setNumber("Mesh.MshFileVersion", float(gmsh_format))
                gmsh.write(str(output_msh_path))
                result.mesh_file = str(output_msh_path)
                logger.info("Mesh written: %s", output_msh_path)
            except Exception as exc:
                result.error_message = (
                    f"Failed to write .msh file '{output_msh_path}': {exc}"
                )
                return result

            # Write .vtk (optional)
            if write_vtk and export_config.get("write_vtk", True):
                vtk_path = output_msh_path.with_suffix(".vtk")
                try:
                    gmsh.write(str(vtk_path))
                    result.vtk_file = str(vtk_path)
                    logger.info("VTK written: %s", vtk_path)
                except Exception as exc:
                    result.warnings.append(
                        f"VTK export failed and was skipped: {exc}"
                    )

            # Extract element and node counts
            element_count, node_count = _extract_mesh_counts()
            result.element_count = element_count
            result.node_count = node_count

    finally:
        # Clean up temporary STEP file regardless of success or failure
        if tmp_step is not None and tmp_step.exists():
            try:
                tmp_step.unlink()
            except Exception:
                pass

    # --- Quality warnings ---
    quality_warnings = _quality_warnings_from_result(result, quality_config)
    result.warnings.extend(quality_warnings)

    if quality_warnings:
        for w in quality_warnings:
            logger.warning("[MESH QA] %s", w)

    result.success = True
    logger.info(
        "Meshing complete: preset='%s', elements=%s, nodes=%s, file='%s'.",
        preset_name,
        result.element_count,
        result.node_count,
        result.mesh_file,
    )
    return result


# ---------------------------------------------------------------------------
# Case convenience helper
# ---------------------------------------------------------------------------

def generate_mesh_for_case(
    case_definition: Any,
    solid_geometry: cq.Workplane,
    output_directory: str | Path,
    preset_name: str | None = None,
    apply_refinement_rules: bool | None = None,
    project_root: str | Path | None = None,
) -> MeshResult:
    """
    Mesh a solid for a specific pipeline case, writing output to the case
    mesh directory.

    ARCHITECTURAL DECISION — preset selection priority:
        1. ``preset_name`` argument (explicit override by caller).
        2. Stage-specific preset from ``meshing.yaml [staged_meshing]``
           (not yet automatically determined here; TODO for orchestrator).
        3. ``"default"`` as the fallback.

    ARCHITECTURAL DECISION — duck-typed on case_definition:
        Not importing ``CaseDefinition`` to avoid cross-layer coupling.
        The function accesses ``.case_id`` for log messages only.

    Args:
        case_definition:       Object with ``.case_id`` (e.g. CaseDefinition).
        solid_geometry:        3D CadQuery solid to mesh.
        output_directory:      Directory where mesh files will be written.
        preset_name:           Preset name override (optional).
        apply_refinement_rules: Whether to apply feature refinement hint.
        project_root:          Project root for config resolution (optional).

    Returns:
        ``MeshResult``.
    """
    output_dir = Path(output_directory)
    output_msh = output_dir / "model.msh"

    # Determine preset
    effective_preset = preset_name or "default"

    # Determine refinement rule application
    effective_refinement = apply_refinement_rules if apply_refinement_rules is not None else False

    case_id = getattr(case_definition, "case_id", "unknown_case")
    logger.info(
        "Meshing case '%s': preset='%s', output='%s'.",
        case_id, effective_preset, output_msh,
    )

    return generate_volume_mesh(
        solid_geometry=solid_geometry,
        output_msh_path=output_msh,
        preset_name=effective_preset,
        write_vtk=True,
        apply_refinement_rules=effective_refinement,
        project_root=project_root,
    )


# ---------------------------------------------------------------------------
# Hard-fail wrapper
# ---------------------------------------------------------------------------

def require_successful_mesh(
    solid_geometry: cq.Workplane,
    output_msh_path: str | Path,
    preset_name: str = "default",
    write_vtk: bool = True,
    apply_refinement_rules: bool = False,
    project_root: str | Path | None = None,
) -> MeshResult:
    """
    Run meshing and raise ``MeshingError`` if the result is unsuccessful.

    Use this when the calling code cannot continue without a valid mesh
    (e.g. immediately before solver export in a non-sweep single-case run).

    Args:
        See ``generate_volume_mesh()`` for argument descriptions.

    Returns:
        ``MeshResult`` with ``success=True``.

    Raises:
        MeshingError: if meshing fails.
    """
    result = generate_volume_mesh(
        solid_geometry=solid_geometry,
        output_msh_path=output_msh_path,
        preset_name=preset_name,
        write_vtk=write_vtk,
        apply_refinement_rules=apply_refinement_rules,
        project_root=project_root,
    )
    if not result.success:
        raise MeshingError(
            f"Meshing failed (preset='{preset_name}'): {result.error_message}"
        )
    return result
