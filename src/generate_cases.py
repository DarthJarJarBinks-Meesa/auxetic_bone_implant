"""
src/generate_cases.py
======================
Case-generation module for the auxetic plate pipeline.

This module reads sweep/config definitions, generates version-1
``CaseDefinition`` objects, supports baseline and staged sweep strategies,
and provides deterministic case IDs for downstream workflow execution.

PIPELINE POSITION:
    sweep_config.yaml  →  [THIS MODULE]  →  list[CaseDefinition]
                                         →  orchestrator.run_cases(...)
                                         →  CLI entrypoints

ARCHITECTURAL DECISION — config-driven, no hard-coded sweep values:
    All sweep values, baseline values, enabled designs/materials/load-cases,
    and stage definitions are read from ``sweep_config.yaml`` at runtime.
    The generator produces only typed, validated ``CaseDefinition`` objects —
    it has no geometry, meshing, or solver knowledge.

ARCHITECTURAL DECISION — three generation modes:
    1. ``baseline_only``:
           One case per design × material × load-case at baseline parameter
           values and baseline plate thickness.  Minimal but always valid.
    2. ``baseline_plus_one_factor_variation`` (default):
           Baseline case + cases varying one parameter at a time (including
           plate thickness) while all others hold at baseline.  Provides
           useful sensitivity information with a manageable case count.
    3. ``full_factorial``:
           Full Cartesian product across all sweep parameters, plate thickness
           values, materials, and load cases.  May produce O(100–1000) cases.
           Subject to ``max_case_count`` if provided.

ARCHITECTURAL DECISION — fillet_radius is NEVER a sweep parameter:
    ``fillet_radius`` for the tetrachiral design is derived from
    ``node_radius`` via the formula ``0.25 * (node_radius / 1.05)``.
    This module silently ignores any ``fillet_radius`` key that appears
    in sweep config to prevent inconsistent geometry.  See
    ``TetrachiralParameters.fillet_radius`` (a read-only property).

ARCHITECTURAL DECISION — deterministic case IDs via SHA-256 hash:
    Case IDs include a compact hex prefix derived from a canonical
    parameter string so that the same inputs always produce the same ID
    even across Python sessions.  IDs are filesystem-safe and readable.

UNITS: consistent with project-wide convention (mm, N, MPa, degrees).
"""

from __future__ import annotations

import hashlib
import itertools
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

from workflow.case_schema import (
    CaseDefinition,
    ReentrantParameters,
    RotatingSquareParameters,
    TetrachiralParameters,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias for design parameter union
# ---------------------------------------------------------------------------
DesignParameterSet = Union[
    ReentrantParameters, RotatingSquareParameters, TetrachiralParameters
]

# Tetrachiral-derived parameter names that must never be used as independent
# sweep dimensions.
_TETRACHIRAL_DERIVED_PARAMS: frozenset[str] = frozenset(
    {"fillet_radius", "fillet_radius_derived"}
)

# Canonical mode strings
_VALID_MODES: frozenset[str] = frozenset(
    {"baseline_only", "baseline_plus_one_factor_variation", "full_factorial"}
)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class CaseGenerationError(Exception):
    """
    Raised for unrecoverable case-generation failures.

    Examples:
      - malformed or missing sweep config sections
      - unsupported generation mode string
      - missing baseline values for a required design
      - unknown design, material, or load-case name
    """


# ---------------------------------------------------------------------------
# Typed dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CaseGenerationOptions:
    """
    Options controlling which cases are generated and how.

    Attributes:
        mode:                         Generation strategy:
                                      ``"baseline_only"``,
                                      ``"baseline_plus_one_factor_variation"``,
                                      or ``"full_factorial"``.
        include_baseline_case:        Always include the pure-baseline case even in
                                      one-factor or full-factorial modes.
        use_first_pass_values:        Use ``first_pass_parameters`` / ``first_pass_values_mm``
                                      instead of the full sweep value lists.
        stage_name:                   Named stage from ``staged_execution`` in
                                      ``sweep_config.yaml`` (for filtering).
        enabled_designs_override:     If set, overrides ``global_defaults.enabled_designs``.
        enabled_materials_override:   If set, overrides ``materials.enabled_materials``.
        enabled_loadcases_override:   If set, overrides ``loadcases.enabled_loadcases``.
        max_case_count:               Hard limit on generated cases (truncate with warning).
        sort_cases:                   Sort the final list by case ID for determinism.
    """

    mode: str = "baseline_plus_one_factor_variation"
    include_baseline_case: bool = True
    use_first_pass_values: bool = False
    stage_name: str | None = None
    enabled_designs_override: list[str] | None = None
    enabled_materials_override: list[str] | None = None
    enabled_loadcases_override: list[str] | None = None
    max_case_count: int | None = None
    sort_cases: bool = True


@dataclass
class GeneratedCaseSet:
    """
    Output of the case-generation pipeline.

    Attributes:
        cases:    Validated ``CaseDefinition`` objects ready for orchestration.
        warnings: Non-fatal notes (skipped invalid combos, deduplication, etc.).
        metadata: Summary counts, mode, timings.
    """

    cases: list[CaseDefinition] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "total_cases": len(self.cases),
            "case_ids":    [c.case_id for c in self.cases],
            "warnings":    self.warnings,
            "metadata":    self.metadata,
        }


# ---------------------------------------------------------------------------
# Sweep-config loading helpers
# ---------------------------------------------------------------------------

def load_sweep_config(
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """
    Load and return the raw ``sweep_config.yaml`` dictionary.

    Args:
        project_root: Project root (auto-detected if None).

    Returns:
        Raw dict from ``sweep_config.yaml``.

    Raises:
        CaseGenerationError: if the file cannot be loaded or parsed.
    """
    try:
        from utils.config_loader import load_pipeline_config
        cfg = load_pipeline_config(project_root)
        return dict(cfg.sweep_config)
    except Exception as exc:
        raise CaseGenerationError(
            f"Could not load sweep_config.yaml: {exc}"
        ) from exc


def get_enabled_design_names(
    sweep_config: Mapping[str, Any],
    override: list[str] | None = None,
) -> list[str]:
    """
    Return the list of enabled design names for case generation.

    Args:
        sweep_config: Raw sweep config dict.
        override:     If set, returns this list unchanged (after lower-casing).

    Returns:
        List of design name strings (e.g. ``["reentrant", "rotating_square"]``).
    """
    if override is not None:
        return [d.lower().strip() for d in override]
    global_defaults = sweep_config.get("global_defaults", {})
    enabled: list[str] = global_defaults.get("enabled_designs", [])
    # Also cross-check the design_sweeps section — skip disabled designs
    design_sweeps = sweep_config.get("design_sweeps", {})
    result: list[str] = []
    for name in enabled:
        key = name.lower().replace("-", "_").replace(" ", "_")
        ds = design_sweeps.get(key, {})
        if ds.get("enabled", True):
            result.append(key)
    return result


def get_enabled_material_names(
    sweep_config: Mapping[str, Any],
    override: list[str] | None = None,
) -> list[str]:
    """
    Return the list of enabled material names for case generation.

    Args:
        sweep_config: Raw sweep config dict.
        override:     If set, returns this list (stripped).

    Returns:
        List of material name strings matching ``materials.yaml`` keys.
    """
    if override is not None:
        return [m.strip() for m in override]
    materials_cfg = sweep_config.get("materials", {})
    return list(materials_cfg.get("enabled_materials", []))


def get_enabled_loadcase_names(
    sweep_config: Mapping[str, Any],
    override: list[str] | None = None,
) -> list[str]:
    """
    Return the list of enabled load-case names for case generation.

    Args:
        sweep_config: Raw sweep config dict.
        override:     If set, returns this list (stripped).

    Returns:
        List of load-case key strings matching ``loadcases.yaml`` entries.
    """
    if override is not None:
        return [lc.strip() for lc in override]
    lc_cfg = sweep_config.get("loadcases", {})
    return list(lc_cfg.get("enabled_loadcases", []))


# ---------------------------------------------------------------------------
# Design-parameter builders
# ---------------------------------------------------------------------------

def _build_reentrant_parameters(
    data: Mapping[str, Any],
) -> ReentrantParameters:
    """
    Build a ``ReentrantParameters`` object from a parameter dict.

    Args:
        data: Dict with keys ``cell_size``, ``wall_thickness``,
              ``reentrant_angle_deg``.

    Returns:
        ``ReentrantParameters`` (not yet validated).
    """
    return ReentrantParameters(
        cell_size=float(data["cell_size"]),
        wall_thickness=float(data["wall_thickness"]),
        reentrant_angle_deg=float(data["reentrant_angle_deg"]),
    )


def _build_rotating_square_parameters(
    data: Mapping[str, Any],
) -> RotatingSquareParameters:
    """
    Build a ``RotatingSquareParameters`` object from a parameter dict.

    Args:
        data: Dict with keys ``cell_size``, ``rotation_angle_deg``,
              ``hinge_thickness``.

    Returns:
        ``RotatingSquareParameters`` (not yet validated).
    """
    return RotatingSquareParameters(
        cell_size=float(data["cell_size"]),
        rotation_angle_deg=float(data["rotation_angle_deg"]),
        hinge_thickness=float(data["hinge_thickness"]),
    )


def _build_tetrachiral_parameters(
    data: Mapping[str, Any],
) -> TetrachiralParameters:
    """
    Build a ``TetrachiralParameters`` object from a parameter dict.

    ``fillet_radius`` and ``fillet_radius_derived`` keys are silently ignored
    because fillet radius is a derived property of ``node_radius``.

    Args:
        data: Dict with keys ``cell_size``, ``node_radius``,
              ``ligament_thickness``.

    Returns:
        ``TetrachiralParameters`` (not yet validated).
    """
    return TetrachiralParameters(
        cell_size=float(data["cell_size"]),
        node_radius=float(data["node_radius"]),
        ligament_thickness=float(data["ligament_thickness"]),
    )


def build_design_parameters(
    design_name: str,
    parameter_values: Mapping[str, Any],
) -> DesignParameterSet:
    """
    Dispatch to the correct parameter builder based on design name.

    Args:
        design_name:      One of ``"reentrant"``, ``"rotating_square"``,
                          ``"tetrachiral"``.
        parameter_values: Dict of parameter name → value.

    Returns:
        Typed design parameter object.

    Raises:
        CaseGenerationError: if ``design_name`` is not supported.
    """
    dispatch: dict[str, Any] = {
        "reentrant":       _build_reentrant_parameters,
        "rotating_square": _build_rotating_square_parameters,
        "tetrachiral":     _build_tetrachiral_parameters,
    }
    builder = dispatch.get(design_name.lower())
    if builder is None:
        raise CaseGenerationError(
            f"Unsupported design name '{design_name}'.  "
            f"Supported: {sorted(dispatch.keys())}."
        )
    return builder(parameter_values)


# ---------------------------------------------------------------------------
# Baseline and sweep-value extraction helpers
# ---------------------------------------------------------------------------

def get_design_baseline_values(
    design_name: str,
    sweep_config: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Return the baseline parameter dict for a design from sweep config.

    Args:
        design_name:  Design key (e.g. ``"reentrant"``).
        sweep_config: Raw sweep config dict.

    Returns:
        Flat dict of baseline parameter values.

    Raises:
        CaseGenerationError: if the design or baseline block is missing.
    """
    design_sweeps = sweep_config.get("design_sweeps", {})
    ds = design_sweeps.get(design_name)
    if ds is None:
        raise CaseGenerationError(
            f"Design '{design_name}' not found in design_sweeps section of "
            "sweep_config.yaml."
        )
    baseline = ds.get("baseline")
    if not baseline:
        raise CaseGenerationError(
            f"No 'baseline' block found for design '{design_name}' in "
            "sweep_config.yaml."
        )
    return dict(baseline)


def get_design_sweep_values(
    design_name: str,
    sweep_config: Mapping[str, Any],
    use_first_pass_values: bool = False,
) -> dict[str, list[Any]]:
    """
    Return per-parameter sweep value lists for a design.

    Drops derived parameters (``fillet_radius``, ``fillet_radius_derived``)
    silently.

    Args:
        design_name:          Design key.
        sweep_config:         Raw sweep config dict.
        use_first_pass_values: If True, use ``first_pass_parameters`` instead
                               of ``sweep_parameters``.

    Returns:
        Dict mapping parameter name → list of values to sweep.
    """
    design_sweeps = sweep_config.get("design_sweeps", {})
    ds = design_sweeps.get(design_name, {})
    section_key = "first_pass_parameters" if use_first_pass_values else "sweep_parameters"
    raw_params = ds.get(section_key, {})

    result: dict[str, list[Any]] = {}
    for param_name, param_cfg in raw_params.items():
        if param_name in _TETRACHIRAL_DERIVED_PARAMS:
            logger.debug(
                "Skipping derived parameter '%s' for design '%s'.",
                param_name, design_name,
            )
            continue
        if isinstance(param_cfg, dict):
            values = param_cfg.get("values", [])
        elif isinstance(param_cfg, list):
            values = param_cfg
        else:
            values = [param_cfg]
        result[str(param_name)] = list(values)
    return result


# ---------------------------------------------------------------------------
# Plate thickness / material / load-case helpers
# ---------------------------------------------------------------------------

def get_plate_thickness_values(
    sweep_config: Mapping[str, Any],
    use_first_pass_values: bool = False,
) -> list[float]:
    """
    Return the plate thickness values to sweep.

    Args:
        sweep_config:         Raw sweep config dict.
        use_first_pass_values: If True, use ``first_pass_values_mm``.

    Returns:
        List of plate thickness floats (mm).
    """
    pt_cfg = sweep_config.get("plate_thickness", {})
    key = "first_pass_values_mm" if use_first_pass_values else "values_mm"
    values = pt_cfg.get(key, [])
    if not values:
        baseline = float(pt_cfg.get("baseline_mm", 2.5))
        return [baseline]
    return [float(v) for v in values]


def _get_plate_thickness_baseline(sweep_config: Mapping[str, Any]) -> float:
    """Return the single baseline plate thickness (mm)."""
    pt_cfg = sweep_config.get("plate_thickness", {})
    return float(pt_cfg.get("baseline_mm", 2.5))


def resolve_material_objects(
    material_names: Sequence[str],
    project_root: str | Path | None = None,
) -> list[Any]:
    """
    Resolve material name strings to ``MaterialRecord`` objects.

    Args:
        material_names: Names matching ``materials.yaml`` keys.
        project_root:   Project root for config resolution.

    Returns:
        List of material record objects.

    Raises:
        CaseGenerationError: if no materials can be loaded.
    """
    try:
        from simulation.materials import load_material_library
        lib = load_material_library(project_root=project_root)
    except Exception as exc:
        raise CaseGenerationError(
            f"Could not load material library: {exc}"
        ) from exc

    materials: list[Any] = []
    for name in material_names:
        try:
            mat = lib.get(name)
            if mat is None:
                raise KeyError(name)
            materials.append(mat)
        except Exception as exc:
            raise CaseGenerationError(
                f"Material '{name}' not found in materials.yaml: {exc}"
            ) from exc
    return materials


def resolve_loadcase_objects(
    loadcase_names: Sequence[str],
    project_root: str | Path | None = None,
) -> list[Any]:
    """
    Resolve load-case name strings to ``LoadCaseRecord`` objects.

    Args:
        loadcase_names: Keys matching ``loadcases.yaml`` entries.
        project_root:   Project root for config resolution.

    Returns:
        List of load-case record objects.

    Raises:
        CaseGenerationError: if any load case cannot be resolved.
    """
    try:
        from simulation.loadcases import load_loadcase_library
        lib = load_loadcase_library(project_root=project_root)
    except Exception as exc:
        raise CaseGenerationError(
            f"Could not load load-case library: {exc}"
        ) from exc

    cases: list[Any] = []
    for name in loadcase_names:
        try:
            lc = lib.get(name)
            if lc is None:
                raise KeyError(name)
            cases.append(lc)
        except Exception as exc:
            raise CaseGenerationError(
                f"Load case '{name}' not found in loadcases.yaml: {exc}"
            ) from exc
    return cases


# ---------------------------------------------------------------------------
# Deterministic case-id helper
# ---------------------------------------------------------------------------

def make_case_id(
    design_name: str,
    parameter_values: Mapping[str, Any],
    plate_thickness: float,
    material_name: str,
    loadcase_name: str,
    lattice_repeats_x: int = 5,
    lattice_repeats_y: int = 3,
) -> str:
    """
    Generate a deterministic, filesystem-friendly case ID.

    Format example::

        reentrant_pt2p5_ti64_axialcompression_a1b2c3d4

    The trailing 8-character hex suffix is derived from a SHA-256 hash of
    all input parameters so that any change to parameter values produces a
    different ID.

    Args:
        design_name:       Design type key (e.g. ``"reentrant"``).
        parameter_values:  Flat dict of all design parameters.
        plate_thickness:   Plate thickness in mm.
        material_name:     Material name string.
        loadcase_name:     Load-case key string.
        lattice_repeats_x: Lattice X repeat count (always 5 in v1).
        lattice_repeats_y: Lattice Y repeat count (always 3 in v1).

    Returns:
        Deterministic case ID string (lowercase, underscores, no spaces).
    """
    # Build a compact canonical token string for hashing
    sorted_params = sorted(
        (k, v) for k, v in parameter_values.items()
        if k not in _TETRACHIRAL_DERIVED_PARAMS
    )
    canonical = (
        f"{design_name}|"
        + "|".join(f"{k}={v!r}" for k, v in sorted_params)
        + f"|pt={plate_thickness:.4f}"
        + f"|mat={material_name}"
        + f"|lc={loadcase_name}"
        + f"|lat={lattice_repeats_x}x{lattice_repeats_y}"
    )
    hd = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    short_hash = "".join([c for i, c in enumerate(hd) if i < 8])

    # Human-readable fragments
    mat_full = (
        material_name.lower()
        .replace("-", "")
        .replace(" ", "")
        .replace(".", "")
    )
    mat_slug = "".join([c for i, c in enumerate(mat_full) if i < 8])
    lc_full = (
        loadcase_name.lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )
    lc_slug = "".join([c for i, c in enumerate(lc_full) if i < 16])
    thickness_slug = f"pt{plate_thickness:.1f}".replace(".", "p")

    return (
        f"{design_name}_{thickness_slug}_{mat_slug}_{lc_slug}_{short_hash}"
    )


# ---------------------------------------------------------------------------
# Case-construction helper
# ---------------------------------------------------------------------------

def build_case_definition(
    design_name: str,
    parameter_values: Mapping[str, Any],
    plate_thickness: float,
    material: Any,
    loadcase: Any,
    project_root: str | Path | None = None,
) -> CaseDefinition | None:
    """
    Build and validate one ``CaseDefinition`` from typed inputs.

    Returns ``None`` (with a debug log) if validation fails so callers can
    skip invalid parameter combinations gracefully.

    Args:
        design_name:      Design key string.
        parameter_values: Flat dict of design parameter values.
        plate_thickness:  Extrusion depth in mm.
        material:         ``MaterialRecord`` object.
        loadcase:         ``LoadCaseRecord`` object.
        project_root:     Project root (optional, used for STEP path resolution).

    Returns:
        ``CaseDefinition`` or ``None`` if validation fails.
    """
    # Build typed design-parameter object
    try:
        params = build_design_parameters(design_name, parameter_values)
    except (KeyError, TypeError, ValueError) as exc:
        logger.debug(
            "Parameter construction failed for design='%s', params=%s: %s",
            design_name, parameter_values, exc,
        )
        return None

    # Validate design parameters in isolation
    try:
        params.validate()
    except ValueError as exc:
        logger.debug(
            "Parameter validation failed for design='%s', params=%s: %s",
            design_name, parameter_values, exc,
        )
        return None

    # Material name for ID and metadata
    material_name: str = getattr(material, "name", str(material))
    loadcase_name: str = (
        getattr(loadcase, "load_case_id", None)
        or getattr(loadcase, "load_case_type", str(loadcase))
    )
    loadcase_val = getattr(loadcase_name, "value", None)
    if loadcase_val is not None:
        loadcase_name = loadcase_val

    case_id = make_case_id(
        design_name=design_name,
        parameter_values=parameter_values,
        plate_thickness=plate_thickness,
        material_name=material_name,
        loadcase_name=str(loadcase_name),
    )

    try:
        cd = CaseDefinition(
            case_id=case_id,
            design_parameters=params,
            plate_thickness=plate_thickness,
            material=material.to_case_schema_material() if hasattr(material, "to_case_schema_material") else material,
            load_case=loadcase.to_case_schema_loadcase() if hasattr(loadcase, "to_case_schema_loadcase") else loadcase,
            lattice_repeats_x=5,
            lattice_repeats_y=3,
        )
        if hasattr(cd, "validate"):
            cd.validate()
        return cd
    except Exception as exc:
        print(f"Exception for {case_id}: {exc}")
        logger.debug(
            "CaseDefinition construction/validation failed for '%s': %s",
            case_id, exc,
        )
        return None


# ---------------------------------------------------------------------------
# Generation strategy implementations
# ---------------------------------------------------------------------------

def generate_baseline_cases(
    sweep_config: Mapping[str, Any],
    design_names: Sequence[str],
    materials: Sequence[Any],
    loadcases: Sequence[Any],
    project_root: str | Path | None = None,
) -> tuple[list[CaseDefinition], list[str]]:
    """
    Generate one case per design × material × load-case at baseline values.

    Plate thickness is set to the configured ``plate_thickness.baseline_mm``.

    Args:
        sweep_config: Raw sweep config dict.
        design_names: Enabled design names.
        materials:    Resolved material objects.
        loadcases:    Resolved load-case objects.
        project_root: Project root (optional).

    Returns:
        Tuple ``(cases, warnings)``.
    """
    baseline_thickness = _get_plate_thickness_baseline(sweep_config)
    cases: list[CaseDefinition] = []
    warnings: list[str] = []

    for design_name in design_names:
        try:
            baseline = get_design_baseline_values(design_name, sweep_config)
        except CaseGenerationError as exc:
            warnings.append(str(exc))
            continue

        for mat in materials:
            for lc in loadcases:
                cd = build_case_definition(
                    design_name=design_name,
                    parameter_values=baseline,
                    plate_thickness=baseline_thickness,
                    material=mat,
                    loadcase=lc,
                    project_root=project_root,
                )
                if cd is not None:
                    cases.append(cd)
                else:
                    warnings.append(
                        f"Baseline case skipped (invalid params): "
                        f"design={design_name}, mat={getattr(mat, 'name', mat)}, "
                        f"lc={getattr(lc, 'load_case_id', lc)}."
                    )

    return cases, warnings


def generate_baseline_plus_one_factor_variation_cases(
    sweep_config: Mapping[str, Any],
    design_names: Sequence[str],
    materials: Sequence[Any],
    loadcases: Sequence[Any],
    include_baseline: bool = True,
    use_first_pass_values: bool = False,
    project_root: str | Path | None = None,
) -> tuple[list[CaseDefinition], list[str]]:
    """
    Generate baseline cases plus one-factor-at-a-time parameter variations.

    Strategy per design:
      1. Optionally emit the pure-baseline case.
      2. For each geometric parameter, vary it across its sweep values while
         all others hold at baseline.  Plate thickness stays at baseline.
      3. Vary plate thickness across its sweep values while all geometric
         parameters stay at baseline.
      4. Combine each combination with every enabled material × load-case.

    Args:
        sweep_config:         Raw sweep config dict.
        design_names:         Enabled design names.
        materials:            Resolved material objects.
        loadcases:            Resolved load-case objects.
        include_baseline:     Include the pure-baseline case.
        use_first_pass_values: Use first-pass sweep value lists.
        project_root:         Project root (optional).

    Returns:
        Tuple ``(cases, warnings)``.
    """
    baseline_thickness = _get_plate_thickness_baseline(sweep_config)
    thickness_sweep  = get_plate_thickness_values(
        sweep_config, use_first_pass_values=use_first_pass_values
    )
    cases: list[CaseDefinition] = []
    warnings: list[str] = []

    for design_name in design_names:
        try:
            baseline = get_design_baseline_values(design_name, sweep_config)
            sweep_vals = get_design_sweep_values(
                design_name, sweep_config,
                use_first_pass_values=use_first_pass_values,
            )
        except CaseGenerationError as exc:
            warnings.append(str(exc))
            continue

        # Collect parameter combinations for this design
        param_combos: list[dict[str, Any]] = []

        # --- Pure baseline ---
        if include_baseline:
            param_combos.append(dict(baseline))

        # --- One-factor-at-a-time over geometric parameters ---
        for param_name, param_values in sweep_vals.items():
            for pval in param_values:
                combo = dict(baseline)
                combo[param_name] = pval
                # Skip if identical to baseline (avoids duplicates)
                if combo == baseline:
                    continue
                param_combos.append(combo)

        # --- One-factor-at-a-time over plate thickness ---
        thickness_combos: list[float] = [
            t for t in thickness_sweep if t != baseline_thickness
        ]

        # Cross with materials × load-cases
        for combo in param_combos:
            for mat in materials:
                for lc in loadcases:
                    cd = build_case_definition(
                        design_name=design_name,
                        parameter_values=combo,
                        plate_thickness=baseline_thickness,
                        material=mat,
                        loadcase=lc,
                        project_root=project_root,
                    )
                    if cd is not None:
                        cases.append(cd)
                    else:
                        warnings.append(
                            f"Combo skipped (invalid): design={design_name}, "
                            f"params={combo}, pt={baseline_thickness}."
                        )

        # Plate-thickness variations at baseline geometric params
        for pt in thickness_combos:
            for mat in materials:
                for lc in loadcases:
                    cd = build_case_definition(
                        design_name=design_name,
                        parameter_values=dict(baseline),
                        plate_thickness=pt,
                        material=mat,
                        loadcase=lc,
                        project_root=project_root,
                    )
                    if cd is not None:
                        cases.append(cd)
                    else:
                        warnings.append(
                            f"Thickness variation skipped (invalid): "
                            f"design={design_name}, pt={pt}."
                        )

    return cases, warnings


def generate_full_factorial_cases(
    sweep_config: Mapping[str, Any],
    design_names: Sequence[str],
    materials: Sequence[Any],
    loadcases: Sequence[Any],
    include_baseline: bool = True,
    use_first_pass_values: bool = False,
    max_case_count: int | None = None,
    project_root: str | Path | None = None,
) -> tuple[list[CaseDefinition], list[str]]:
    """
    Generate full Cartesian product of all sweep parameter combinations.

    For each design: itertools.product across all sweep-parameter value lists,
    then × plate thickness × materials × load-cases.

    Warning: this can produce O(100–1000) cases.  ``max_case_count`` is
    honoured by truncating after that limit with a warning.

    Args:
        sweep_config:          Raw sweep config dict.
        design_names:          Enabled design names.
        materials:             Resolved material objects.
        loadcases:             Resolved load-case objects.
        include_baseline:      Prepend the baseline case for each design.
        use_first_pass_values: Use first-pass value lists.
        max_case_count:        Hard limit on generated cases (truncate with warning).
        project_root:          Project root (optional).

    Returns:
        Tuple ``(cases, warnings)``.
    """
    baseline_pt = _get_plate_thickness_baseline(sweep_config)
    thickness_vals = get_plate_thickness_values(
        sweep_config, use_first_pass_values=use_first_pass_values
    )
    cases: list[CaseDefinition] = []
    warnings: list[str] = []


    for design_name in design_names:
        try:
            baseline = get_design_baseline_values(design_name, sweep_config)
            sweep_vals = get_design_sweep_values(
                design_name, sweep_config,
                use_first_pass_values=use_first_pass_values,
            )
        except CaseGenerationError as exc:
            warnings.append(str(exc))
            continue

        # Build parameter names and value lists for itertools.product
        param_names: list[str] = list(sweep_vals.keys())
        param_value_lists: list[list[Any]] = [sweep_vals[k] for k in param_names]

        # Emit baseline first
        if include_baseline:
            for mat in materials:
                for lc in loadcases:
                    for pt in [baseline_pt]:
                        if max_case_count is not None:
                            limit: int = int(str(max_case_count))
                            if len(cases) >= limit:
                                warnings.append(
                                    f"max_case_count={limit} reached; "
                                    "remaining cases truncated."
                                )
                                return cases, warnings
                        cd = build_case_definition(
                            design_name=design_name,
                            parameter_values=dict(baseline),
                            plate_thickness=pt,
                            material=mat,
                            loadcase=lc,
                            project_root=project_root,
                        )
                        if cd is not None:
                            cases.append(cd)

        # Full factorial over sweep parameters
        if param_value_lists:
            for combo_vals in itertools.product(*param_value_lists):
                combo: dict[str, Any] = dict(baseline)
                for pname, pval in zip(param_names, combo_vals):
                    combo[pname] = pval

                for pt in thickness_vals:
                    for mat in materials:
                        for lc in loadcases:
                            if max_case_count is not None:
                                limit: int = int(str(max_case_count))
                                if len(cases) >= limit:
                                    warnings.append(
                                        f"max_case_count={limit} reached; "
                                        "remaining cases truncated."
                                    )
                                    return cases, warnings
                            cd = build_case_definition(
                                design_name=design_name,
                                parameter_values=combo,
                                plate_thickness=pt,
                                material=mat,
                                loadcase=lc,
                                project_root=project_root,
                            )
                            if cd is not None:
                                cases.append(cd)

    return cases, warnings


# ---------------------------------------------------------------------------
# Stage filtering helper
# ---------------------------------------------------------------------------

def filter_cases_by_stage(
    case_definitions: Sequence[CaseDefinition],
    stage_name: str | None,
    project_root: str | Path | None = None,
) -> list[CaseDefinition]:
    """
    Filter a case list to only those belonging to a named execution stage.

    Reads the ``staged_execution`` section of ``sweep_config.yaml`` to
    identify which materials and load-case types belong to this stage.

    Args:
        case_definitions: Full list of case definitions to filter.
        stage_name:       Named stage key (e.g. ``"stage_1"``).
                          If None, all cases are returned unchanged.
        project_root:     Project root for config resolution.

    Returns:
        Filtered list.  Falls back to all cases if stage is unresolvable.
    """
    if stage_name is None:
        return list(case_definitions)

    stage_materials:  set[str] = set()
    stage_loadcases:  set[str] = set()

    try:
        sweep_config = load_sweep_config(project_root)
        se = sweep_config.get("staged_execution", {})
        # Stage keys in the YAML are like "stage_1", "stage_2", ...
        # Also accept the stage name value (e.g. "geometry_and_coarse_screen")
        stage_cfg: dict[str, Any] | None = None
        for sk, sv in se.items():
            if sk == stage_name or (
                isinstance(sv, dict) and sv.get("name") == stage_name
            ):
                stage_cfg = sv
                break

        if stage_cfg is None:
            logger.warning(
                "Stage '%s' not found in staged_execution config.  "
                "No filtering applied.",
                stage_name,
            )
            return list(case_definitions)

        stage_materials  = {m.lower() for m in (stage_cfg.get("materials", []) if stage_cfg is not None else [])}
        stage_loadcases  = {lc.lower() for lc in (stage_cfg.get("loadcases", []) if stage_cfg is not None else [])}

    except Exception as exc:
        logger.warning(
            "Could not load stage config for '%s': %s.  "
            "No filtering applied.",
            stage_name, exc,
        )
        return list(case_definitions)

    filtered: list[CaseDefinition] = []
    for cd in case_definitions:
        mat_name = getattr(cd.material, "name", str(cd.material)).lower()
        lc_name = (
            getattr(cd.load_case, "load_case_id", None)
            or str(getattr(cd.load_case, "load_case_type", ""))
        ).lower()
        # Strip any enum suffix like "<LoadCaseType.AXIAL_COMPRESSION: ...>"
        if ":" in lc_name:
            lc_name = lc_name.split(":")[-1].strip(" >").strip("'")

        mat_ok = not stage_materials or mat_name in stage_materials
        lc_ok  = not stage_loadcases or lc_name in stage_loadcases
        if mat_ok and lc_ok:
            filtered.append(cd)

    logger.info(
        "Stage filter '%s': %d/%d cases selected.",
        stage_name, len(filtered), len(case_definitions),
    )
    return filtered


# ---------------------------------------------------------------------------
# Deduplication helper
# ---------------------------------------------------------------------------

def deduplicate_cases(
    case_definitions: Sequence[CaseDefinition],
) -> list[CaseDefinition]:
    """
    Remove cases with duplicate ``case_id`` values, preserving insertion order.

    Args:
        case_definitions: Possibly-redundant list of case definitions.

    Returns:
        Deduplicated list.  First occurrence of each case_id is kept.
    """
    seen: dict[str, bool] = {}
    result: list[CaseDefinition] = []
    for cd in case_definitions:
        if cd.case_id not in seen:
            seen[cd.case_id] = True
            result.append(cd)
        else:
            logger.debug("Duplicate case_id '%s' dropped.", cd.case_id)
    dropped = len(case_definitions) - len(result)
    if dropped:
        logger.info(
            "Deduplication: %d duplicate case(s) removed; %d unique cases remain.",
            dropped, len(result),
        )
    return result


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def summarize_generated_cases(
    case_definitions: Sequence[CaseDefinition],
) -> dict[str, Any]:
    """
    Produce a human-readable summary dict of the generated case set.

    Args:
        case_definitions: Generated case definitions.

    Returns:
        Dict with total count, by-design/material/load-case breakdowns,
        and min/max plate thickness.
    """
    by_design:    dict[str, int] = {}
    by_material:  dict[str, int] = {}
    by_load_case: dict[str, int] = {}
    thicknesses:  list[float]    = []

    for cd in case_definitions:
        dt  = cd.design_type.value
        mat = getattr(cd.material, "name", str(cd.material))
        lc  = getattr(
            cd.load_case, "load_case_id",
            str(getattr(cd.load_case, "load_case_type", "?"))
        )
        pt  = cd.plate_thickness

        by_design[dt]       = by_design.get(dt, 0) + 1
        by_material[mat]    = by_material.get(mat, 0) + 1
        by_load_case[lc]    = by_load_case.get(lc, 0) + 1
        thicknesses.append(pt)

    return {
        "total_cases":       len(case_definitions),
        "by_design":         by_design,
        "by_material":       by_material,
        "by_load_case":      by_load_case,
        "plate_thickness_min_mm": min(thicknesses) if thicknesses else None,
        "plate_thickness_max_mm": max(thicknesses) if thicknesses else None,
    }


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_cases(
    options: CaseGenerationOptions | None = None,
    project_root: str | Path | None = None,
) -> GeneratedCaseSet:
    """
    Generate the full set of ``CaseDefinition`` objects for a pipeline sweep.

    This is the canonical entry point for case generation. Behavior:
        1.  Load configuration (sweep + pipeline config).
        2.  Validate and resolve options.
        3.  Determine enabled designs, materials, and load cases.
        4.  Resolve material and load-case objects from their libraries.
        5.  Run the requested generation strategy.
        6.  Optionally filter by stage (``options.stage_name``).
        7.  Deduplicate case IDs.
        8.  Optionally sort deterministically by case ID.
        9.  Enforce ``max_case_count`` if provided.
        10. Return a ``GeneratedCaseSet``.

    Args:
        options:      Generation options (defaults used if None).
        project_root: Project root directory (auto-detected if None).

    Returns:
        ``GeneratedCaseSet`` with validated cases, warnings, and metadata.

    Raises:
        CaseGenerationError: if no cases can be generated due to config errors.
    """
    if options is None:
        options = CaseGenerationOptions()

    if options.mode not in _VALID_MODES:
        raise CaseGenerationError(
            f"Unsupported generation mode '{options.mode}'.  "
            f"Valid modes: {sorted(_VALID_MODES)}."
        )

    warnings: list[str] = []
    meta: dict[str, Any] = {"mode": options.mode}

    # ------------------------------------------------------------------
    # Load sweep config
    # ------------------------------------------------------------------
    try:
        sweep_config = load_sweep_config(project_root)
    except CaseGenerationError:
        raise
    except Exception as exc:
        raise CaseGenerationError(
            f"Failed to load sweep configuration: {exc}"
        ) from exc

    # ------------------------------------------------------------------
    # Resolve enabled designs / materials / load cases
    # ------------------------------------------------------------------
    design_names  = get_enabled_design_names(
        sweep_config, override=options.enabled_designs_override
    )
    material_names = get_enabled_material_names(
        sweep_config, override=options.enabled_materials_override
    )
    loadcase_names = get_enabled_loadcase_names(
        sweep_config, override=options.enabled_loadcases_override
    )

    if not design_names:
        raise CaseGenerationError("No enabled designs found in sweep config.")
    if not material_names:
        raise CaseGenerationError("No enabled materials found in sweep config.")
    if not loadcase_names:
        raise CaseGenerationError("No enabled load cases found in sweep config.")

    # ------------------------------------------------------------------
    # Resolve objects from libraries
    # ------------------------------------------------------------------
    try:
        materials = resolve_material_objects(material_names, project_root)
    except CaseGenerationError as exc:
        raise CaseGenerationError(
            f"Material resolution failed: {exc}"
        ) from exc

    try:
        loadcases = resolve_loadcase_objects(loadcase_names, project_root)
    except CaseGenerationError as exc:
        raise CaseGenerationError(
            f"Load-case resolution failed: {exc}"
        ) from exc

    meta["enabled_designs"]   = design_names
    meta["enabled_materials"] = material_names
    meta["enabled_loadcases"] = loadcase_names

    # ------------------------------------------------------------------
    # Run generation strategy
    # ------------------------------------------------------------------
    cases: list[CaseDefinition] = []
    gen_warnings: list[str] = []

    if options.mode == "baseline_only":
        cases, gen_warnings = generate_baseline_cases(
            sweep_config=sweep_config,
            design_names=design_names,
            materials=materials,
            loadcases=loadcases,
            project_root=project_root,
        )

    elif options.mode == "baseline_plus_one_factor_variation":
        cases, gen_warnings = generate_baseline_plus_one_factor_variation_cases(
            sweep_config=sweep_config,
            design_names=design_names,
            materials=materials,
            loadcases=loadcases,
            include_baseline=options.include_baseline_case,
            use_first_pass_values=options.use_first_pass_values,
            project_root=project_root,
        )

    elif options.mode == "full_factorial":
        cases, gen_warnings = generate_full_factorial_cases(
            sweep_config=sweep_config,
            design_names=design_names,
            materials=materials,
            loadcases=loadcases,
            include_baseline=options.include_baseline_case,
            use_first_pass_values=options.use_first_pass_values,
            max_case_count=options.max_case_count,
            project_root=project_root,
        )

    warnings.extend(gen_warnings)

    # ------------------------------------------------------------------
    # Stage filtering
    # ------------------------------------------------------------------
    if options.stage_name:
        before = len(cases)
        cases = filter_cases_by_stage(cases, options.stage_name, project_root)
        after = len(cases)
        if before != after:
            warnings.append(
                f"Stage filter '{options.stage_name}': "
                f"{before - after} cases removed, {after} remain."
            )

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------
    pre_dedup = len(cases)
    cases = deduplicate_cases(cases)
    if pre_dedup != len(cases):
        warnings.append(
            f"Deduplication removed {pre_dedup - len(cases)} duplicate case(s)."
        )

    # ------------------------------------------------------------------
    # Sort
    # ------------------------------------------------------------------
    if options.sort_cases:
        cases.sort(key=lambda cd: cd.case_id)

    # ------------------------------------------------------------------
    # max_case_count enforcement (final gate)
    # ------------------------------------------------------------------
    max_cases_val = options.max_case_count
    if max_cases_val is not None and len(cases) > max_cases_val:
        truncated = len(cases) - max_cases_val
        top_c = []
        for i, c in enumerate(cases):
            if i >= max_cases_val:
                break
            top_c.append(c)
        cases = top_c
        warnings.append(
            f"Case list truncated: {truncated} case(s) dropped to enforce "
            f"max_case_count={max_cases_val}."
        )

    # ------------------------------------------------------------------
    # Warn if large
    # ------------------------------------------------------------------
    filters_cfg = sweep_config.get("filters", {})
    threshold = int(filters_cfg.get("max_total_cases_warning_threshold", 200))
    if len(cases) > threshold:
        warnings.append(
            f"Total generated case count ({len(cases)}) exceeds warning "
            f"threshold ({threshold}).  Consider staged execution or "
            f"first-pass values to reduce the sweep size."
        )

    # ------------------------------------------------------------------
    # Summary metadata
    # ------------------------------------------------------------------
    meta.update(summarize_generated_cases(cases))
    meta["stage_name"]          = options.stage_name
    meta["use_first_pass_values"] = options.use_first_pass_values
    meta["max_case_count"]      = options.max_case_count

    logger.info(
        "generate_cases: mode='%s', %d case(s) generated "
        "(designs=%s, mats=%s, lcs=%s).",
        options.mode, len(cases),
        design_names, material_names, loadcase_names,
    )

    return GeneratedCaseSet(cases=cases, warnings=warnings, metadata=meta)


# ---------------------------------------------------------------------------
# Strict wrapper
# ---------------------------------------------------------------------------

def require_generated_cases(
    options: CaseGenerationOptions | None = None,
    project_root: str | Path | None = None,
) -> GeneratedCaseSet:
    """
    Generate cases and raise ``CaseGenerationError`` if none are produced.

    Args:
        options:      Generation options.
        project_root: Project root.

    Returns:
        ``GeneratedCaseSet`` with at least one case.

    Raises:
        CaseGenerationError: if generation fails or produces zero cases.
    """
    result = generate_cases(options=options, project_root=project_root)
    if not result.cases:
        raise CaseGenerationError(
            "Case generation produced zero cases.  "
            "Check sweep_config.yaml enabled_designs, enabled_materials, "
            "and enabled_loadcases entries."
        )
    return result
