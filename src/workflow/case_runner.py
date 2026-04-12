"""
src/workflow/case_runner.py
============================
Single-case execution module for the auxetic plate pipeline.

This module runs one ``CaseDefinition`` end-to-end through the version-1
pipeline stages:

    status setup
    → unit-cell construction
    → lattice assembly
    → 3D extrusion
    → validation
    → meshing
    → solver input export
    → (optional) solver execution
    → postprocessing
    → engineering metrics + fatigue proxy
    → CaseResult assembly

PIPELINE POSITION:
    orchestrator.py  →  [THIS MODULE]  →  CaseRunSummary
                                       →  runs/<case_id>/  (all artifacts)
                                       →  status files

ARCHITECTURAL DECISION — orchestrates, never reimplements:
    This module calls existing sub-modules directly and in sequence.  It
    does not reimplement geometry, meshing, solving, or postprocessing logic.
    All domain logic stays in its respective module.  This file is the
    "glue" layer only.

ARCHITECTURAL DECISION — honest partial results:
    If a stage fails (mesh, solver, postprocess), the failure is recorded in
    ``CaseRunSummary.error_message`` and the case is marked FAILED.  Partial
    results from earlier stages are not discarded — they remain on disk and
    are referenced in ``CaseRunArtifacts``.  This allows debugging without
    re-running expensive upstream stages.

ARCHITECTURAL DECISION — solver execution is opt-in:
    ``CaseRunOptions.run_solver = False`` is the default, matching
    ``base_config.yaml solver.run_solver_by_default: false``.  In export-only
    mode the pipeline runs fully (geometry → mesh → .inp file) and the
    ``CaseResult`` reflects that no FE results are available.  Downstream
    modules (postprocess, metrics) handle missing FE results gracefully with
    ``None`` metric values and explicit warnings.

ARCHITECTURAL DECISION — status updates are written at each stage boundary:
    ``status_tracker.mark_case_running(stage=...)`` is called with the current
    stage name before entering each major pipeline stage.  This gives the
    orchestrator real-time visibility into which stage a slow or stuck case is
    executing, and allows crash recovery to identify the last completed stage
    from the on-disk status file.

UNITS: consistent with project-wide convention (mm, N, MPa).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from workflow.case_schema import (
    CaseDefinition,
    CaseResult,
    CaseStatus,
    MetricSet,
    SolverStatus,
    MeshStatus,
)
from workflow.cache import (
    ensure_case_directories,
    geometry_signature,
    mesh_signature,
)
from workflow.status_tracker import (
    mark_case_running,
    mark_case_completed,
    mark_case_failed,
    mark_case_skipped,
    case_should_be_skipped,
    StatusTrackerError,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class CaseRunnerError(Exception):
    """
    Raised for unrecoverable case-level orchestration failures.

    Individual stage failures are represented as ``CaseRunSummary.success=False``
    with a structured error message.  This exception is only raised by the
    ``require_successful_case_run`` hard-fail wrapper.
    """


# ---------------------------------------------------------------------------
# Typed dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CaseRunOptions:
    """
    Configuration options controlling how a single case is executed.

    Attributes:
        skip_completed_cases:         Skip the case if it is already COMPLETED.
        write_case_config_snapshot:   Write case_config.json to the run directory.
        write_case_metadata_snapshot: Write metadata.json snapshot at completion.
        write_intermediate_geometry:  Attempt STEP export for each geometry stage.
        write_mesh_files:             Include mesh files in stage output.
        write_solver_input_files:     Write the CalculiX .inp input deck.
        run_solver:                   Execute CalculiX after exporting the deck.
        solver_timeout_seconds:       Wall-clock timeout for CalculiX (None = no limit).
        meshing_preset:               gmsh meshing preset name (e.g. ``"default"``).
        centered_extrusion:           Centre the solid about Z=0 when extruding.
        minimum_feature_size_mm:      Optional feature-size threshold for validation.
        strict_validation:            Raise on validation warnings (not just failures).
        continue_on_partial_postprocess: Continue if postprocess yields only partial results.
    """

    skip_completed_cases: bool = True
    write_case_config_snapshot: bool = True
    write_case_metadata_snapshot: bool = True
    write_intermediate_geometry: bool = True
    write_mesh_files: bool = True
    write_solver_input_files: bool = True
    run_solver: bool = False
    solver_timeout_seconds: int | None = None
    meshing_preset: str = "default"
    centered_extrusion: bool = True
    minimum_feature_size_mm: float | None = None
    strict_validation: bool = False
    continue_on_partial_postprocess: bool = True


@dataclass
class CaseRunArtifacts:
    """
    Registry of canonical artifact file paths produced by one case run.

    All paths are strings (or None if the artifact was not produced) for
    JSON serialisability.

    Attributes:
        run_directory:            ``runs/<case_id>/``
        geometry_directory:       ``runs/<case_id>/geometry/``
        mesh_directory:           ``runs/<case_id>/mesh/``
        solver_directory:         ``runs/<case_id>/solver/``
        results_directory:        ``runs/<case_id>/results/``
        case_config_file:         Case definition snapshot (JSON).
        metadata_file:            Run metadata snapshot (JSON).
        unit_cell_geometry_file:  2D unit-cell STEP (optional export).
        lattice_geometry_file:    2D lattice STEP (optional export).
        solid_geometry_file:      3D extruded solid STEP.
        mesh_file:                gmsh .msh mesh file.
        solver_input_file:        CalculiX .inp input deck.
        stdout_log_file:          CalculiX stdout log (if solver was run).
        stderr_log_file:          CalculiX stderr log (if solver was run).
    """

    run_directory: str
    geometry_directory: str
    mesh_directory: str
    solver_directory: str
    results_directory: str
    case_config_file: str | None = None
    metadata_file: str | None = None
    unit_cell_geometry_file: str | None = None
    lattice_geometry_file: str | None = None
    solid_geometry_file: str | None = None
    mesh_file: str | None = None
    solver_input_file: str | None = None
    stdout_log_file: str | None = None
    stderr_log_file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of all artifact paths."""
        return {
            "run_directory":           self.run_directory,
            "geometry_directory":      self.geometry_directory,
            "mesh_directory":          self.mesh_directory,
            "solver_directory":        self.solver_directory,
            "results_directory":       self.results_directory,
            "case_config_file":        self.case_config_file,
            "metadata_file":           self.metadata_file,
            "unit_cell_geometry_file": self.unit_cell_geometry_file,
            "lattice_geometry_file":   self.lattice_geometry_file,
            "solid_geometry_file":     self.solid_geometry_file,
            "mesh_file":               self.mesh_file,
            "solver_input_file":       self.solver_input_file,
            "stdout_log_file":         self.stdout_log_file,
            "stderr_log_file":         self.stderr_log_file,
        }


@dataclass
class CaseRunSummary:
    """
    Complete result of a single-case pipeline execution.

    Attributes:
        success:       True if the case completed all required stages.
        case_result:   Populated ``CaseResult`` for downstream ranking/reporting.
        artifacts:     Registry of all artifact file paths produced.
        warnings:      Non-fatal notes from any pipeline stage.
        metadata:      Supporting context (stage timings, signatures, etc.).
        error_message: Primary failure description if ``success`` is False.
    """

    success: bool
    case_result: CaseResult
    artifacts: CaseRunArtifacts
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of the full run summary."""
        return {
            "success":       self.success,
            "case_result":   self.case_result.to_dict()
                             if hasattr(self.case_result, "to_dict")
                             else str(self.case_result),
            "artifacts":     self.artifacts.to_dict(),
            "warnings":      self.warnings,
            "metadata":      self.metadata,
            "error_message": self.error_message,
        }


# ---------------------------------------------------------------------------
# Default options loader
# ---------------------------------------------------------------------------

def default_case_run_options(
    project_root: str | Path | None = None,
) -> CaseRunOptions:
    """
    Build ``CaseRunOptions`` from ``base_config.yaml`` solver/workflow settings.

    Reads:
        - ``solver.run_solver_by_default``    → ``run_solver``
        - ``solver.solver_timeout_seconds``   → ``solver_timeout_seconds``
        - ``meshing.default_preset``          → ``meshing_preset``
        - ``workflow.skip_completed_cases``   → ``skip_completed_cases``

    Falls back to ``CaseRunOptions()`` defaults on any config loading failure.

    Args:
        project_root: Project root for config resolution (optional).

    Returns:
        ``CaseRunOptions`` populated from config where possible.
    """
    opts = CaseRunOptions()
    try:
        from utils.config_loader import load_pipeline_config
        cfg = load_pipeline_config(project_root)

        solver_cfg = cfg.solver
        opts.run_solver = bool(solver_cfg.get("run_solver_by_default", False))
        timeout = solver_cfg.get("solver_timeout_seconds")
        if timeout is not None:
            opts.solver_timeout_seconds = int(timeout)

        meshing_cfg = cfg.meshing_config
        opts.meshing_preset = str(
            meshing_cfg.get("default_preset", "default")
        )

        workflow_cfg = getattr(cfg, "workflow", {}) or {}
        opts.skip_completed_cases = bool(
            workflow_cfg.get("skip_completed_cases", True)
        )
    except Exception as exc:
        logger.warning(
            "Could not load case run options from config: %s.  "
            "Using defaults (run_solver=False).",
            exc,
        )
    return opts


# ---------------------------------------------------------------------------
# Default file-naming helpers
# ---------------------------------------------------------------------------

def _default_solid_export_path(artifacts: CaseRunArtifacts, case_id: str) -> Path:
    """Return the canonical path for the 3D solid STEP file."""
    return Path(artifacts.geometry_directory) / f"{case_id}_solid.step"


def _default_mesh_path(artifacts: CaseRunArtifacts, case_id: str) -> Path:
    """Return the canonical path for the gmsh .msh mesh file."""
    return Path(artifacts.mesh_directory) / f"{case_id}.msh"


def _default_solver_input_path(artifacts: CaseRunArtifacts, case_id: str) -> Path:
    """Return the canonical path for the CalculiX .inp input deck."""
    return Path(artifacts.solver_directory) / f"{case_id}.inp"


# ---------------------------------------------------------------------------
# Metadata / config snapshot helpers
# ---------------------------------------------------------------------------

def _write_case_config_snapshot(
    case_definition: CaseDefinition,
    path: Path,
) -> None:
    """
    Write a JSON snapshot of the case definition to disk.

    This serves as an audit record of exactly what inputs were used to
    produce the artifacts in this run directory.

    ARCHITECTURAL DECISION — JSON, not YAML:
        Writing YAML from the standard library requires a custom serialiser.
        A JSON file named ``case_config.json`` is written instead.  It is
        human-readable and machine-readable.  The path in cache.py is named
        ``case_config_file`` (no extension enforcement), so this is compatible.

    Args:
        case_definition: Full case definition to serialise.
        path:            Destination file path.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(
                case_definition.to_dict()
                if hasattr(case_definition, "to_dict")
                else str(case_definition),
                fh,
                indent=2,
                default=str,
            )
    except Exception as exc:
        logger.warning(
            "Could not write case config snapshot to '%s': %s", path, exc
        )


def _write_case_metadata_snapshot(
    payload: Mapping[str, Any],
    path: Path,
) -> None:
    """
    Write a JSON metadata snapshot (run context, timings, signatures) to disk.

    Args:
        payload: JSON-serialisable dict.
        path:    Destination file path.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(dict(payload), fh, indent=2, default=str)
    except Exception as exc:
        logger.warning(
            "Could not write metadata snapshot to '%s': %s", path, exc
        )


# ---------------------------------------------------------------------------
# Internal stage helpers
# ---------------------------------------------------------------------------

def _prepare_case_run(
    case_definition: CaseDefinition,
    options: CaseRunOptions,
    project_root: str | Path | None,
) -> tuple[CaseRunArtifacts, dict[str, Any]]:
    """
    Create run directories and write the case config snapshot.

    Args:
        case_definition: Full case definition.
        options:         Run options.
        project_root:    Project root.

    Returns:
        Tuple ``(CaseRunArtifacts, run_metadata_dict)``.
    """
    cache_paths = ensure_case_directories(case_definition, project_root)

    artifacts = CaseRunArtifacts(
        run_directory=cache_paths.run_directory,
        geometry_directory=cache_paths.geometry_directory,
        mesh_directory=cache_paths.mesh_directory,
        solver_directory=cache_paths.solver_directory,
        results_directory=cache_paths.results_directory,
    )

    run_meta: dict[str, Any] = {
        "case_id":         case_definition.case_id,
        "meshing_preset":  options.meshing_preset,
        "run_solver":      options.run_solver,
        "stage_timings":   {},
        "geometry_signature": geometry_signature(case_definition).short_signature,
        "mesh_signature":     mesh_signature(
            case_definition, options.meshing_preset
        ).short_signature,
    }

    # Write case config snapshot
    if options.write_case_config_snapshot:
        config_path = Path(cache_paths.run_directory) / "case_config.json"
        _write_case_config_snapshot(case_definition, config_path)
        artifacts.case_config_file = str(config_path)

    return artifacts, run_meta


def _run_geometry_pipeline(
    case_definition: CaseDefinition,
    options: CaseRunOptions,
    artifacts: CaseRunArtifacts,
    run_meta: dict[str, Any],
    warnings: list[str],
) -> Any:
    """
    Run the geometry pipeline: unit-cell → lattice → 3D solid.

    Validates at each stage.  Optionally exports STEP files.

    Args:
        case_definition: Full case definition.
        options:         Run options.
        artifacts:       Artifact registry (mutated in-place with paths).
        run_meta:        Run metadata dict (mutated with timings).
        warnings:        Mutable warning list (appended to).

    Returns:
        The 3D CadQuery solid object.

    Raises:
        CaseRunnerError: if any geometry stage fails.
    """
    from geometry.unitcell_factory import create_unit_cell
    from geometry.lattice_builder import build_lattice_from_unit_cell
    from geometry.extruder import extrude_lattice_geometry
    from geometry.validators import (
        validate_unit_cell_object,
        validate_planar_geometry,
        validate_lattice_geometry,
        validate_solid_geometry,
    )

    # ------------------------------------------------------------------
    # Stage: Unit cell construction
    # ------------------------------------------------------------------
    t0 = time.monotonic()
    try:
        unit_cell = create_unit_cell(case_definition.design_parameters)
    except Exception as exc:
        raise CaseRunnerError(
            f"Unit-cell construction failed for '{case_definition.case_id}': {exc}"
        ) from exc

    uc_val = validate_unit_cell_object(unit_cell)
    if not uc_val.is_valid:
        raise CaseRunnerError(
            f"Unit-cell validation failed: {'; '.join(f'[{m.code}] {m.message}' for m in uc_val.errors)}"
        )
    if uc_val.warnings:
        warnings.extend(f"[unit_cell_val] {w}" for w in uc_val.warnings)

    run_meta["stage_timings"]["unit_cell_seconds"] = float(f"{time.monotonic() - t0:.3f}")
    logger.info(
        "Case '%s': unit-cell constructed (%.3f s).",
        case_definition.case_id,
        run_meta["stage_timings"]["unit_cell_seconds"],
    )

    # ------------------------------------------------------------------
    # Stage: 2D unit-cell geometry
    # ------------------------------------------------------------------
    t0 = time.monotonic()
    try:
        unit_cell_geom = unit_cell.build_2d()
    except Exception as exc:
        raise CaseRunnerError(
            f"2D unit-cell geometry build failed: {exc}"
        ) from exc

    planar_val = validate_planar_geometry(
        unit_cell_geom,
        minimum_feature_size_mm=options.minimum_feature_size_mm,
    )
    if not planar_val.is_valid:
        raise CaseRunnerError(
            f"Planar unit-cell geometry invalid: {'; '.join(f'[{m.code}] {m.message}' for m in planar_val.errors)}"
        )
    if planar_val.warnings:
        warnings.extend(f"[planar_val] {w}" for w in planar_val.warnings)

    # Optional STEP export of unit-cell geometry
    if options.write_intermediate_geometry:
        uc_step = (
            Path(artifacts.geometry_directory)
            / f"{case_definition.case_id}_unit_cell.step"
        )
        # ARCHITECTURAL DECISION — STEP export of 2D geometry is not universally
        # supported across all CadQuery Sketch/Workplane types in version 1.
        # We attempt it and warn on failure rather than aborting the pipeline.
        try:
            import cadquery as cq
            if hasattr(unit_cell_geom, "val"):
                cq.exporters.export(unit_cell_geom, str(uc_step))
                artifacts.unit_cell_geometry_file = str(uc_step)
            else:
                warnings.append(
                    "Unit-cell STEP export skipped: geometry object has no "
                    "exportable .val() — not a resolved Workplane.  "
                    "This is expected for Sketch-only outputs in version 1."
                )
        except Exception as exc:
            warnings.append(
                f"Unit-cell STEP export failed (non-fatal): {exc}"
            )

    run_meta["stage_timings"]["unit_cell_geom_seconds"] = float(f"{time.monotonic() - t0:.3f}")

    # ------------------------------------------------------------------
    # Stage: 2D lattice assembly
    # ------------------------------------------------------------------
    t0 = time.monotonic()
    try:
        lattice_geom = build_lattice_from_unit_cell(
            unit_cell=unit_cell,
            repeats_x=case_definition.lattice_repeats_x,
            repeats_y=case_definition.lattice_repeats_y,
        )
    except Exception as exc:
        raise CaseRunnerError(
            f"Lattice assembly failed: {exc}"
        ) from exc

    lattice_val = validate_lattice_geometry(
        lattice_geom,
        expected_repeats_x=case_definition.lattice_repeats_x,
        expected_repeats_y=case_definition.lattice_repeats_y,
        cell_size=case_definition.design_parameters.cell_size,
    )
    if not lattice_val.is_valid:
        raise CaseRunnerError(
            f"Lattice geometry validation failed: {'; '.join(f'[{m.code}] {m.message}' for m in lattice_val.errors)}"
        )
    if lattice_val.warnings:
        warnings.extend(f"[lattice_val] {w}" for w in lattice_val.warnings)

    # Optional STEP export of lattice geometry
    if options.write_intermediate_geometry:
        lat_step = (
            Path(artifacts.geometry_directory)
            / f"{case_definition.case_id}_lattice.step"
        )
        try:
            import cadquery as cq
            if hasattr(lattice_geom, "val"):
                cq.exporters.export(lattice_geom, str(lat_step))
                artifacts.lattice_geometry_file = str(lat_step)
        except Exception as exc:
            warnings.append(
                f"Lattice STEP export failed (non-fatal): {exc}"
            )

    run_meta["stage_timings"]["lattice_seconds"] = float(f"{time.monotonic() - t0:.3f}")
    logger.info(
        "Case '%s': lattice assembled (%dx%d, %.3f s).",
        case_definition.case_id,
        case_definition.lattice_repeats_x,
        case_definition.lattice_repeats_y,
        run_meta["stage_timings"]["lattice_seconds"],
    )

    # ------------------------------------------------------------------
    # Stage: 3D extrusion
    # ------------------------------------------------------------------
    t0 = time.monotonic()
    try:
        solid = extrude_lattice_geometry(
            lattice_geom,
            plate_thickness=case_definition.plate_thickness,
            centered=options.centered_extrusion,
        )
    except Exception as exc:
        raise CaseRunnerError(
            f"3D extrusion failed: {exc}"
        ) from exc

    solid_val = validate_solid_geometry(
        solid,
        minimum_feature_size_mm=options.minimum_feature_size_mm,
    )
    if not solid_val.is_valid:
        raise CaseRunnerError(
            f"Solid geometry validation failed: {'; '.join(f'[{m.code}] {m.message}' for m in solid_val.errors)}"
        )
    if solid_val.warnings:
        warnings.extend(f"[solid_val] {w}" for w in solid_val.warnings)

    # Export solid STEP (primary geometry artifact)
    solid_step = _default_solid_export_path(artifacts, case_definition.case_id)
    try:
        import cadquery as cq
        cq.exporters.export(solid, str(solid_step))
        artifacts.solid_geometry_file = str(solid_step)
        logger.info(
            "Case '%s': solid exported to '%s'.",
            case_definition.case_id, solid_step,
        )
    except Exception as exc:
        warnings.append(
            f"Solid STEP export failed (non-fatal): {exc}.  "
            "The mesh will still be attempted from the in-memory solid."
        )

    run_meta["stage_timings"]["extrusion_seconds"] = float(f"{time.monotonic() - t0:.3f}")
    logger.info(
        "Case '%s': extrusion complete (thickness=%.2f mm, %.3f s).",
        case_definition.case_id,
        case_definition.plate_thickness,
        run_meta["stage_timings"]["extrusion_seconds"],
    )

    return solid


def _run_meshing_pipeline(
    case_definition: CaseDefinition,
    solid: Any,
    options: CaseRunOptions,
    artifacts: CaseRunArtifacts,
    run_meta: dict[str, Any],
    warnings: list[str],
) -> Path:
    """
    Mesh the 3D solid using gmsh.

    Args:
        case_definition: Full case definition.
        solid:           CadQuery solid from the geometry pipeline.
        options:         Run options.
        artifacts:       Artifact registry (mutated with mesh path).
        run_meta:        Run metadata dict (mutated with timings/status).
        warnings:        Mutable warning list.

    Returns:
        Path to the generated ``.msh`` file.

    Raises:
        CaseRunnerError: if meshing fails.
    """
    from simulation.mesher import generate_volume_mesh

    mesh_path = _default_mesh_path(artifacts, case_definition.case_id)

    t0 = time.monotonic()
    try:
        mesh_result = generate_volume_mesh(
            solid_geometry=solid,
            output_msh_path=mesh_path,
            preset_name=options.meshing_preset,
        )
    except Exception as exc:
        raise CaseRunnerError(
            f"Meshing raised an unhandled exception: {exc}"
        ) from exc

    elapsed = float(f"{time.monotonic() - t0:.3f}")
    run_meta["stage_timings"]["meshing_seconds"] = elapsed

    if not mesh_result.success:
        run_meta["mesh_status"] = MeshStatus.FAILED.value
        raise CaseRunnerError(
            f"Meshing failed: {mesh_result.error_message}"
        )

    if mesh_result.warnings:
        warnings.extend(f"[mesher] {w}" for w in mesh_result.warnings)

    artifacts.mesh_file = str(mesh_path)
    run_meta["mesh_status"] = MeshStatus.SUCCESS.value
    run_meta["mesh_node_count"]    = mesh_result.metadata.get("node_count")
    run_meta["mesh_element_count"] = mesh_result.metadata.get("element_count")

    # --- Convert to CalculiX INP include (Version 2) ---
    try:
        t0_conv = time.monotonic()
        from simulation.calculix_converter import convert_msh_to_inp
        convert_msh_to_inp(mesh_path)
        run_meta["stage_timings"]["mesh_convert_seconds"] = float(f"{time.monotonic() - t0_conv:.3f}")
    except Exception as exc:
        warnings.append(f"Mesh to INP conversion failed: {exc}")

    logger.info(
        "Case '%s': meshed in %.3f s (%s nodes, %s elements).",
        case_definition.case_id, elapsed,
        run_meta.get("mesh_node_count", "?"),
        run_meta.get("mesh_element_count", "?"),
    )
    return mesh_path


def _error_message_for_failed_solver_run(
    warnings: list[str],
    return_code: int | None,
) -> str:
    """Pick a concise error string after a non-zero CalculiX exit."""
    for w in reversed(warnings):
        if "CalculiX returned failure:" in w or "CalculiX failed:" in w:
            return w
    if return_code is not None:
        return f"CalculiX exited with code {return_code}."
    return "CalculiX did not complete successfully."


def _run_solver_pipeline(
    case_definition: CaseDefinition,
    mesh_path: Path,
    options: CaseRunOptions,
    artifacts: CaseRunArtifacts,
    run_meta: dict[str, Any],
    warnings: list[str],
) -> tuple[bool, int | None]:
    """
    Export the CalculiX input deck and optionally execute the solver.

    Args:
        case_definition: Full case definition.
        mesh_path:       Path to the ``.msh`` file.
        options:         Run options.
        artifacts:       Artifact registry (mutated with solver paths).
        run_meta:        Run metadata dict (mutated with solver status).
        warnings:        Mutable warning list.

    Returns:
        Tuple ``(solver_success: bool, return_code: int | None)``.
        ``solver_success`` is True for export-only mode (no execution = no failure).
    """
    from simulation.materials import MaterialLibrary
    from simulation.loadcases import LoadCaseLibrary
    from simulation.solver_exporter import (
        export_calculix_input_deck,
        SolverExportOptions,
    )
    from simulation.runner import (
        run_calculix_input_deck,
        SolverRunOptions,
    )

    # --- Load material and load-case records ---
    try:
        from utils.config_loader import load_pipeline_config
        cfg = load_pipeline_config()
        from simulation.materials import load_material_library
        from simulation.loadcases import load_loadcase_library
        mat_lib = load_material_library(project_root=cfg.project_root)
        lc_lib  = load_loadcase_library(project_root=cfg.project_root)
        material = mat_lib.get(case_definition.material.name)
        loadcase = lc_lib.get(case_definition.load_case.load_case_type.value)
    except Exception as exc:
        warnings.append(
            f"Could not load material/loadcase records from config: {exc}.  "
            "Solver export will be attempted with available data."
        )
        material = None
        loadcase = None

    # If either is unavailable, we cannot write a meaningful .inp
    if material is None or loadcase is None:
        warnings.append(
            "Solver export skipped: material or loadcase record is unavailable.  "
            "Check materials.yaml and loadcases.yaml."
        )
        run_meta["solver_status"] = SolverStatus.NOT_RUN.value
        return True, None

    # --- Export solver input deck ---
    inp_path = _default_solver_input_path(artifacts, case_definition.case_id)
    export_opts = SolverExportOptions()

    t0 = time.monotonic()
    try:
        export_result = export_calculix_input_deck(
            mesh_path=mesh_path,
            material=material,
            loadcase=loadcase,
            output_inp_path=inp_path,
            case_definition=case_definition,
            options=export_opts,
        )
    except Exception as exc:
        warnings.append(f"Solver export raised an unexpected exception: {exc}")
        run_meta["solver_status"] = SolverStatus.NOT_RUN.value
        return True, None  # non-fatal: export failure does not abort postprocess

    run_meta["stage_timings"]["solver_export_seconds"] = float(f"{time.monotonic() - t0:.3f}")

    if not export_result.success:
        warnings.append(
            f"Solver input deck export failed: {export_result.error_message}"
        )
        run_meta["solver_status"] = SolverStatus.NOT_RUN.value
        return True, None

    if export_result.warnings:
        warnings.extend(f"[solver_export] {w}" for w in export_result.warnings)

    artifacts.solver_input_file = export_result.input_deck_path
    logger.info(
        "Case '%s': solver input deck written to '%s'.",
        case_definition.case_id, inp_path,
    )

    # --- Optionally run the solver ---
    if not options.run_solver:
        run_meta["solver_status"] = SolverStatus.NOT_RUN.value
        warnings.append(
            "Solver execution skipped (run_solver=False).  "
            "Set CaseRunOptions.run_solver=True to execute CalculiX."
        )
        return True, None

    run_opts = SolverRunOptions(
        run_solver=True,
        timeout_seconds=options.solver_timeout_seconds,
    )
    t0 = time.monotonic()
    try:
        run_result = run_calculix_input_deck(inp_path, run_opts)
    except Exception as exc:
        run_meta["solver_status"] = SolverStatus.FAILED.value
        warnings.append(f"Solver runner raised an unexpected exception: {exc}")
        return False, None

    run_meta["stage_timings"]["solver_run_seconds"] = float(f"{time.monotonic() - t0:.3f}")

    if run_result.stdout_log_path:
        artifacts.stdout_log_file = run_result.stdout_log_path
    if run_result.stderr_log_path:
        artifacts.stderr_log_file = run_result.stderr_log_path
    if run_result.warnings:
        warnings.extend(f"[solver_run] {w}" for w in run_result.warnings)

    if run_result.success:
        run_meta["solver_status"] = SolverStatus.SUCCESS.value
        logger.info(
            "Case '%s': CalculiX completed (return_code=%s, %.3f s).",
            case_definition.case_id,
            run_result.return_code,
            run_meta["stage_timings"].get("solver_run_seconds", 0.0),
        )
        return True, run_result.return_code
    else:
        run_meta["solver_status"] = SolverStatus.FAILED.value
        warnings.append(
            f"CalculiX returned failure: {run_result.error_message}"
        )
        return False, run_result.return_code


def _run_analysis_pipeline(
    case_definition: CaseDefinition,
    artifacts: CaseRunArtifacts,
    run_meta: dict[str, Any],
    warnings: list[str],
    continue_on_partial: bool,
) -> MetricSet:
    """
    Postprocess solver outputs and compute engineering metrics + fatigue proxy.

    Args:
        case_definition:    Full case definition.
        artifacts:          Artifact registry (results directory used for scanning).
        run_meta:           Run metadata dict (mutated with metric summaries).
        warnings:           Mutable warning list.
        continue_on_partial: If True, return partial metrics rather than aborting.

    Returns:
        ``MetricSet`` with all computable metrics populated (others as None).
    """
    from analysis.postprocess import postprocess_solver_outputs
    from analysis.metrics import compute_engineering_metrics

    # Try to load material and loadcase for fatigue proxy (best-effort)
    material = None
    loadcase  = None
    try:
        from utils.config_loader import load_pipeline_config
        from simulation.materials import load_material_library
        from simulation.loadcases import load_loadcase_library
        cfg = load_pipeline_config()
        material = load_material_library(project_root=cfg.project_root).get(
            case_definition.material.name
        )
        loadcase = load_loadcase_library(project_root=cfg.project_root).get(
            case_definition.load_case.load_case_type.value
        )
    except Exception as exc:
        warnings.append(
            f"Could not load material/loadcase for fatigue proxy: {exc}.  "
            "Fatigue risk score will be None."
        )

    # Postprocess from solver_directory (contains .inp, .frd, .dat, logs)
    t0 = time.monotonic()
    try:
        postprocess_result = postprocess_solver_outputs(
            artifacts.solver_directory
        )
    except Exception as exc:
        if continue_on_partial:
            warnings.append(
                f"Postprocessing raised an exception (continuing): {exc}"
            )
            from analysis.postprocess import PostprocessResult
            postprocess_result = PostprocessResult(
                success=False,
                error_message=str(exc),
            )
        else:
            raise CaseRunnerError(
                f"Postprocessing failed: {exc}"
            ) from exc

    run_meta["stage_timings"]["postprocess_seconds"] = float(f"{time.monotonic() - t0:.3f}")

    if postprocess_result.warnings:
        warnings.extend(
            f"[postprocess] {w}" for w in postprocess_result.warnings
        )

    # Try to resolve applied force from the loadcase record for stiffness calc
    applied_force_n: float | None = None
    if loadcase is not None:
        applied_force_n = getattr(loadcase, "force_n", None) or getattr(
            loadcase, "mean_force_n", None
        )

    # Compute engineering metrics
    t0 = time.monotonic()
    try:
        metrics_result = compute_engineering_metrics(
            postprocess_result=postprocess_result,
            applied_force_n=applied_force_n,
            material=material,
            loadcase=loadcase,
        )
    except Exception as exc:
        warnings.append(
            f"Metrics computation raised an exception (continuing): {exc}"
        )
        return MetricSet()

    run_meta["stage_timings"]["metrics_seconds"] = float(f"{time.monotonic() - t0:.3f}")

    if metrics_result.warnings:
        warnings.extend(
            f"[metrics] {w}" for w in metrics_result.warnings
        )

    if metrics_result.fatigue_proxy_result:
        run_meta["fatigue_proxy_category"] = (
            metrics_result.fatigue_proxy_result.metadata.get(
                "risk_category", "unknown"
            )
        )

    return metrics_result.metric_set


# ---------------------------------------------------------------------------
# CaseResult assembly helper
# ---------------------------------------------------------------------------

def _build_case_result(
    case_definition: CaseDefinition,
    status: CaseStatus,
    success: bool,
    metric_set: MetricSet,
    artifacts: CaseRunArtifacts,
    run_meta: dict[str, Any],
    error_message: str | None,
    runtime_seconds: float | None,
    solver_return_code: int | None,
    warnings: list[str],
) -> CaseResult:
    """
    Assemble the canonical ``CaseResult`` from all pipeline stage outputs.

    Args:
        case_definition:   Full case definition.
        status:            Final lifecycle status.
        success:           Overall success flag.
        metric_set:        Computed engineering metrics.
        artifacts:         Artifact registry.
        run_meta:          Run metadata dict.
        error_message:     Primary failure description (or None).
        runtime_seconds:   Total elapsed runtime.
        solver_return_code: CalculiX return code (or None).
        warnings:          All accumulated warnings.

    Returns:
        ``CaseResult`` populated for downstream ranking and reporting.
    """
    artifact_dict: dict[str, str] = {}
    for attr, label in (
        ("solid_geometry_file",     "solid_step"),
        ("mesh_file",               "mesh_msh"),
        ("solver_input_file",       "solver_inp"),
        ("stdout_log_file",         "stdout_log"),
        ("stderr_log_file",         "stderr_log"),
        ("case_config_file",        "case_config"),
        ("unit_cell_geometry_file", "unit_cell_step"),
        ("lattice_geometry_file",   "lattice_step"),
    ):
        val = getattr(artifacts, attr, None)
        if val:
            artifact_dict[label] = val

    return CaseResult(
        case_id=case_definition.case_id,
        status=status,
        success=success,
        metrics=metric_set,
        error_message=error_message,
        runtime_seconds=runtime_seconds,
        solver_return_code=solver_return_code,
        artifacts=artifact_dict,
        metadata={
            **run_meta,
            "case_runner_warnings": warnings,
            "design_type":          case_definition.design_type.value,
            "plate_thickness_mm":   case_definition.plate_thickness,
            "material_name":        case_definition.material.name,
            "load_case_type":       case_definition.load_case.load_case_type.value,
        },
    )


# ---------------------------------------------------------------------------
# Main single-case runner
# ---------------------------------------------------------------------------

def run_case(
    case_definition: CaseDefinition,
    options: CaseRunOptions | None = None,
    project_root: str | Path | None = None,
) -> CaseRunSummary:
    """
    Execute one pipeline case end-to-end and return a structured summary.

    Pipeline stages (in order):
        1.  Validate inputs and resolve options.
        2.  Create run directories and write config snapshot.
        3.  Check skip condition (already COMPLETED).
        4.  Mark status RUNNING.
        5.  Build unit-cell → lattice → 3D solid (geometry pipeline).
        6.  Mesh the solid (meshing pipeline).
        7.  Export CalculiX input deck; optionally run solver.
        8.  Postprocess solver outputs; compute engineering metrics.
        9.  Assemble ``CaseResult``.
        10. Mark status COMPLETED or FAILED.
        11. Return ``CaseRunSummary``.

    Args:
        case_definition: Full ``CaseDefinition`` for this case.
        options:         Run configuration.  Uses ``default_case_run_options``
                         if None.
        project_root:    Project root for directory resolution (auto-detected
                         if None).

    Returns:
        ``CaseRunSummary`` with ``success``, ``case_result``, ``artifacts``,
        ``warnings``, and any ``error_message``.
    """
    if options is None:
        options = default_case_run_options(project_root)

    case_id = case_definition.case_id
    warnings: list[str] = []
    t_case_start = time.monotonic()

    logger.info("CaseRunner: starting case '%s'.", case_id)

    # ------------------------------------------------------------------
    # 1. Validate case_definition basics
    # ------------------------------------------------------------------
    try:
        if hasattr(case_definition, "validate"):
            case_definition.validate()
        if not case_definition.case_id:
            raise ValueError("case_definition.case_id must not be empty.")
    except Exception as exc:
        error_msg = f"Case definition invalid: {exc}"
        logger.error("Case '%s': %s", case_id, error_msg)
        dummy_artifacts = CaseRunArtifacts(
            run_directory="", geometry_directory="",
            mesh_directory="", solver_directory="", results_directory="",
        )
        return CaseRunSummary(
            success=False,
            case_result=CaseResult(
                case_id=case_id,
                status=CaseStatus.FAILED,
                success=False,
                error_message=error_msg,
            ),
            artifacts=dummy_artifacts,
            error_message=error_msg,
        )

    # ------------------------------------------------------------------
    # 2. Prepare directories and config snapshot
    # ------------------------------------------------------------------
    try:
        artifacts, run_meta = _prepare_case_run(
            case_definition, options, project_root
        )
    except Exception as exc:
        error_msg = f"Failed to prepare run directories: {exc}"
        logger.error("Case '%s': %s", case_id, error_msg)
        dummy_artifacts = CaseRunArtifacts(
            run_directory="", geometry_directory="",
            mesh_directory="", solver_directory="", results_directory="",
        )
        return CaseRunSummary(
            success=False,
            case_result=CaseResult(
                case_id=case_id,
                status=CaseStatus.FAILED,
                success=False,
                error_message=error_msg,
            ),
            artifacts=dummy_artifacts,
            error_message=error_msg,
        )

    # ------------------------------------------------------------------
    # 3. Skip check
    # ------------------------------------------------------------------
    if case_should_be_skipped(
        case_id,
        skip_completed_cases=options.skip_completed_cases,
        project_root=project_root,
    ):
        logger.info("Case '%s': skipping (already completed or skipped).", case_id)
        mark_case_skipped(
            case_id,
            project_root=project_root,
            reason="Skipped by case_runner: result already exists.",
        )
        metric_set = MetricSet()
        case_result = _build_case_result(
            case_definition=case_definition,
            status=CaseStatus.SKIPPED,
            success=True,
            metric_set=metric_set,
            artifacts=artifacts,
            run_meta=run_meta,
            error_message=None,
            runtime_seconds=None,
            solver_return_code=None,
            warnings=warnings,
        )
        return CaseRunSummary(
            success=True,
            case_result=case_result,
            artifacts=artifacts,
            warnings=["Case skipped: result already exists."],
            metadata=run_meta,
        )

    # ------------------------------------------------------------------
    # 4 → 8. Execute pipeline stages with per-stage status updates
    # ------------------------------------------------------------------
    solid = None
    mesh_path: Path | None = None
    solver_success = True
    solver_return_code: int | None = None
    metric_set = MetricSet()
    final_status = CaseStatus.FAILED
    error_message: str | None = None

    try:
        # --- Stage: geometry ---
        mark_case_running(case_id, project_root=project_root, stage="geometry")
        solid = _run_geometry_pipeline(
            case_definition, options, artifacts, run_meta, warnings
        )

        # --- Stage: meshing ---
        mark_case_running(case_id, project_root=project_root, stage="meshing")
        mesh_path = _run_meshing_pipeline(
            case_definition, solid, options, artifacts, run_meta, warnings
        )

        # --- Stage: solver export / run ---
        mark_case_running(
            case_id, project_root=project_root,
            stage="solver_export" if not options.run_solver else "solver_run",
        )
        solver_success, solver_return_code = _run_solver_pipeline(
            case_definition, mesh_path, options, artifacts, run_meta, warnings
        )

        # --- Stage: postprocess + metrics ---
        mark_case_running(case_id, project_root=project_root, stage="analysis")
        metric_set = _run_analysis_pipeline(
            case_definition, artifacts, run_meta, warnings,
            continue_on_partial=options.continue_on_partial_postprocess,
        )

        if options.run_solver and not solver_success:
            final_status = CaseStatus.FAILED
            error_message = _error_message_for_failed_solver_run(
                warnings, solver_return_code
            )
            logger.error(
                "Case '%s': FAILED — solver run unsuccessful (return_code=%s).",
                case_id,
                solver_return_code,
            )
        else:
            final_status = CaseStatus.COMPLETED
            logger.info("Case '%s': all pipeline stages completed.", case_id)

    except CaseRunnerError as exc:
        error_message = str(exc)
        final_status  = CaseStatus.FAILED
        logger.error("Case '%s': pipeline failed — %s", case_id, error_message)

    except Exception as exc:
        error_message = f"Unexpected error in case runner: {exc}"
        final_status  = CaseStatus.FAILED
        logger.exception("Case '%s': unexpected exception.", case_id)

    # ------------------------------------------------------------------
    # 9. Compute total runtime and assemble CaseResult
    # ------------------------------------------------------------------
    runtime_seconds = float(f"{time.monotonic() - t_case_start:.3f}")
    success = (final_status == CaseStatus.COMPLETED)

    case_result = _build_case_result(
        case_definition=case_definition,
        status=final_status,
        success=success,
        metric_set=metric_set,
        artifacts=artifacts,
        run_meta=run_meta,
        error_message=error_message,
        runtime_seconds=runtime_seconds,
        solver_return_code=solver_return_code,
        warnings=warnings,
    )

    # ------------------------------------------------------------------
    # 10. Write final status
    # ------------------------------------------------------------------
    try:
        if success:
            mark_case_completed(
                case_id,
                project_root=project_root,
                message="Case completed successfully.",
                runtime_seconds=runtime_seconds,
                metadata={"total_warnings": len(warnings)},
            )
        else:
            mark_case_failed(
                case_id,
                project_root=project_root,
                error_message=error_message or "Pipeline failed.",
                stage=run_meta.get("current_stage"),
                runtime_seconds=runtime_seconds,
                metadata={"total_warnings": len(warnings)},
            )
    except StatusTrackerError as exc:
        warnings.append(f"Failed to write final status file: {exc}")

    # Optionally write metadata snapshot
    if options.write_case_metadata_snapshot:
        meta_path = Path(artifacts.run_directory) / "run_metadata.json"
        _write_case_metadata_snapshot(
            {
                **run_meta,
                "final_status":    final_status.value,
                "runtime_seconds": runtime_seconds,
                "total_warnings":  len(warnings),
                "warnings":        warnings,
                "success":         success,
            },
            meta_path,
        )
        artifacts.metadata_file = str(meta_path)

    logger.info(
        "Case '%s': finished in %.3f s — status=%s.",
        case_id, runtime_seconds, final_status.value,
    )

    return CaseRunSummary(
        success=success,
        case_result=case_result,
        artifacts=artifacts,
        warnings=warnings,
        metadata=run_meta,
        error_message=error_message,
    )


# ---------------------------------------------------------------------------
# Strict wrapper
# ---------------------------------------------------------------------------

def require_successful_case_run(
    case_definition: CaseDefinition,
    options: CaseRunOptions | None = None,
    project_root: str | Path | None = None,
) -> CaseRunSummary:
    """
    Run a case and raise ``CaseRunnerError`` if it fails.

    ARCHITECTURAL DECISION — SKIPPED is not a failure:
        A case that was skipped because it is already completed is returned
        as ``success=True``.  Only ``CaseStatus.FAILED`` triggers the raise.

    Args:
        case_definition: Full ``CaseDefinition``.
        options:         Run options.
        project_root:    Project root.

    Returns:
        ``CaseRunSummary`` with ``success=True``.

    Raises:
        CaseRunnerError: if the case runs and fails.
    """
    summary = run_case(case_definition, options, project_root)
    if not summary.success:
        raise CaseRunnerError(
            f"Case '{case_definition.case_id}' failed: {summary.error_message}"
        )
    return summary
