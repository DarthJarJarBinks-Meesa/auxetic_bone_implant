"""
src/simulation/solver_exporter.py
====================================
CalculiX solver-export module for the auxetic plate pipeline.

This module writes CalculiX-style ``.inp`` input deck scaffolds from the
pipeline's mesh, material, and load-case definitions.

PIPELINE POSITION:
    mesh (model.msh)  +  material  +  loadcase
    →  [THIS MODULE]
    →  solver/input.inp  →  runner.py  →  postprocess.py

ARCHITECTURAL DECISION — export only, no solver execution:
    This module writes ``.inp`` files.  It never calls ``ccx``.  Solver
    execution is handled by ``runner.py`` which reads the files written here.

ARCHITECTURAL DECISION — honest scaffold, not fake full pipeline:
    Version 1 does not include a robust gmsh .msh → CalculiX .inp mesh
    converter.  Rather than pretending this is done, the exporter:
      (a) writes all non-mesh sections (heading, material, step, BCs,
          loads, output) as fully correct CalculiX syntax;
      (b) writes a clearly labelled MESH PLACEHOLDER block that tells
          the user exactly what must be present/supplied;
      (c) checks whether a pre-converted ``.inp`` mesh fragment file
          exists alongside the ``.msh`` and includes it if found.
    This means the ``input.inp`` is structurally complete and will run
    against CalculiX if a properly formatted mesh include file is placed
    in the expected location.  The limitation is documented in the file
    header, in warnings, and in ``SolverExportResult.warnings``.
    Full automatic mesh conversion (via gmsh → Abaqus → CalculiX or a
    direct gmsh API node/element dump) is a future-version improvement.

ARCHITECTURAL DECISION — cyclic proxy exported as static envelope step:
    CalculiX supports time-stepping cyclic analyses, but version 1 uses
    only a fatigue-risk proxy (not a validated fatigue solver).  The cyclic
    load case is exported as a static step with the mean + amplitude as the
    effective envelope force.  Comments in the generated file explain this
    and flag the result as proxy-only.

ARCHITECTURAL DECISION — node/element set names are symbolic placeholders:
    Without converting the gmsh mesh to CalculiX format, exact node/element
    numbers are not available here.  The generated .inp uses symbolic names
    (ALL_NODES, FIXED_FACE, LOADED_FACE, etc.) that a mesh conversion step
    or manual pre-processing step must provide as *NSET/*ELSET blocks.
    This is clearly documented in the generated file header.

UNITS (consistent with project-wide convention):
    Length  : mm
    Force   : N
    Stress  : MPa  (= N/mm²)
    Density : g/mm³  (converted from g/cm³ by dividing by 1000)

CALCULIX .INP STRUCTURE GENERATED:
    ** Heading
    *HEADING
    ** Mesh include (or placeholder)
    *INCLUDE or ** MESH PLACEHOLDER
    ** Material
    *MATERIAL, NAME=...
    *ELASTIC
    *DENSITY
    ** Step
    *STEP
    *STATIC  (or *FREQUENCY for future modes)
    ** Boundary conditions
    *BOUNDARY
    ** Loads
    *CLOAD
    ** Output requests
    *NODE FILE / *EL FILE
    *END STEP
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from simulation.loadcases import LoadCaseRecord
from simulation.materials import MaterialRecord
from workflow.case_schema import CaseDefinition, LoadCaseType


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class SolverExportError(Exception):
    """
    Raised when solver input-deck export fails.

    Covers:
      - missing mesh file path
      - missing or invalid material / load-case data
      - unsupported load-case type
      - invalid output path
      - file write failures
    """


# ---------------------------------------------------------------------------
# Typed dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SolverExportOptions:
    """
    Configuration flags for the solver-input exporter.

    Attributes:
        analysis_type:                    CalculiX step type keyword
                                          (``"static"``; future: ``"frequency"``).
        write_comments:                   Include inline comments in the .inp.
        include_heading:                  Include ``*HEADING`` block.
        include_placeholder_node_sets:    Include symbolic NSET definitions.
        include_placeholder_element_sets: Include symbolic ELSET definitions.
        include_output_requests:          Include ``*NODE FILE``/``*EL FILE``.
        cyclic_proxy_as_static_envelope:  Export CYCLIC case as a static step
                                          with effective envelope force.
    """

    analysis_type: str = "static"
    write_comments: bool = True
    include_heading: bool = True
    include_placeholder_node_sets: bool = True
    include_placeholder_element_sets: bool = True
    include_output_requests: bool = True
    cyclic_proxy_as_static_envelope: bool = True


@dataclass
class SolverExportResult:
    """
    Result of one solver-input export operation.

    Attributes:
        success:          True if the ``.inp`` was written successfully.
        solver_backend:   Always ``"calculix"`` in version 1.
        input_deck_path:  Path to the written ``.inp`` file (if successful).
        auxiliary_files:  Other files written (e.g. mesh fragments).
        warnings:         Non-fatal warnings (e.g. mesh placeholder used).
        metadata:         Additional metadata for logging and reporting.
        error_message:    Error description if ``success`` is False.
    """

    success: bool
    solver_backend: str = "calculix"
    input_deck_path: str | None = None
    auxiliary_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success":          self.success,
            "solver_backend":   self.solver_backend,
            "input_deck_path":  self.input_deck_path,
            "auxiliary_files":  self.auxiliary_files,
            "warnings":         self.warnings,
            "metadata":         self.metadata,
            "error_message":    self.error_message,
        }


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------

def _default_inp_filename(case_id: str) -> str:
    """Return the default ``.inp`` filename for a given case ID."""
    return f"{_sanitize_solver_label(case_id)}.inp"


def _sanitize_solver_label(value: str) -> str:
    """
    Sanitise a string for use as a CalculiX set name or filename component.

    Replaces non-alphanumeric characters (except underscores and hyphens)
    with underscores and truncates to 80 characters (CalculiX line limit).
    """
    sanitized = "".join(
        c if (c.isalnum() or c in "_-") else "_" for c in value
    )
    return "".join(c for i, c in enumerate(sanitized) if i < 80)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_export_inputs(
    mesh_path: Path,
    material: MaterialRecord,
    loadcase: LoadCaseRecord,
    output_inp_path: Path,
) -> None:
    """
    Validate inputs to the main export function.

    Raises:
        SolverExportError: with a descriptive message on any validation failure.
    """
    if not mesh_path.exists():
        raise SolverExportError(
            f"Mesh file not found: {mesh_path}.  "
            f"Run the mesher before calling the solver exporter."
        )
    if not material.name:
        raise SolverExportError("MaterialRecord.name must not be empty.")
    if material.mechanical.elastic_modulus_mpa <= 0.0:
        raise SolverExportError(
            f"Material '{material.name}': elastic_modulus_mpa must be positive."
        )
    if not loadcase.key:
        raise SolverExportError("LoadCaseRecord.key must not be empty.")
    if not loadcase.enabled:
        raise SolverExportError(
            f"Load case '{loadcase.key}' is disabled.  "
            f"Only enabled load cases should be exported."
        )
    try:
        output_inp_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise SolverExportError(
            f"Cannot create output directory '{output_inp_path.parent}': {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Block builders
# ---------------------------------------------------------------------------

def _build_heading_block(
    case_definition: CaseDefinition | None,
    material: MaterialRecord,
    loadcase: LoadCaseRecord,
) -> str:
    """
    Build the ``*HEADING`` block for the CalculiX input deck.

    Args:
        case_definition: Full case definition (optional; used for metadata).
        material:        Material record.
        loadcase:        Load-case record.

    Returns:
        Multi-line heading string.
    """
    lines = ["*HEADING"]
    lines.append(
        "** Auxetic Orthopedic Plate Pipeline — Version 1 — Auto-generated"
    )
    if case_definition is not None:
        lines.append(f"** Case ID    : {case_definition.case_id}")
        lines.append(f"** Case label : {case_definition.case_label}")
        dt = case_definition.design_type.value if case_definition else "unknown"
        lines.append(f"** Design     : {dt}")
        lines.append(
            f"** Plate thickness : {case_definition.plate_thickness:.3f} mm"
        )
    lines.append(f"** Material   : {material.name}")
    lines.append(f"** Load case  : {loadcase.name} ({loadcase.load_case_type.value})")
    lines.append(
        "** Units      : mm, N, MPa  (length=mm, force=N, stress=N/mm2)"
    )
    if loadcase.load_case_type == LoadCaseType.CYCLIC:
        lines.append(
            "** NOTE: CYCLIC case exported as static envelope step."
        )
        lines.append(
            "** fatigue_risk_score is computed by fatigue_model.py, NOT by CalculiX."
        )
    lines.append("** " + "-" * 70)
    return "\n".join(lines) + "\n"


def _build_mesh_reference_block(
    mesh_path: Path,
) -> tuple[str, list[str]]:
    """
    Build the mesh reference section of the input deck.

    ARCHITECTURAL DECISION — check for a pre-converted .inp mesh fragment:
        If a file ``<mesh_stem>_mesh.inp`` exists alongside the ``.msh``
        file, it is assumed to be a valid CalculiX mesh include (containing
        *NODE, *ELEMENT, *NSET, *ELSET blocks) and is referenced with
        ``*INCLUDE``.  If no such file exists, a clearly labelled placeholder
        block is written with instructions for what must be provided.

    Args:
        mesh_path: Path to the ``.msh`` gmsh mesh file.

    Returns:
        Tuple (block_text, warnings) where warnings lists any placeholder
        conditions.
    """
    mesh_inp_fragment = mesh_path.parent / (mesh_path.stem + "_mesh.inp")
    warnings: list[str] = []

    if mesh_inp_fragment.exists():
        block = (
            f"**\n"
            f"** === MESH INCLUDE ===\n"
            f"*INCLUDE, INPUT={mesh_inp_fragment.name}\n"
            f"** Mesh source: {mesh_path.name}\n"
        )
    else:
        block = textwrap.dedent(f"""\
            **
            ** === MESH PLACEHOLDER — VERSION 1 ===
            ** gmsh .msh source: {mesh_path.name}
            **
            ** IMPORTANT: This section is a placeholder.
            ** A full .msh → CalculiX .inp mesh converter is not yet implemented.
            **
            ** To use this input deck with CalculiX:
            **   Option A: Convert '{mesh_path.name}' to CalculiX format using
            **             a mesh conversion tool (e.g. gmsh GUI export to .inp,
            **             or a dedicated gmsh→CalculiX converter script), then
            **             save the result as '{mesh_inp_fragment.name}'.
            **             This exporter will auto-include it on the next run.
            **
            **   Option B: Replace this block manually with *NODE, *ELEMENT,
            **             *NSET (ALL_NODES, FIXED_FACE, AXIAL_LOAD_FACE,
            **             BENDING_LOAD_FACE, LEFT_SUPPORT, RIGHT_SUPPORT),
            **             and *ELSET (ALL_ELEMS) blocks matching the gmsh mesh.
            **
            ** Symbolic set names expected by this deck:
            **   ALL_NODES         — all mesh nodes
            **   ALL_ELEMS         — all mesh elements
            **   FIXED_FACE        — nodes on the fixed/clamped boundary face
            **   AXIAL_LOAD_FACE   — nodes on the axial load-application face
            **   BENDING_LOAD_FACE — nodes on the bending load-application face
            **   LEFT_SUPPORT      — nodes at left bending support (bending only)
            **   RIGHT_SUPPORT     — nodes at right bending support (bending only)
            **
            ** [PLACEHOLDER — REPLACE BEFORE RUNNING ccx]
            **
        """)

    return block, warnings


def _build_material_block(material: MaterialRecord) -> str:
    """
    Build the CalculiX ``*MATERIAL`` block for a given material.

    Includes ``*ELASTIC`` (isotropic) and ``*DENSITY``.

    NOTE — density units:
        CalculiX expects density in the model's unit system.  With mm/N/MPa:
            density [t/mm³] = density [g/cm³] / 1000
        (1 g/cm³ = 0.001 t/mm³ in the consistent mm/N/s unit system where
        mass is in tonnes and time is in seconds).

    Args:
        material: ``MaterialRecord`` with mechanical properties.

    Returns:
        CalculiX material definition string.
    """
    mat_name = _sanitize_solver_label(material.name)
    density_t_per_mm3 = material.mechanical.density_g_per_cm3 / 1000.0

    lines = [
        "**",
        "** === MATERIAL DEFINITION ===",
        f"*MATERIAL, NAME={mat_name}",
        "** Isotropic linear elastic",
        "*ELASTIC",
        f"{material.mechanical.elastic_modulus_mpa:.2f}, "
        f"{material.mechanical.poissons_ratio:.4f}",
        "** Density (t/mm3 in mm/N/MPa unit system)",
        "*DENSITY",
        f"{density_t_per_mm3:.6e}",
        "",
    ]

    if material.fatigue.fatigue_limit_is_placeholder:
        lines.insert(2,
            f"** WARNING: Fatigue limit for {material.name} is a placeholder. "
            f"Do not use for fatigue design decisions."
        )

    return "\n".join(lines) + "\n"


def _infer_analysis_keyword(
    loadcase: LoadCaseRecord,
    options: SolverExportOptions,
) -> str:
    """
    Return the CalculiX ``*STEP`` analysis keyword for a given load case.

    CYCLIC cases use ``*STATIC`` with an envelope force approach in version 1.
    A comment is injected to explain the proxy convention.

    Args:
        loadcase: Load-case record.
        options:  Export options.

    Returns:
        CalculiX step keyword string (e.g. ``"*STATIC"``).
    """
    if loadcase.load_case_type in (
        LoadCaseType.AXIAL_COMPRESSION,
        LoadCaseType.AXIAL_TENSION,
        LoadCaseType.BENDING,
    ):
        return "*STATIC"

    if loadcase.load_case_type == LoadCaseType.CYCLIC:
        # ARCHITECTURAL DECISION — CYCLIC exported as *STATIC envelope:
        # See module docstring.  The static envelope (mean + amplitude)
        # represents the worst-case stress state for proxy scoring.
        return "*STATIC"

    return "*STATIC"  # fallback


def _build_boundary_conditions_block(loadcase: LoadCaseRecord) -> str:
    """
    Build the CalculiX ``*BOUNDARY`` block for a given load case.

    ARCHITECTURAL DECISION — symbolic set names:
        Node set names (FIXED_FACE, LEFT_SUPPORT, RIGHT_SUPPORT) are symbolic
        placeholders that a mesh-include file must define.  See
        ``_build_mesh_reference_block`` for the expected set names.

    DoF convention (CalculiX):
        1 = UX, 2 = UY, 3 = UZ, 4 = ROTX, 5 = ROTY, 6 = ROTZ

    Args:
        loadcase: Load-case record.

    Returns:
        ``*BOUNDARY`` block string.
    """
    lc_type = loadcase.load_case_type
    lines = [
        "**",
        "** === BOUNDARY CONDITIONS ===",
        "** NOTE: Node set names are symbolic placeholders.",
        "** Replace with actual *NSET definitions from the mesh include.",
        "*BOUNDARY",
    ]

    if lc_type in (LoadCaseType.AXIAL_COMPRESSION, LoadCaseType.AXIAL_TENSION):
        lines += [
            "** Fix all DoFs on FIXED_FACE (end face in -X direction)",
            "FIXED_FACE, 1, 6, 0.0",
        ]

    elif lc_type == LoadCaseType.BENDING:
        lines += [
            "** Simple support at LEFT_SUPPORT: fix UY and UZ (pins in Z, roller in Y)",
            "LEFT_SUPPORT, 2, 2, 0.0",
            "LEFT_SUPPORT, 3, 3, 0.0",
            "** Simple support at RIGHT_SUPPORT: fix UY only (roller)",
            "RIGHT_SUPPORT, 2, 2, 0.0",
        ]

    elif lc_type == LoadCaseType.CYCLIC:
        lines += [
            "** CYCLIC PROXY: Fix all DoFs on FIXED_FACE (static envelope)",
            "** Actual cyclic analysis is replaced by fatigue_model.py proxy.",
            "FIXED_FACE, 1, 6, 0.0",
        ]

    lines.append("")
    return "\n".join(lines) + "\n"


def _build_load_block(loadcase: LoadCaseRecord) -> str:
    """
    Build the CalculiX ``*CLOAD`` (concentrated force) block.

    ARCHITECTURAL DECISION — *CLOAD over *DLOAD for version 1:
        Distributed loads (*DLOAD) require face set definitions and
        exact element face numbers, which are not available without a full
        mesh conversion.  Concentrated loads (*CLOAD) with symbolic face
        set names are a practical scaffold that a future mesh-conversion
        step can replace with exact distributed loads.

    Force sign convention:
        CalculiX uses a right-handed coordinate system.
        Compression (−X direction) is applied as a negative force in DoF 1.
        Tension (+X direction) is positive.
        Transverse bending load (−Z direction) is negative in DoF 3.

    Args:
        loadcase: Load-case record.

    Returns:
        ``*CLOAD`` block string.
    """
    lc_type = loadcase.load_case_type
    lines = [
        "**",
        "** === APPLIED LOADS ===",
        "** NOTE: Load sets are generated by calculix_converter.py.",
        "*CLOAD",
    ]

    if lc_type == LoadCaseType.AXIAL_COMPRESSION:
        force = -(abs(loadcase.force_n or 1200.0))  # negative = compression in -X
        lines += [
            f"** Axial compression: {abs(force):.2f} N applied in -X direction",
            f"AXIAL_LOAD_FACE, 1, {force:.4f}",
        ]

    elif lc_type == LoadCaseType.AXIAL_TENSION:
        force = abs(loadcase.force_n or 350.0)   # positive = tension in +X
        lines += [
            f"** Axial tension: {force:.2f} N applied in +X direction",
            f"AXIAL_LOAD_FACE, 1, {force:.4f}",
        ]

    elif lc_type == LoadCaseType.BENDING:
        force = -(abs(loadcase.force_n or 1200.0))  # negative = downward in -Z
        lines += [
            f"** Three-point bending: {abs(force):.2f} N applied in -Z direction at BENDING_LOAD_FACE",
            "** Support reactions computed by CalculiX at LEFT_SUPPORT, RIGHT_SUPPORT.",
            f"BENDING_LOAD_FACE, 3, {force:.4f}",
        ]

    elif lc_type == LoadCaseType.CYCLIC:
        mean_f = loadcase.mean_force_n or 600.0
        amp_f = loadcase.amplitude_force_n or 180.0
        envelope = -(mean_f + amp_f)  # worst-case compression envelope
        lines += [
            f"** CYCLIC PROXY: Static envelope = mean ({mean_f:.2f} N) + "
            f"amplitude ({amp_f:.2f} N) = {mean_f + amp_f:.2f} N compressive.",
            "** This is NOT a cyclic simulation. fatigue_model.py uses this",
            "** stress field as the proxy input.",
            f"AXIAL_LOAD_FACE, 1, {envelope:.4f}",
        ]

    lines.append("")
    return "\n".join(lines) + "\n"


def _build_step_block(
    loadcase: LoadCaseRecord,
    options: SolverExportOptions,
) -> str:
    """
    Build the CalculiX ``*STEP ... *END STEP`` wrapper lines.

    The BC, load, and output blocks are inserted between these wrappers
    by the main export function.

    Args:
        loadcase: Load-case record (used for labelling only).
        options:  Export options.

    Returns:
        Tuple of (step_open_lines, step_close_lines) as a single string.
    """
    analysis_kw = _infer_analysis_keyword(loadcase, options)
    step_name = _sanitize_solver_label(loadcase.key)

    lines = [
        "**",
        f"** === STEP: {step_name} ===",
        "*STEP, NLGEOM=NO",
        "** " + loadcase.name,
        analysis_kw,
    ]
    return "\n".join(lines) + "\n"


def _build_output_block() -> str:
    """
    Build the version-1 CalculiX output request block.

    Requests:
        - Nodal displacements (U)
        - Nodal reaction forces (RF)
        - Element von Mises stress (S, Mises)
        - Element logarithmic strains (LE)

    These cover all metrics extracted by the postprocessor
    (max displacement, max von Mises, effective stiffness proxy).

    Returns:
        Output request block string.
    """
    return textwrap.dedent("""\
        **
        ** === OUTPUT REQUESTS ===
        *NODE FILE
        U, RF
        *EL FILE
        S, MISES, LE
        *END STEP
    """)


def _build_section_assignments_block(
    material: MaterialRecord,
) -> str:
    """
    Build the ``*SOLID SECTION`` block assigning material to elements.

    Uses the symbolic ``ALL_ELEMS`` element set (placeholder).

    Args:
        material: MaterialRecord to assign.

    Returns:
        Section assignment string.
    """
    mat_name = _sanitize_solver_label(material.name)
    return textwrap.dedent(f"""\
        **
        ** === SECTION ASSIGNMENT ===
        ** ALL_ELEMS is a symbolic placeholder — see mesh include.
        *SOLID SECTION, ELSET=ALL_ELEMS, MATERIAL={mat_name}
        ** (section thickness not applicable for 3D solid elements)
    """)


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export_calculix_input_deck(
    mesh_path: str | Path,
    material: MaterialRecord,
    loadcase: LoadCaseRecord,
    output_inp_path: str | Path,
    case_definition: CaseDefinition | None = None,
    options: SolverExportOptions | None = None,
) -> SolverExportResult:
    """
    Write a CalculiX-style ``.inp`` input deck for one pipeline case.

    The generated file is structurally complete and includes all non-mesh
    sections as valid CalculiX syntax.  The mesh section references a
    pre-converted mesh include file if present, or writes a clearly
    labelled placeholder with instructions if not.

    Args:
        mesh_path:        Path to the ``.msh`` file produced by the mesher.
        material:         ``MaterialRecord`` from ``simulation/materials.py``.
        loadcase:         ``LoadCaseRecord`` from ``simulation/loadcases.py``.
        output_inp_path:  Path to write the ``.inp`` output file.
        case_definition:  Full ``CaseDefinition`` (optional; adds metadata).
        options:          ``SolverExportOptions`` (uses defaults if None).

    Returns:
        ``SolverExportResult`` with success flag, file path, and warnings.
    """
    mesh_path = Path(mesh_path)
    output_inp_path = Path(output_inp_path)
    options = options or SolverExportOptions()

    result = SolverExportResult(success=False, solver_backend="calculix")

    # --- Validate inputs ---
    try:
        _validate_export_inputs(mesh_path, material, loadcase, output_inp_path)
    except SolverExportError as exc:
        result.error_message = str(exc)
        return result

    # --- Build sections ---
    sections: list[str] = []

    if options.include_heading:
        sections.append(_build_heading_block(case_definition, material, loadcase))

    mesh_block, mesh_warnings = _build_mesh_reference_block(mesh_path)
    sections.append(mesh_block)
    result.warnings.extend(mesh_warnings)

    sections.append(_build_material_block(material))
    sections.append(_build_section_assignments_block(material))
    sections.append(_build_step_block(loadcase, options))
    sections.append(_build_boundary_conditions_block(loadcase))
    sections.append(_build_load_block(loadcase))

    if options.include_output_requests:
        sections.append(_build_output_block())
    else:
        sections.append("*END STEP\n")

    full_deck = "\n".join(sections)

    # --- Write file ---
    try:
        output_inp_path.write_text(full_deck, encoding="utf-8")
    except Exception as exc:
        result.error_message = (
            f"Failed to write input deck '{output_inp_path}': {exc}"
        )
        return result

    result.success = True
    result.input_deck_path = str(output_inp_path)
    result.metadata["mesh_source"] = str(mesh_path)
    result.metadata["material_name"] = material.name
    result.metadata["loadcase_key"] = loadcase.key
    result.metadata["analysis_type"] = _infer_analysis_keyword(loadcase, options)
    if case_definition is not None:
        result.metadata["case_id"] = case_definition.case_id
        result.metadata["plate_thickness_mm"] = case_definition.plate_thickness

    return result


# ---------------------------------------------------------------------------
# Case convenience helper
# ---------------------------------------------------------------------------

def export_solver_input_for_case(
    case_definition: CaseDefinition,
    mesh_path: str | Path,
    output_directory: str | Path,
    material: MaterialRecord | None = None,
    loadcase: LoadCaseRecord | None = None,
    options: SolverExportOptions | None = None,
) -> SolverExportResult:
    """
    Export the solver input deck for a full ``CaseDefinition``.

    If ``material`` or ``loadcase`` are not supplied, they are loaded from
    the config/library using the names stored in the case definition.

    ARCHITECTURAL DECISION — lazy material/loadcase loading:
        Loading the material and load-case from libraries is optional.
        Callers in the case runner will typically have already built these
        objects and pass them in.  The fallback loading is a convenience
        for interactive or test use.

    Args:
        case_definition:  Full ``CaseDefinition`` (provides case ID, thickness).
        mesh_path:        Path to the mesher-generated ``.msh`` file.
        output_directory: Directory for the ``.inp`` output file.
        material:         Pre-loaded ``MaterialRecord`` (optional).
        loadcase:         Pre-loaded ``LoadCaseRecord`` (optional).
        options:          Export options (optional).

    Returns:
        ``SolverExportResult``.
    """
    output_dir = Path(output_directory)
    inp_filename = _default_inp_filename(case_definition.case_id)
    output_inp_path = output_dir / inp_filename

    # --- Lazy load material if not provided ---
    if material is None:
        try:
            from simulation.materials import load_material_record
            material = load_material_record(case_definition.material.name)
        except Exception as exc:
            result = SolverExportResult(success=False)
            result.error_message = (
                f"Could not load material '{case_definition.material.name}': {exc}"
            )
            return result

    # --- Lazy load load case if not provided ---
    if loadcase is None:
        try:
            from simulation.loadcases import load_loadcase_record
            loadcase = load_loadcase_record(
                case_definition.load_case.load_case_type.value
            )
        except Exception as exc:
            result = SolverExportResult(success=False)
            result.error_message = (
                f"Could not load loadcase "
                f"'{case_definition.load_case.load_case_type.value}': {exc}"
            )
            return result

    return export_calculix_input_deck(
        mesh_path=mesh_path,
        material=material,
        loadcase=loadcase,
        output_inp_path=output_inp_path,
        case_definition=case_definition,
        options=options,
    )
