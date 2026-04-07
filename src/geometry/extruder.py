"""
src/geometry/extruder.py
=========================
2D-to-3D extrusion module for the auxetic plate pipeline.

This module takes a 2D planar lattice geometry and extrudes it into a 3D
solid plate along the Z axis using the case-defined ``plate_thickness``.

PIPELINE POSITION:
    5×3 lattice (2D, XY)  →  [THIS MODULE]  →  3D solid  →  mesher.py

ARCHITECTURAL DECISION — this module is the boundary between 2D and 3D:
    All geometry above this module in the pipeline (unit-cell generators,
    lattice builder) works in the XY plane (z=0) and returns 2D profiles.
    Everything below this module (mesher, solver exporter) works in 3D.
    Keeping this separation means the 2D design layer can be tested and
    validated independently of the 3D extrusion, and the plate thickness
    can be swept without regenerating 2D geometry.

ARCHITECTURAL DECISION — ``plate_thickness`` is the real sweep variable:
    The reference STEP files were extruded to 7 mm (reference geometry only).
    The actual extrusion depth used by the pipeline is always ``plate_thickness``
    from ``CaseDefinition``.  This module enforces that no other thickness
    source is used.

ARCHITECTURAL DECISION — centered=True as default:
    Centering the extrusion about the XY plane (z = ±plate_thickness/2)
    places the plate's neutral surface at z=0.  This is the conventional
    FE model origin and makes symmetric boundary condition setup simpler
    in solver_exporter.py.  Callers that need a z=0 bottom face (e.g. for
    contact boundary conditions) can set centered=False.

ARCHITECTURAL DECISION — cq.Workplane extrusion path, not Sketch.extrude:
    CadQuery Sketch objects support ``.extrude()`` only when the Sketch was
    created from a Workplane context (``wp.sketch()...finalize().extrude()``).
    Standalone ``cq.Sketch()`` objects (used by our design classes) do not
    expose ``.extrude()`` directly.  Normalising all input to a
    ``cq.Workplane`` carrying a 2D face and calling ``Workplane.extrude()``
    provides a single reliable extrusion path regardless of input type.

ARCHITECTURAL DECISION — no screw holes, no outer plate outline:
    Version 1 extrudes the raw tiled auxetic envelope only.  These features
    are out of scope and must not be added here.

UNITS: mm throughout.
"""

from __future__ import annotations

import logging
from typing import Any, Union

import cadquery as cq

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Geometry2D = Union[cq.Workplane, cq.Sketch]
"""Type alias for 2D planar CadQuery geometry accepted as input."""

Solid3D = cq.Workplane
"""Type alias for 3D solid CadQuery geometry returned as output."""


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class ExtrusionError(Exception):
    """
    Raised when extrusion fails or receives invalid input.

    Covers:
      - non-positive ``plate_thickness``
      - None or unsupported input geometry objects
      - CadQuery extrusion failures (degenerate faces, OCC errors)
    """


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_plate_thickness(plate_thickness: float) -> None:
    """
    Assert that ``plate_thickness`` is strictly positive.

    Args:
        plate_thickness: Extrusion depth [mm].

    Raises:
        ExtrusionError: if ``plate_thickness <= 0``.
    """
    if plate_thickness <= 0.0:
        raise ExtrusionError(
            f"plate_thickness must be strictly positive, got {plate_thickness} mm.  "
            f"Check the case definition or extrusion call site."
        )


def _validate_planar_geometry(planar_geometry: Any) -> None:
    """
    Assert that ``planar_geometry`` is non-None and a recognised CadQuery type.

    Args:
        planar_geometry: Object to check.

    Raises:
        ExtrusionError: if None or not a ``cq.Workplane`` / ``cq.Sketch``.
    """
    if planar_geometry is None:
        raise ExtrusionError(
            "planar_geometry is None.  The lattice builder must return a valid "
            "CadQuery geometry object before extrusion."
        )
    if not isinstance(planar_geometry, (cq.Workplane, cq.Sketch)):
        raise ExtrusionError(
            f"planar_geometry must be a cq.Workplane or cq.Sketch, "
            f"got {type(planar_geometry).__name__}."
        )


# ---------------------------------------------------------------------------
# Normalisation helper (2D input → Workplane ready for extrude)
# ---------------------------------------------------------------------------

def _to_extrudable_workplane(planar_geometry: Geometry2D) -> cq.Workplane:
    """
    Normalise any supported 2D geometry into a ``cq.Workplane`` that is ready
    for a ``.extrude()`` call.

    ARCHITECTURAL DECISION — single normalisation path:
        See module docstring.  Both ``cq.Workplane`` and ``cq.Sketch`` inputs
        are normalised to a common ``cq.Workplane``-based path here so the
        extrusion function below does not branch on type.

    For a ``cq.Workplane`` input:
        The existing Workplane is used directly if it already holds 2D faces
        (typical output from ``lattice_builder.py``).  If it holds a 3D solid,
        we attempt to select the bottom face at z=0 to recover the planar
        profile.

    For a ``cq.Sketch`` input:
        We use the thin-extrude + face-extraction idiom to convert the Sketch
        faces into a ``cq.Workplane`` profile, identical to the strategy in
        ``lattice_builder._to_workplane()``.

    Args:
        planar_geometry: 2D CadQuery geometry.

    Returns:
        ``cq.Workplane`` positioned at z=0 carrying the 2D face(s).

    Raises:
        ExtrusionError: if normalisation fails.
    """
    if isinstance(planar_geometry, cq.Workplane):
        return planar_geometry

    # Sketch → Workplane normalisation (same approach as lattice_builder)
    try:
        wp = planar_geometry.finalize()
        if isinstance(wp, cq.Workplane):
            return wp
    except (AttributeError, Exception):
        pass

    try:
        EPSILON = 0.001  # mm — negligible thin extrude to recover face topology
        solid_wp = (
            cq.Workplane("XY")
            .add(planar_geometry)
            .extrude(EPSILON)
        )
        return solid_wp.faces("<Z")
    except Exception as exc:
        raise ExtrusionError(
            f"Failed to convert cq.Sketch to extrudable cq.Workplane: {exc}.  "
            f"Ensure the lattice geometry is a valid non-degenerate Sketch."
        ) from exc


# ---------------------------------------------------------------------------
# Core extrusion function
# ---------------------------------------------------------------------------

def extrude_planar_geometry(
    planar_geometry: Geometry2D,
    plate_thickness: float,
    centered: bool = True,
) -> Solid3D:
    """
    Extrude a 2D planar geometry into a 3D solid plate along the Z axis.

    This is the primary extrusion entry point used by the pipeline.

    Args:
        planar_geometry: 2D CadQuery geometry (``cq.Workplane`` or
                         ``cq.Sketch``) representing the tiled auxetic
                         lattice profile in the XY plane at z=0.
        plate_thickness: Extrusion depth [mm].  Must be positive.
                         This is the real sweep variable from
                         ``CaseDefinition``; do not substitute a
                         hard-coded value.
        centered:        If ``True`` (default), extrude symmetrically about
                         the XY plane so the plate occupies
                         z ∈ [−t/2, +t/2].
                         If ``False``, extrude in the +Z direction only,
                         so the plate occupies z ∈ [0, +t].

    Returns:
        ``cq.Workplane`` containing the 3D solid plate.

    Raises:
        ExtrusionError: on invalid inputs or if the CadQuery extrusion fails.

    Example::

        solid = extrude_planar_geometry(lattice_wp, plate_thickness=3.5)
    """
    _validate_plate_thickness(plate_thickness)
    _validate_planar_geometry(planar_geometry)

    logger.debug(
        "Extruding 2D geometry: plate_thickness=%.3f mm, centered=%s.",
        plate_thickness, centered,
    )

    base_wp = _to_extrudable_workplane(planar_geometry)

    try:
        if centered:
            # ARCHITECTURAL DECISION — centered extrusion via both=True:
            #   CadQuery's ``Workplane.extrude(distance, both=True)`` extrudes
            #   by distance/2 in each direction (±Z), placing the neutral
            #   surface at the original workplane (z=0).  This is the cleanest
            #   way to achieve plate centering without a manual translate step.
            solid = base_wp.extrude(plate_thickness / 2.0, both=True)
        else:
            # One-sided extrusion in +Z from the XY plane.
            solid = base_wp.extrude(plate_thickness)
    except Exception as exc:
        raise ExtrusionError(
            f"CadQuery extrusion failed (plate_thickness={plate_thickness} mm, "
            f"centered={centered}): {exc}.  "
            f"The input planar geometry may contain degenerate or "
            f"self-intersecting faces.  Check the lattice builder output."
        ) from exc

    logger.info(
        "Extrusion complete: plate_thickness=%.3f mm, centered=%s.",
        plate_thickness, centered,
    )

    return solid


# ---------------------------------------------------------------------------
# Lattice-specific convenience wrapper
# ---------------------------------------------------------------------------

def extrude_lattice_geometry(
    lattice_geometry: Geometry2D,
    plate_thickness: float,
    centered: bool = True,
) -> Solid3D:
    """
    Extrude a tiled auxetic lattice geometry into a 3D solid plate.

    This function is a named wrapper around ``extrude_planar_geometry`` with
    identical behaviour.  Keeping a dedicated name makes the pipeline call
    chain self-documenting::

        lattice_wp = build_lattice_from_unit_cell(cell)
        solid      = extrude_lattice_geometry(lattice_wp, plate_thickness=3.5)

    ARCHITECTURAL DECISION — named wrapper, not alias:
        Using a named function (rather than ``extrude_lattice_geometry =
        extrude_planar_geometry``) allows the function to gain lattice-specific
        behaviour (e.g. lattice-aware validation, logging context) in a future
        version without changing the call signature used by downstream modules.

    Args:
        lattice_geometry: 2D CadQuery tiled lattice geometry.
        plate_thickness:  Extrusion depth [mm].
        centered:         Centring mode (see ``extrude_planar_geometry``).

    Returns:
        ``cq.Workplane`` containing the 3D solid.

    Raises:
        ExtrusionError: on invalid inputs or extrusion failure.
    """
    return extrude_planar_geometry(
        planar_geometry=lattice_geometry,
        plate_thickness=plate_thickness,
        centered=centered,
    )


# ---------------------------------------------------------------------------
# CaseDefinition convenience helper
# ---------------------------------------------------------------------------

def extrude_from_case(
    case_definition: Any,
    planar_geometry: Geometry2D,
    centered: bool = True,
) -> Solid3D:
    """
    Read ``plate_thickness`` from a case definition and extrude the supplied
    planar geometry.

    ARCHITECTURAL DECISION — duck-typed on ``case_definition.plate_thickness``:
        ``CaseDefinition`` is not imported explicitly to avoid coupling the
        geometry layer to the workflow layer.  Any object that exposes a
        ``.plate_thickness`` float attribute works.

    Args:
        case_definition: Object with a ``.plate_thickness`` float attribute
                         (typically a ``CaseDefinition`` instance).
        planar_geometry: 2D tiled lattice geometry.
        centered:        Centring mode (see ``extrude_planar_geometry``).

    Returns:
        ``cq.Workplane`` containing the 3D solid.

    Raises:
        ExtrusionError: if ``plate_thickness`` is missing or invalid, or
                        if extrusion fails.

    Example::

        solid = extrude_from_case(case, lattice_wp)
    """
    if not hasattr(case_definition, "plate_thickness"):
        raise ExtrusionError(
            f"case_definition has no 'plate_thickness' attribute.  "
            f"Got: {type(case_definition).__name__}.  "
            f"Ensure the case definition object is a valid CaseDefinition."
        )

    plate_thickness: float = case_definition.plate_thickness

    return extrude_planar_geometry(
        planar_geometry=planar_geometry,
        plate_thickness=plate_thickness,
        centered=centered,
    )


# ---------------------------------------------------------------------------
# Metadata helper
# ---------------------------------------------------------------------------

def extrusion_metadata(
    plate_thickness: float,
    centered: bool = True,
) -> dict[str, Any]:
    """
    Return a plain metadata dictionary describing the extrusion configuration.

    Useful for case log files, run metadata JSON, and reporting.

    Args:
        plate_thickness: Extrusion depth [mm].
        centered:        Whether the extrusion is centred about z=0.

    Returns:
        JSON-serialisable dict with the following keys:

        - ``plate_thickness_mm``  : float  — total plate thickness
        - ``centered``            : bool   — centering mode
        - ``extrusion_axis``      : str    — always ``"Z"``
        - ``z_min_mm``            : float  — bottom z coordinate
        - ``z_max_mm``            : float  — top z coordinate
        - ``mode``                : str    — ``"centered"`` or ``"one_sided_positive_z"``
        - ``note``                : str    — version-1 constraint reminder
    """
    _validate_plate_thickness(plate_thickness)

    if centered:
        z_min = -plate_thickness / 2.0
        z_max =  plate_thickness / 2.0
        mode = "centered"
    else:
        z_min = 0.0
        z_max = plate_thickness
        mode = "one_sided_positive_z"

    return {
        "plate_thickness_mm": plate_thickness,
        "centered":           centered,
        "extrusion_axis":     "Z",
        "z_min_mm":           z_min,
        "z_max_mm":           z_max,
        "mode":               mode,
        "note": (
            "Raw tiled auxetic envelope extruded only.  "
            "No screw holes and no outer plate outline (version-1 scope).  "
            "plate_thickness is the real sweep variable; the reference STEP "
            "extrusion of 7.0 mm is not used here."
        ),
    }


# ---------------------------------------------------------------------------
# Optional bounding-box helper
# ---------------------------------------------------------------------------

def solid_bounding_box_dimensions(solid: Solid3D) -> tuple[float, float, float]:
    """
    Return the approximate ``(width_x, height_y, depth_z)`` bounding-box
    dimensions of a 3D solid ``cq.Workplane`` in mm.

    Uses CadQuery's ``BoundingBox`` via ``solid.val().BoundingBox()`` where
    available.  If the bounding box cannot be computed (e.g. empty solid,
    OCC kernel error), returns ``(0.0, 0.0, 0.0)`` and logs a warning rather
    than raising, so that metadata collection does not abort a case run.

    ARCHITECTURAL DECISION — soft failure (warn, return zeros):
        Bounding box is used for metadata and debugging only, not for
        geometry correctness.  A failure here should not abort a sweep case.
        Errors in the solid itself will surface during meshing.

    Args:
        solid: ``cq.Workplane`` containing a 3D solid.

    Returns:
        Tuple ``(xsize_mm, ysize_mm, zsize_mm)`` or ``(0.0, 0.0, 0.0)``
        on failure.
    """
    try:
        bb = solid.val().BoundingBox()
        x_size = bb.xmax - bb.xmin
        y_size = bb.ymax - bb.ymin
        z_size = bb.zmax - bb.zmin
        return (x_size, y_size, z_size)
    except Exception as exc:
        logger.warning(
            "Could not compute bounding box for solid: %s.  "
            "Returning (0.0, 0.0, 0.0).",
            exc,
        )
        return (0.0, 0.0, 0.0)
