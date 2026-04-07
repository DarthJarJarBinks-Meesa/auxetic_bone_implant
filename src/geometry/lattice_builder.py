"""
src/geometry/lattice_builder.py
=================================
2D lattice builder for the auxetic plate pipeline.

This module takes a single 2D unit-cell geometry and tiles it into the
full 5×3 planar lattice in the XY plane.  The raw outer auxetic envelope
is preserved — no outer plate boundary is added, no clipping is applied.

PIPELINE POSITION:
    2D unit cell  →  [THIS MODULE]  →  5×3 lattice  →  extruder.py

ARCHITECTURAL DECISION — this module performs tiling only:
    No extrusion, no meshing, no material assignment, and no boundary
    clipping are performed here.  The module accepts 2D geometry and returns
    2D geometry.  The one responsibility is correct XY translation of tile
    copies and their union into a single connected planar region.

ARCHITECTURAL DECISION — centered lattice about the origin:
    The tile origins are computed so that the overall lattice centre of mass
    is at (0, 0).  This ensures that downstream extrusion and meshing modules
    see the model centred consistently regardless of lattice size.  The
    centering formula for repeats_x tiles of pitch p is::

        x_i = (i − (repeats_x − 1) / 2) × p    for i ∈ {0, …, repeats_x−1}

    producing symmetric offsets around x=0.

ARCHITECTURAL DECISION — Workplane-based tiling over Sketch-based tiling:
    CadQuery Sketch does not natively support translating an existing Sketch
    object as a whole.  The recommended idiom for positioning multiple copies
    of a 2D shape is to extract a ``cq.Workplane`` face from a Sketch and
    use ``Workplane.translate()`` for each copy before performing the union.
    The union is performed iteratively via ``Workplane.union()`` rather than
    a batch boolean, which is more reliable for complex auxetic outlines
    that may contain many faces.  If a unit cell is delivered as a
    ``cq.Workplane``, it is used directly; if it is a ``cq.Sketch``, it
    is normalised into a ``cq.Workplane`` face before tiling.

ARCHITECTURAL DECISION — iterative union over batch:
    Attempting a single N-operand boolean union on complex faces (re-entrant
    thin walls, chiral nodes) can be unreliable in OCC via CadQuery.
    Building the union incrementally (merge each new tile into the running
    accumulator) produces more predictable results and easier-to-debug
    intermediate states.  Performance is acceptable for N ≤ 15 (5×3 lattice).

UNITS: mm throughout.
"""

from __future__ import annotations

import logging
from typing import Any, Union

import cadquery as cq

from designs.base_cell import BaseUnitCell

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# ARCHITECTURAL DECISION — same SketchLike union as designs/base_cell.py:
#   Using the same alias makes the public API consistent across the geometry
#   layer.  Callers can pass the output of build_2d() directly into the
#   lattice builder without conversion.
Geometry2D = Union[cq.Workplane, cq.Sketch]
"""Type alias for 2D CadQuery geometry (Workplane or Sketch)."""


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class LatticeBuilderError(Exception):
    """
    Raised when lattice construction fails or receives invalid input.

    Covers:
      - invalid repeat counts (non-positive)
      - non-positive cell_size
      - None or unsupported geometry objects
      - failed boolean union during tile assembly
    """


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_cell_size(cell_size: float) -> None:
    """Assert cell_size is strictly positive."""
    if cell_size <= 0.0:
        raise LatticeBuilderError(
            f"cell_size must be strictly positive, got {cell_size}."
        )


def _validate_repeats(repeats_x: int, repeats_y: int) -> None:
    """Assert both repeat counts are strictly positive integers."""
    if repeats_x <= 0:
        raise LatticeBuilderError(
            f"repeats_x must be a positive integer, got {repeats_x}."
        )
    if repeats_y <= 0:
        raise LatticeBuilderError(
            f"repeats_y must be a positive integer, got {repeats_y}."
        )


def _validate_geometry(geometry: Any, source_label: str = "unit_cell_geometry") -> None:
    """Assert the geometry object is non-None and a recognised CadQuery type."""
    if geometry is None:
        raise LatticeBuilderError(
            f"'{source_label}' is None.  The unit-cell build_2d() method must "
            f"return a non-None CadQuery geometry object."
        )
    if not isinstance(geometry, (cq.Workplane, cq.Sketch)):
        raise LatticeBuilderError(
            f"'{source_label}' must be a cq.Workplane or cq.Sketch, "
            f"got {type(geometry).__name__}."
        )


# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------

def _to_workplane(geometry: Geometry2D) -> cq.Workplane:
    """
    Normalise any supported 2D geometry object into a ``cq.Workplane``
    containing a single planar face in the XY plane.

    ARCHITECTURAL DECISION — Sketch → Workplane normalisation:
        The tiling loop uses ``Workplane.translate()`` and
        ``Workplane.union()`` for reliability.  CadQuery Sketch objects do
        not expose ``translate()`` directly.  Normalising here keeps the
        tiling loop type-consistent and avoids scattered isinstance checks.

    If the input is already a ``cq.Workplane``, it is returned unchanged.
    If it is a ``cq.Sketch``, its face(s) are converted to a ``Workplane``
    by building a 3D extrude-to-zero equivalent face workplane via the
    Sketch's ``finalize()`` path (or face extraction if available).

    Args:
        geometry: 2D CadQuery geometry (Workplane or Sketch).

    Returns:
        ``cq.Workplane`` carrying the 2D face(s).

    Raises:
        LatticeBuilderError: if conversion fails.
    """
    if isinstance(geometry, cq.Workplane):
        return geometry

    # Sketch → Workplane:
    # cq.Sketch stores accumulated faces; the most robust extraction is to
    # call .finalize() which returns the parent Workplane if the Sketch was
    # created from one, or to convert via an extrude+projection path.
    # For standalone Sketch objects (created with cq.Sketch()), we build a
    # thin extrusion and retrieve the bottom face as a planar Workplane.
    try:
        # Try .finalize() first — works when the Sketch was created from a
        # Workplane context (e.g. wp.sketch()...finalize()).
        wp = geometry.finalize()
        if isinstance(wp, cq.Workplane):
            return wp
    except (AttributeError, Exception):
        pass

    # Fallback: extrude the sketch by a tiny thickness and extract the
    # bottom face as a standalone Workplane face.
    # This is a robust idiom for standalone cq.Sketch() objects.
    try:
        EPSILON = 0.001  # mm — negligible; we only want the face topology
        solid_wp = (
            cq.Workplane("XY")
            .add(geometry)
            .extrude(EPSILON)
        )
        # Select the face at z=0 (bottom of the thin extrusion)
        face_wp = solid_wp.faces("<Z")
        return face_wp
    except Exception as exc:
        raise LatticeBuilderError(
            f"Failed to convert cq.Sketch to cq.Workplane for tiling: {exc}.  "
            f"Ensure build_2d() returns a valid, non-degenerate Sketch."
        ) from exc


# ---------------------------------------------------------------------------
# Tile offset computation
# ---------------------------------------------------------------------------

def compute_tile_offsets(
    cell_size: float,
    repeats_x: int,
    repeats_y: int,
) -> list[tuple[float, float]]:
    """
    Compute the XY translation offsets for all tiles in the lattice,
    centred about the origin.

    Uses edge-to-edge pitch equal to ``cell_size`` in both directions.

    Centering formula::

        x_i = (i − (repeats_x − 1) / 2) × cell_size
        y_j = (j − (repeats_y − 1) / 2) × cell_size

    for i ∈ {0, …, repeats_x−1} and j ∈ {0, …, repeats_y−1}.

    Args:
        cell_size:  Nominal square envelope side length [mm].
        repeats_x:  Number of unit cells in the X direction.
        repeats_y:  Number of unit cells in the Y direction.

    Returns:
        List of (x, y) offsets, one per tile, in row-major order
        (y varies slowest).

    Example (3×2 lattice, cell_size=7.85)::

        offsets = compute_tile_offsets(7.85, 3, 2)
        # x in {−7.85, 0.0, +7.85}, y in {−3.925, +3.925}
    """
    _validate_cell_size(cell_size)
    _validate_repeats(repeats_x, repeats_y)

    x_centre_offset = (repeats_x - 1) / 2.0 * cell_size
    y_centre_offset = (repeats_y - 1) / 2.0 * cell_size

    offsets: list[tuple[float, float]] = []
    for j in range(repeats_y):
        for i in range(repeats_x):
            x = i * cell_size - x_centre_offset
            y = j * cell_size - y_centre_offset
            offsets.append((x, y))

    return offsets


# ---------------------------------------------------------------------------
# Lattice metadata
# ---------------------------------------------------------------------------

def lattice_metadata(
    cell_size: float,
    repeats_x: int,
    repeats_y: int,
) -> dict[str, Any]:
    """
    Return a plain metadata dictionary describing the overall lattice
    geometry.

    Useful for case logs, run metadata files, and reporting modules.

    Args:
        cell_size:  Nominal square unit-cell envelope side length [mm].
        repeats_x:  Number of unit cells in X.
        repeats_y:  Number of unit cells in Y.

    Returns:
        JSON-serialisable dict with the following keys:

        - ``repeats_x``             : int
        - ``repeats_y``             : int
        - ``cell_size_mm``          : float
        - ``pitch_x_mm``            : float  (= cell_size; edge-to-edge)
        - ``pitch_y_mm``            : float  (= cell_size; edge-to-edge)
        - ``overall_nominal_width_mm``  : float  (= repeats_x × cell_size)
        - ``overall_nominal_height_mm`` : float  (= repeats_y × cell_size)
        - ``nominal_tile_count``    : int    (= repeats_x × repeats_y)
        - ``centered_about_origin`` : bool   (always True)
    """
    _validate_cell_size(cell_size)
    _validate_repeats(repeats_x, repeats_y)

    return {
        "repeats_x":                  repeats_x,
        "repeats_y":                  repeats_y,
        "cell_size_mm":               cell_size,
        "pitch_x_mm":                 cell_size,   # edge-to-edge; no gap
        "pitch_y_mm":                 cell_size,   # edge-to-edge; no gap
        "overall_nominal_width_mm":   repeats_x * cell_size,
        "overall_nominal_height_mm":  repeats_y * cell_size,
        "nominal_tile_count":         repeats_x * repeats_y,
        "centered_about_origin":      True,
    }


# ---------------------------------------------------------------------------
# Core tiling logic
# ---------------------------------------------------------------------------

def _tile_workplane(
    base_wp: cq.Workplane,
    offsets: list[tuple[float, float]],
) -> cq.Workplane:
    """
    Tile a base ``cq.Workplane`` face geometry across the supplied XY offsets
    and return a single unioned ``cq.Workplane``.

    ARCHITECTURAL DECISION — iterative union, not batch:
        See module docstring.  Incremental union is more reliable for complex
        auxetic face topologies than a single N-way boolean.

    ARCHITECTURAL DECISION — Z translation is zero:
        All tiling is strictly in XY (z=0).  The Z component is always 0.0.
        This is enforced by the translate call below.

    Args:
        base_wp:  ``cq.Workplane`` containing the base unit-cell face.
        offsets:  List of (dx, dy) translation offsets for each tile.

    Returns:
        Single ``cq.Workplane`` with all tiles unioned into one planar body.

    Raises:
        LatticeBuilderError: if no offsets are supplied or union fails.
    """
    if not offsets:
        raise LatticeBuilderError("No tile offsets supplied; cannot build lattice.")

    # Initialise the accumulator with the first tile (offset (dx, dy)).
    dx0, dy0 = offsets[0]
    accumulator: cq.Workplane = base_wp.translate((dx0, dy0, 0.0))

    for dx, dy in offsets[1:]:
        try:
            tile = base_wp.translate((dx, dy, 0.0))
            accumulator = accumulator.union(tile)
        except Exception as exc:
            raise LatticeBuilderError(
                f"Boolean union failed while tiling at offset ({dx:.4f}, {dy:.4f}) mm: "
                f"{exc}.  The unit-cell geometry may contain self-intersecting or "
                f"degenerate faces.  Check build_2d() output for the design."
            ) from exc

    return accumulator


# ---------------------------------------------------------------------------
# Public builder functions
# ---------------------------------------------------------------------------

def build_lattice_from_geometry(
    unit_cell_geometry: Geometry2D,
    cell_size: float,
    repeats_x: int = 5,
    repeats_y: int = 3,
) -> cq.Workplane:
    """
    Tile already-built 2D geometry into a rectangular lattice in the XY plane.

    Accepts a ``cq.Workplane`` or ``cq.Sketch`` representing one unit cell
    and tiles it edge-to-edge into a ``repeats_x × repeats_y`` grid.  The
    lattice is centred about the origin.

    This function is useful when the 2D unit-cell geometry has already been
    generated (e.g. in a cached workflow step) and does not need to be
    rebuilt via a ``BaseUnitCell`` object.

    Args:
        unit_cell_geometry: 2D CadQuery geometry for one unit cell,
                            centred at the origin.
        cell_size:          Nominal square envelope side length [mm].
                            Determines the tiling pitch in both X and Y.
        repeats_x:          Number of unit cells in X (default 5).
        repeats_y:          Number of unit cells in Y (default 3).

    Returns:
        ``cq.Workplane`` containing the full tiled lattice as a single
        connected planar body.

    Raises:
        LatticeBuilderError: on invalid inputs or failed tiling.

    Example::

        lattice_wp = build_lattice_from_geometry(sketch, cell_size=7.85)
    """
    _validate_geometry(unit_cell_geometry, "unit_cell_geometry")
    _validate_cell_size(cell_size)
    _validate_repeats(repeats_x, repeats_y)

    logger.debug(
        "Building %d×%d lattice from raw geometry (cell_size=%.3f mm).",
        repeats_x, repeats_y, cell_size,
    )

    base_wp = _to_workplane(unit_cell_geometry)
    offsets = compute_tile_offsets(cell_size, repeats_x, repeats_y)
    lattice_wp = _tile_workplane(base_wp, offsets)

    logger.info(
        "Lattice built: %d×%d = %d tiles, nominal size %.3f × %.3f mm.",
        repeats_x, repeats_y, repeats_x * repeats_y,
        repeats_x * cell_size, repeats_y * cell_size,
    )

    return lattice_wp


def build_lattice_from_unit_cell(
    unit_cell: BaseUnitCell,
    repeats_x: int = 5,
    repeats_y: int = 3,
) -> cq.Workplane:
    """
    Build the full tiled lattice from a ``BaseUnitCell`` instance.

    Calls ``unit_cell.build_and_validate_2d()`` to generate the validated
    2D geometry, then tiles it into a ``repeats_x × repeats_y`` lattice.

    This is the primary entry point used by the geometry pipeline when
    constructing from a fresh case run.

    Args:
        unit_cell:  Concrete ``BaseUnitCell`` subclass instance (e.g.
                    ``ReentrantUnitCell``, ``RotatingSquareUnitCell``,
                    ``TetrachiralUnitCell``).  Must have valid parameters.
        repeats_x:  Number of unit cells in X (default 5).
        repeats_y:  Number of unit cells in Y (default 3).

    Returns:
        ``cq.Workplane`` containing the full tiled lattice as a single
        connected planar body.

    Raises:
        LatticeBuilderError: if the unit cell is None, if ``build_2d()``
                             fails, or if tiling fails.

    Example::

        params = ReentrantParameters(cell_size=7.85, wall_thickness=1.5,
                                     reentrant_angle_deg=70.0)
        cell = ReentrantUnitCell(params)
        lattice_wp = build_lattice_from_unit_cell(cell)
    """
    if unit_cell is None:
        raise LatticeBuilderError(
            "unit_cell must not be None.  Provide a concrete BaseUnitCell "
            "subclass instance."
        )
    if not isinstance(unit_cell, BaseUnitCell):
        raise LatticeBuilderError(
            f"unit_cell must be a BaseUnitCell subclass instance, "
            f"got {type(unit_cell).__name__}."
        )

    logger.debug(
        "Building %d×%d lattice from unit cell: %s",
        repeats_x, repeats_y, unit_cell,
    )

    # build_and_validate_2d() runs full validation before geometry generation.
    try:
        geometry = unit_cell.build_and_validate_2d()
    except ValueError as exc:
        raise LatticeBuilderError(
            f"Unit cell geometry build failed for {unit_cell}: {exc}"
        ) from exc

    cell_size = unit_cell.cell_size

    return build_lattice_from_geometry(
        unit_cell_geometry=geometry,
        cell_size=cell_size,
        repeats_x=repeats_x,
        repeats_y=repeats_y,
    )


# ---------------------------------------------------------------------------
# CaseDefinition convenience helper
# ---------------------------------------------------------------------------

def build_lattice_from_case(
    case_definition: Any,
    unit_cell: BaseUnitCell | None = None,
) -> cq.Workplane:
    """
    Build the tiled lattice using a case definition and an optional
    pre-built unit-cell object.

    Uses ``case_definition.lattice_repeats_x`` and
    ``case_definition.lattice_repeats_y`` if available, defaulting to 5×3.

    ARCHITECTURAL DECISION — duck-typed on case_definition:
        ``CaseDefinition`` is not imported explicitly here to avoid coupling
        the geometry layer to the workflow layer.  Any object that provides
        the expected attributes (``.lattice_repeats_x``, ``.lattice_repeats_y``,
        ``.design_parameters.cell_size``) works.

    If ``unit_cell`` is provided, it is used directly.  Otherwise, this
    function raises ``LatticeBuilderError`` — it does not instantiate a
    unit cell itself, to avoid importing the factory here and creating a
    circular dependency chain.  Callers should use
    ``unitcell_factory.create_unit_cell_from_case()`` to build the unit cell
    and pass it in.

    Args:
        case_definition: Object with lattice repeat attributes (e.g.
                         ``CaseDefinition``).
        unit_cell:       Pre-built ``BaseUnitCell`` instance.  Required —
                         pass the output of ``create_unit_cell_from_case()``.

    Returns:
        ``cq.Workplane`` containing the full tiled lattice.

    Raises:
        LatticeBuilderError: if ``unit_cell`` is not provided or is invalid.

    Example::

        cell = create_unit_cell_from_case(case)
        lattice_wp = build_lattice_from_case(case, unit_cell=cell)
    """
    if unit_cell is None:
        raise LatticeBuilderError(
            "build_lattice_from_case() requires a pre-built unit_cell object.  "
            "Use unitcell_factory.create_unit_cell_from_case(case) to build one "
            "and pass it as the unit_cell argument."
        )

    # Extract repeat counts from the case definition via duck-typing.
    repeats_x: int = getattr(case_definition, "lattice_repeats_x", 5)
    repeats_y: int = getattr(case_definition, "lattice_repeats_y", 3)

    return build_lattice_from_unit_cell(
        unit_cell=unit_cell,
        repeats_x=repeats_x,
        repeats_y=repeats_y,
    )
