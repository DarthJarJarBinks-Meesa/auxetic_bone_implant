"""
src/designs/reentrant.py
=========================
Parametric 2D re-entrant honeycomb unit-cell generator.

This module implements the re-entrant honeycomb for the auxetic plate pipeline.
It subclasses ``BaseUnitCell`` and generates a 2D CadQuery Sketch profile
suitable for downstream tiling (lattice_builder.py) and extrusion (extruder.py).

GEOMETRY OVERVIEW:
    The unit cell consists of seven groups of rectangular members, all with
    cross-sectional width = wall_thickness:

        1. Top horizontal bar      — at y = +half_cell, spanning ±inner_half
        2. Bottom horizontal bar   — at y = −half_cell, spanning ±inner_half
        3. Upper-right diagonal    — from (+inner_half, +half_cell) to (+half_cell, 0)
        4. Upper-left diagonal     — from (−inner_half, +half_cell) to (−half_cell, 0)
        5. Lower-right diagonal    — from (+inner_half, −half_cell) to (+half_cell, 0)
        6. Lower-left diagonal     — from (−inner_half, −half_cell) to (−half_cell, 0)
        7. Side stubs (×2)         — square patches at (±half_cell, 0)

    The four diagonals form the characteristic "arrowhead / bowtie" re-entrant
    shape that gives the cell its negative Poisson's ratio behaviour.

PARAMETERIZATION:
    The re-entrant angle θ (reentrant_angle_deg) is the angle from the
    horizontal axis to the inclined diagonal centerline.

    Derived inner half-width:
        inner_half = half_cell × (1 − 1/tan(θ))

    Derived diagonal length (matches reference 4.18 mm at baseline):
        diagonal_length = half_cell / sin(θ)

    These formulae are derived by requiring the diagonal to connect
    (inner_half, half_cell) to (half_cell, 0) at angle θ from horizontal.

TILING:
    Cells tile edge-to-edge at cell_size intervals in both X and Y.
    Neighboring cells share:
        - horizontal bars at y = ±half_cell  (top/bottom neighbors)
        - diagonal apex nodes at x = ±half_cell, y = 0 (left/right neighbors)

REFERENCE PROPORTIONS (baseline cell_size=7.85, wall_thickness=1.5, angle=70°):
    top inner width   ≈ 4.99 mm  (reference: 5.36 mm; ~7% approximation)
    diagonal length   ≈ 4.18 mm  (matches reference exactly)
    side tab height   ≈ 1.50 mm  (= wall_thickness; matches reference)
"""

from __future__ import annotations

import math
from typing import Any

import cadquery as cq

from designs.base_cell import BaseUnitCell, SketchLike
from workflow.case_schema import DesignType, ReentrantParameters

# ---------------------------------------------------------------------------
# Module constants — baseline values from the architecture spec
# ---------------------------------------------------------------------------

_BASELINE_CELL_SIZE: float = 7.85       # mm
_BASELINE_WALL_THICKNESS: float = 1.5   # mm
_BASELINE_ANGLE_DEG: float = 70.0       # degrees


# ---------------------------------------------------------------------------
# Unit-cell class
# ---------------------------------------------------------------------------

class ReentrantUnitCell(BaseUnitCell):
    """
    Parametric 2D re-entrant honeycomb unit cell.

    Generates a CadQuery Sketch profile of a single unit cell in the XY
    plane.  The profile is suitable for:
        - tiling into a 5×3 lattice (``geometry/lattice_builder.py``)
        - extrusion into a 3D plate solid (``geometry/extruder.py``)

    Args:
        parameters: ``ReentrantParameters`` dataclass instance from
                    ``workflow/case_schema.py``.
    """

    DESIGN_SLUG: str = DesignType.REENTRANT.value

    def __init__(self, parameters: ReentrantParameters) -> None:
        if not isinstance(parameters, ReentrantParameters):
            raise TypeError(
                f"ReentrantUnitCell requires ReentrantParameters, "
                f"got {type(parameters).__name__}."
            )
        super().__init__(parameters)

    @property
    def params(self) -> ReentrantParameters:
        """Typed access to the re-entrant parameter set."""
        # ARCHITECTURAL DECISION — typed property alias:
        #   BaseUnitCell stores _parameters as DesignParameterSet (a union).
        #   This property narrows the type to ReentrantParameters so subclass
        #   code can access design-specific fields (e.g. params.wall_thickness)
        #   without repeated isinstance checks or type: ignore annotations
        #   scattered throughout the class.
        return self._parameters  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_parameters(self) -> None:
        """
        Design-level geometry feasibility checks for the re-entrant cell.

        These checks extend the schema-level field validation already
        performed by ``ReentrantParameters.validate()`` in case_schema.py.

        Additional checks:
          - wall_thickness < cell_size / 2  (member does not fill the cell)
          - reentrant_angle_deg ∈ (45°, 90°) exclusive
              Below 45°: inner_half ≤ 0; re-entrant topology collapses.
              Above or at 90°: tan(θ) undefined or zero; formula breaks.
          - inner_half > 0 (explicit guard; redundant above 45° but clear)

        ARCHITECTURAL DECISION — (45°, 90°) at design level vs (0°, 90°) at
        schema level:
            The schema allows any angle in (0, 90) for flexibility.  The
            design level adds the 45° lower bound because below that the
            derived inner_half becomes non-positive and the CadQuery Sketch
            construction would fail or produce degenerate geometry.
            Cases at exactly 45° are expected to be filtered by
            sweep_config.filters.skip_invalid_parameter_sets before reaching
            the geometry stage.

        Raises:
            ValueError: with a descriptive message on any violation.
        """
        p = self.params

        self._require_positive("cell_size", p.cell_size)
        self._require_positive("wall_thickness", p.wall_thickness)

        # Wall must not fill more than half the cell envelope
        self._require_less_than(
            "wall_thickness", p.wall_thickness, p.cell_size / 2.0
        )

        self._require_angle_range(
            "reentrant_angle_deg", p.reentrant_angle_deg, 45.0, 90.0
        )

        inner_half = self._inner_half()
        if inner_half <= 0.0:
            raise ValueError(
                f"reentrant_angle_deg={p.reentrant_angle_deg:.2f}° yields "
                f"inner_half={inner_half:.5f} mm ≤ 0. "
                f"An angle strictly above 45° is required for a re-entrant "
                f"(auxetic) topology with positive inner bar width."
            )

    # ------------------------------------------------------------------
    # Derived geometry quantities
    # ------------------------------------------------------------------

    def _half_cell(self) -> float:
        """Half the nominal cell size [mm]."""
        return self.params.cell_size / 2.0

    def _inner_half(self) -> float:
        """
        Half-width of the top/bottom horizontal bars [mm].

        Derivation
        ----------
        The diagonal member runs from (inner_half, half_cell) to (half_cell, 0)
        and makes angle θ (reentrant_angle_deg) with the horizontal axis::

            tan(θ) = half_cell / (half_cell − inner_half)
            ⟹  inner_half = half_cell × (1 − 1/tan(θ))

        Range
        -----
            θ = 45°  →  inner_half = 0          (degenerate, rejected)
            θ = 70°  →  inner_half ≈ 0.636 × half_cell
            θ = 89°  →  inner_half ≈ 0.983 × half_cell  (nearly full width)
        """
        theta = math.radians(self.params.reentrant_angle_deg)
        return self._half_cell() * (1.0 - 1.0 / math.tan(theta))

    def _diagonal_length(self) -> float:
        """
        Centerline length of each inclined diagonal member [mm].

        Derivation
        ----------
        With the diagonal spanning from (inner_half, half_cell) to (half_cell, 0)::

            Δx = half_cell / tan(θ)
            Δy = half_cell
            length = √(Δx² + Δy²) = half_cell / sin(θ)

        At baseline (θ=70°, cell_size=7.85 mm):
            length = 3.925 / sin(70°) ≈ 4.18 mm  ← matches reference exactly.
        """
        theta = math.radians(self.params.reentrant_angle_deg)
        return self._half_cell() / math.sin(theta)

    # ------------------------------------------------------------------
    # CadQuery construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _loc(x: float, y: float, angle_deg: float = 0.0) -> cq.Location:
        """
        Build a CadQuery ``Location`` at (x, y) in the XY plane, with
        optional Z-axis rotation.

        Used with ``cq.Sketch.push([loc]).rect(w, h)`` to position and
        orient rectangular members.

        Args:
            x, y:      Member centre position [mm].
            angle_deg: Rotation around Z axis [degrees]; aligns local X
                       with the member direction.

        Returns:
            ``cq.Location`` object.
        """
        return cq.Location(
            cq.Vector(x, y, 0.0),
            cq.Vector(0.0, 0.0, 1.0),
            angle_deg,
        )

    @staticmethod
    def _member_loc_len(
        x1: float, y1: float,
        x2: float, y2: float,
    ) -> tuple[cq.Location, float]:
        """
        Compute the center ``Location`` and length for a member spanning P1→P2.

        The returned ``Location`` sits at the member midpoint and is rotated
        so that its local X axis aligns along the P1→P2 direction.

        Use with::

            sketch.push([loc]).rect(length, wall_thickness)

        to place a finite-width rectangular member from P1 to P2.

        Args:
            x1, y1: Start point [mm].
            x2, y2: End point [mm].

        Returns:
            Tuple (Location at midpoint, member length [mm]).
        """
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx * dx + dy * dy)
        angle_deg = math.degrees(math.atan2(dy, dx))
        return ReentrantUnitCell._loc(cx, cy, angle_deg), length

    # ------------------------------------------------------------------
    # 2D geometry build
    # ------------------------------------------------------------------

    def build_2d(self) -> SketchLike:
        """
        Build and return the 2D re-entrant honeycomb profile in the XY plane.

        Construction strategy
        ----------------------
        Each structural member is a rectangle of width = wall_thickness
        placed via ``cq.Sketch.push([location]).rect(length, wall_thickness)``.
        Overlapping material at joints (e.g. where diagonals meet the
        horizontal bars) is resolved by the Sketch's `.clean()` operation,
        which unions all accumulating faces into a single connected region.

        ARCHITECTURAL DECISION — cq.Sketch for 2D boolean union:
            The Sketch API accumulates faces in 'a' (additive) mode and
            merges overlapping regions in `.clean()`.  This is more reliable
            and readable than manually computing a non-overlapping polygon
            outline or calling OCC BRep operations directly.  The Workplane
            approach would require explicit extrude-union-return-face steps,
            which are less idiomatic for purely 2D profiles.

        ARCHITECTURAL DECISION — side stubs at lateral tiling apexes:
            The four diagonal members alone produce sharp pointed tips at
            (±half_cell, 0).  Adding a small t×t square stub at each apex
            ensures robust material at the tiling connection points even
            when the isolated unit cell is inspected before lattice assembly.
            After tiling, neighboring cells contribute additional diagonal
            members that reinforce these nodes.

        Returns:
            ``cq.Sketch`` with all members unioned into a single 2D region.

        Raises:
            ValueError: if geometry degenerates (caught earlier by validate).
        """
        t  = self.params.wall_thickness
        hc = self._half_cell()
        ih = self._inner_half()

        # -- Diagonal member locations (center + alignment) + lengths ----
        # Each diagonal spans from (±ih, ±hc) to (±hc, 0).
        # The sign combinations give the four symmetric corners of the cell.
        ur_loc, ur_len = self._member_loc_len( ih,  hc,  hc, 0.0)  # upper-right
        ul_loc, ul_len = self._member_loc_len(-ih,  hc, -hc, 0.0)  # upper-left
        lr_loc, lr_len = self._member_loc_len( ih, -hc,  hc, 0.0)  # lower-right
        ll_loc, ll_len = self._member_loc_len(-ih, -hc, -hc, 0.0)  # lower-left

        # -- Assemble sketch ---------------------------------------------
        sk = (
            cq.Sketch()

            # Top horizontal bar (spans inner width; shared with top neighbor on tiling)
            .push([self._loc(0.0, hc)])
            .rect(2.0 * ih, t)

            # Bottom horizontal bar (shared with bottom neighbor on tiling)
            .push([self._loc(0.0, -hc)])
            .rect(2.0 * ih, t)

            # Four re-entrant diagonal members
            # Each is a rotated rectangle; length = half_cell/sin(θ)
            .push([ur_loc]).rect(ur_len, t)
            .push([ul_loc]).rect(ul_len, t)
            .push([lr_loc]).rect(lr_len, t)
            .push([ll_loc]).rect(ll_len, t)

            # Square stubs at lateral tiling connection apexes.
            # Ensures material at the points where left/right neighbor diagonals
            # will converge after lattice assembly.  Width and height both = t
            # (wall_thickness) so the stub is proportional to the members it joins.
            .push([self._loc( hc, 0.0)]).rect(t, t)
            .push([self._loc(-hc, 0.0)]).rect(t, t)

            # Union all additive faces into a single connected planar region
            .clean()
        )

        return sk

    # ------------------------------------------------------------------
    # Reference metadata
    # ------------------------------------------------------------------

    def reference_metadata(self) -> dict[str, Any]:
        """
        Return reference and derived geometry metadata for this unit cell.

        Useful for case logs, report traceability, and debugging.  Includes
        baseline proportions from the architecture spec for comparison with
        the current parametric values.

        Returns:
            Plain JSON-serialisable dict.
        """
        ih = self._inner_half()
        dl = self._diagonal_length()

        return {
            "design": "re_entrant_honeycomb",
            "version": 1,

            # Baseline from architecture spec
            "baseline_parameters": {
                "cell_size_mm": _BASELINE_CELL_SIZE,
                "wall_thickness_mm": _BASELINE_WALL_THICKNESS,
                "reentrant_angle_deg": _BASELINE_ANGLE_DEG,
            },

            # Current parametric values (serialised from dataclass)
            "current_parameters": self.params.to_dict(),

            # Derived quantities for the current parameter set
            "derived_geometry": {
                "half_cell_mm":         round(self._half_cell(), 4),
                "inner_half_width_mm":  round(ih, 4),
                "top_inner_width_mm":   round(2.0 * ih, 4),
                "diagonal_length_mm":   round(dl, 4),
            },

            # Reference proportions from the architecture spec reference model
            "reference_proportions_at_baseline": {
                "top_inner_width_mm": 5.36,
                "diagonal_length_mm": 4.18,
                "side_tab_width_mm":  1.25,
                "side_tab_height_mm": 1.50,
                "note": (
                    "diagonal_length matches the reference exactly at baseline. "
                    "top_inner_width is a ~7% approximation (4.99 mm vs 5.36 mm) "
                    "due to the simplified centerline-to-apex parameterization. "
                    "This is acceptable for version-1 parametric screening."
                ),
            },

            "geometry_notes": [
                "All members are idealised rectangles on a centerline skeleton.",
                "reentrant_angle_deg is measured from the horizontal (X) axis.",
                "inner_half = half_cell × (1 − 1/tan(θ)); requires θ > 45°.",
                "diagonal_length = half_cell / sin(θ); matches 4.18 mm reference at baseline.",
                "Horizontal bars at y=±half_cell are shared with top/bottom neighbors on tiling.",
                "Diagonal apexes at x=±half_cell, y=0 are shared with left/right neighbors.",
                "Side stubs at (±half_cell, 0) ensure apex material in isolated cell view.",
                "No screw holes; no anatomical plate outline (version-1 raw auxetic envelope).",
            ],
        }
