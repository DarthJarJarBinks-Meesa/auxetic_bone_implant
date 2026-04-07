"""
src/designs/rotating_square.py
================================
Parametric 2D rotating-square unit-cell generator.

This module implements the rotating-square auxetic design for the plate
pipeline.  It subclasses ``BaseUnitCell`` and generates a 2D CadQuery
Sketch profile suitable for downstream tiling and extrusion.

GEOMETRY OVERVIEW:
    The unit cell contains four rigid square plates arranged one per quadrant,
    each rotated by ``rotation_angle_deg`` (θ), connected at their inner
    corners by four slender hinge bridges of width ``hinge_thickness``.

    Layout (top view, baseline θ=22.5°)::

        NW ───── NE
        │    ╱╲    │
        │  ╱    ╲  │
        │╱   ◇   ╲│   ← central diamond formed by 4 inner corners
        │╲        ╱│      connected by 4 hinge bridges
        │  ╲    ╱  │
        │    ╲╱    │
        SW ───── SE

    The 4 outer corners of each square touch the cell boundary exactly, so
    cells tile edge-to-edge with no overlap and no gap.

PARAMETERIZATION:
    Rigid square side length (derived from cell_size and θ):

        s = (cell_size / 2) / (cos θ + sin θ)

    This formula ensures that when each rotated square is placed in its
    cell_size/2 × cell_size/2 sub-cell, its outermost vertices reach the
    sub-cell boundary exactly.

    Inner corner positions (the 4 diamond vertices in the central gap):

        NE inner: (q,  q)  +  r · (cos(θ+225°), sin(θ+225°))  ≈  (+ic, 0)
        NW inner: (-q, q)  +  r · (cos(θ+315°), sin(θ+315°))  ≈  (0, +ic)
        SW inner: (-q,-q)  +  r · (cos(θ+ 45°), sin(θ+ 45°))  ≈  (-ic, 0)
        SE inner: (q, -q)  +  r · (cos(θ+135°), sin(θ+135°))  ≈  (0, -ic)

    where q = cell_size/4, r = s·√2/2, ic = half-diamond half-width.

    Hinge bridge length (distance between adjacent inner corners):

        bridge_len = r · √(2 · (1 − cos(90°))) · ... (computed numerically)

    At baseline (θ=22.5°, cell_size=7.85, hinge_thickness=0.40):
        s           ≈ 3.00 mm  (reference: 3.01 mm ✓)
        inner gap   ≈ 1.63 mm  (diamond diagonal arm length)
        hinge ratio ≈ 0.25     (hinge_thickness / bridge_len; slender ✓)

TILING:
    Each outer vertex of each square sits exactly on the cell boundary line.
    In the tiled lattice, a neighboring cell's outer corner lands at the same
    boundary point, creating a shared hinge location between cells.  The
    hinge material at those boundary points is contributed by the 4 central
    hinge bridges of the neighboring cells after assembly.
"""

from __future__ import annotations

import math
from typing import Any

import cadquery as cq

from designs.base_cell import BaseUnitCell, SketchLike
from workflow.case_schema import DesignType, RotatingSquareParameters

# ---------------------------------------------------------------------------
# Module constants — baseline values from the architecture spec
# ---------------------------------------------------------------------------

_BASELINE_CELL_SIZE: float = 7.85         # mm
_BASELINE_ROTATION_ANGLE_DEG: float = 22.5  # degrees
_BASELINE_HINGE_THICKNESS: float = 0.40   # mm
_BASELINE_RIGID_SEGMENT_MM: float = 3.01  # mm — reference model observation


# ---------------------------------------------------------------------------
# Unit-cell class
# ---------------------------------------------------------------------------

class RotatingSquareUnitCell(BaseUnitCell):
    """
    Parametric 2D rotating-square unit cell.

    Generates a CadQuery Sketch profile consisting of four rotated rigid
    square plates connected by four central hinge bridges.  The profile is
    suitable for tiling into a 5×3 lattice and extrusion into a 3D solid.

    Args:
        parameters: ``RotatingSquareParameters`` dataclass instance from
                    ``workflow/case_schema.py``.
    """

    DESIGN_SLUG: str = DesignType.ROTATING_SQUARE.value

    def __init__(self, parameters: RotatingSquareParameters) -> None:
        if not isinstance(parameters, RotatingSquareParameters):
            raise TypeError(
                f"RotatingSquareUnitCell requires RotatingSquareParameters, "
                f"got {type(parameters).__name__}."
            )
        super().__init__(parameters)

    @property
    def params(self) -> RotatingSquareParameters:
        """
        Typed access to the rotating-square parameter set.

        ARCHITECTURAL DECISION — typed property alias:
            BaseUnitCell stores _parameters as DesignParameterSet (union).
            This property narrows the type to RotatingSquareParameters so
            that design-specific fields (params.hinge_thickness, etc.) are
            accessible without scattered isinstance checks or type: ignore
            annotations throughout the class.
        """
        return self._parameters  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_parameters(self) -> None:
        """
        Design-level geometry feasibility checks for the rotating-square cell.

        Supplements the schema-level field validation in
        ``RotatingSquareParameters.validate()``.

        Additional checks:
          - rotation_angle_deg > 5° (practical lower bound for version 1)
              Below 5°: inner corners almost at origin; bridge length < t_hinge.
          - rotation_angle_deg < 44° (safe upper bound; cos+sin peaks at 45°)
              At 45°: cos(45°)+sin(45°) = √2 = max; the formula is valid but
              the squares rotate to 45° (diamond orientation at sub-cell edges).
              This is geometrically valid but at the design level we keep a
              small margin.
          - hinge_thickness < cell_size / 3
              Generous upper bound; thin hinges are the design intent.
          - hinge_thickness < 0.6 × central_hinge_bridge_length
              Ensures the hinge bridge is slender (aspect ratio > ~1.7).
              A "hinge" wider than ~60% of its own length is not mechanically
              hinge-like.  Cases that violate this are geometrically unusual
              and should be flagged before meshing.

        Raises:
            ValueError: with a descriptive message on any violation.
        """
        p = self.params

        self._require_positive("cell_size", p.cell_size)
        self._require_positive("hinge_thickness", p.hinge_thickness)

        # Practical lower bound: inner corners must be meaningfully separated
        self._require_angle_range(
            "rotation_angle_deg", p.rotation_angle_deg, 5.0, 44.0
        )

        # Generous upper bound on hinge thickness
        max_ht = p.cell_size / 3.0
        if p.hinge_thickness >= max_ht:
            raise ValueError(
                f"hinge_thickness ({p.hinge_thickness:.4f} mm) must be less "
                f"than cell_size / 3 = {max_ht:.4f} mm for the rotating-square "
                f"cell to remain geometrically meaningful."
            )

        # Slender-hinge check: hinge should not be wider than 60% of its length
        bridge_len = self._central_hinge_bridge_length()
        if bridge_len <= 0.0:
            raise ValueError(
                f"Central hinge bridge length is zero or negative "
                f"(rotation_angle_deg={p.rotation_angle_deg:.2f}°).  "
                f"Increase rotation_angle_deg above 5°."
            )
        if p.hinge_thickness >= 0.6 * bridge_len:
            raise ValueError(
                f"hinge_thickness ({p.hinge_thickness:.4f} mm) is ≥ 60% of "
                f"the central hinge bridge length ({bridge_len:.4f} mm). "
                f"The connecting bridge would not be mechanically hinge-like. "
                f"Reduce hinge_thickness or increase rotation_angle_deg."
            )

    # ------------------------------------------------------------------
    # Derived geometry quantities
    # ------------------------------------------------------------------

    def _half_cell(self) -> float:
        """Half the nominal cell size [mm]: cell_size / 2."""
        return self.params.cell_size / 2.0

    def _quadrant_offset(self) -> float:
        """
        Distance from cell origin to the centre of each quadrant square [mm].

        The 4 squares are centred at (±q, ±q) where q = cell_size / 4.
        This ensures each square fits within its cell_size/2 × cell_size/2
        sub-cell when combined with the formula for ``_square_side_length``.
        """
        return self.params.cell_size / 4.0

    def _square_side_length(self) -> float:
        """
        Rigid square side length [mm].

        Derivation
        ----------
        For a square of side s rotated by θ and placed in a box of side b,
        the condition that all outer vertices touch the box boundary is::

            b = s · (cos θ + sin θ)

        With b = cell_size / 2 (each square occupies a quadrant sub-cell)::

            s = (cell_size / 2) / (cos θ + sin θ)

        Verification at baseline (θ=22.5°, cell_size=7.85):
            s = 3.925 / (0.9239 + 0.3827) = 3.925 / 1.3066 ≈ 3.00 mm
            Reference observed: 3.01 mm ✓

        Range
        -----
            θ → 0°:  cos+sin → 1, s → cell_size/2 (square fills sub-cell axis-aligned)
            θ = 45°: cos+sin = √2 (maximum), s = cell_size / (2√2) ≈ 0.354·cell_size
        """
        theta = math.radians(self.params.rotation_angle_deg)
        return self._half_cell() / (math.cos(theta) + math.sin(theta))

    def _vertex_radius(self) -> float:
        """
        Distance from the square centre to any vertex [mm]: s · √2 / 2.

        This is the circumradius of the rigid square.
        """
        return self._square_side_length() * math.sqrt(2.0) / 2.0

    def _inner_corner_positions(
        self,
    ) -> tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]:
        """
        Compute the 4 inner corner positions that form the central diamond.

        Each inner corner is the vertex of a quadrant square that points most
        directly toward the cell origin.  The 4 positions together form a
        diamond (rhombus) shape centred on the origin.

        Returns:
            Tuple (ne, nw, sw, se) where each element is (x, y) in mm.

        Inner corner formulae
        ---------------------
        For a square centred at (cx, cy) rotated by θ, the vertex at relative
        angular position α (measured from square centre) is::

            vertex = (cx + r·cos(α), cy + r·sin(α))

        The inner vertex (pointing toward cell origin) for each quadrant::

            NE at (+q, +q):  α = θ + 225°
            NW at (-q, +q):  α = θ + 315°
            SW at (-q, -q):  α = θ +  45°
            SE at (+q, -q):  α = θ + 135°
        """
        theta = math.radians(self.params.rotation_angle_deg)
        q = self._quadrant_offset()
        r = self._vertex_radius()

        def _vertex(cx: float, cy: float, extra_deg: float) -> tuple[float, float]:
            alpha = theta + math.radians(extra_deg)
            return (cx + r * math.cos(alpha), cy + r * math.sin(alpha))

        ne = _vertex( q,  q, 225.0)
        nw = _vertex(-q,  q, 315.0)
        sw = _vertex(-q, -q,  45.0)
        se = _vertex( q, -q, 135.0)

        return ne, nw, sw, se

    def _central_hinge_bridge_length(self) -> float:
        """
        Length [mm] of each central hinge bridge = distance between adjacent
        inner corners (e.g. NE and NW).

        ARCHITECTURAL DECISION — computed numerically from inner corner
        positions rather than analytically simplified:
            The closed-form simplification involves terms like sin(θ−45°) and
            is non-obvious.  Computing it numerically from the corner positions
            keeps the code self-consistent with the rest of the geometry and is
            easier to audit.
        """
        ne, nw, _, _ = self._inner_corner_positions()
        return math.sqrt((ne[0] - nw[0]) ** 2 + (ne[1] - nw[1]) ** 2)

    # ------------------------------------------------------------------
    # CadQuery location helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _loc(x: float, y: float, angle_deg: float = 0.0) -> cq.Location:
        """
        Build a ``cq.Location`` at (x, y) with a Z-axis rotation.

        Used with ``cq.Sketch.push([loc]).rect(w, h)`` to position and
        orient rectangular members and squares.
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
        Compute the centre ``Location`` and length for a rectangular member
        that spans point P1 = (x1, y1) to point P2 = (x2, y2).

        The returned ``Location`` sits at the midpoint and is rotated so that
        its local X axis aligns with the P1→P2 direction.

        Used to place hinge bridge rectangles::

            sketch.push([loc]).rect(length, hinge_thickness)

        Returns:
            Tuple (Location at midpoint, member length [mm]).
        """
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx * dx + dy * dy)
        angle_deg = math.degrees(math.atan2(dy, dx))
        return RotatingSquareUnitCell._loc(cx, cy, angle_deg), length

    # ------------------------------------------------------------------
    # 2D geometry build
    # ------------------------------------------------------------------

    def build_2d(self) -> SketchLike:
        """
        Build and return the 2D rotating-square profile in the XY plane.

        Construction strategy
        ----------------------
        Step 1 — Four rigid squares:
            Each square of side ``s`` is placed at its quadrant centre
            (±q, ±q) and rotated by ``rotation_angle_deg`` using a
            ``cq.Location`` with the Z-axis rotation baked in.

        Step 2 — Four central hinge bridges:
            Slender rectangles connecting adjacent inner corners (NE↔NW,
            NW↔SW, SW↔SE, SE↔NE).  Each rectangle has:
                length = distance between connected inner corners
                width  = hinge_thickness
            Together they form a diamond-shaped ring of material at the
            centre of the cell.

        Step 3 — Union:
            ``cq.Sketch.clean()`` merges all overlapping faces into a single
            connected planar region.

        ARCHITECTURAL DECISION — all squares rotate by the same +θ:
            Alternating square rotations (+θ, −θ) are seen in some
            literature variants but require asymmetric hinge bridge
            positioning that is harder to parameterise robustly.  Using a
            uniform rotation makes the geometry fully symmetric under 90°
            rotation and simpler to tile.  The auxetic property is preserved
            because the central hinge bridges enforce the constrained rotation
            of the mechanism.

        ARCHITECTURAL DECISION — no explicit outer hinge patches:
            In the tiled lattice, neighboring cells contribute outer corner
            material at the shared cell boundary positions.  Adding outer
            hinge patches here would create double material at those positions
            after tiling.  Outer boundary material is handled by
            lattice_builder.py.

        Returns:
            ``cq.Sketch`` with squares and hinge bridges unioned into a
            single 2D connected region.
        """
        t = self.params.hinge_thickness
        theta = self.params.rotation_angle_deg  # degrees; passed directly to _loc
        q = self._quadrant_offset()
        s = self._square_side_length()

        # Compute inner corner positions and hinge bridge geometry.
        ne, nw, sw, se = self._inner_corner_positions()

        ne_nw_loc, ne_nw_len = self._member_loc_len(ne[0], ne[1], nw[0], nw[1])
        nw_sw_loc, nw_sw_len = self._member_loc_len(nw[0], nw[1], sw[0], sw[1])
        sw_se_loc, sw_se_len = self._member_loc_len(sw[0], sw[1], se[0], se[1])
        se_ne_loc, se_ne_len = self._member_loc_len(se[0], se[1], ne[0], ne[1])

        sk = (
            cq.Sketch()

            # --- Four rigid squares, one per quadrant ---
            # Each uses a Location rotated by theta so the square body is
            # tilted by rotation_angle_deg from axis-alignment.
            .push([self._loc( q,  q, theta)]).rect(s, s)   # NE quadrant
            .push([self._loc(-q,  q, theta)]).rect(s, s)   # NW quadrant
            .push([self._loc(-q, -q, theta)]).rect(s, s)   # SW quadrant
            .push([self._loc( q, -q, theta)]).rect(s, s)   # SE quadrant

            # --- Four central hinge bridges ---
            # Each bridge connects two adjacent inner corners of the central
            # diamond.  Width = hinge_thickness; length = corner-to-corner distance.
            # Together they form a closed diamond-ring of hinge material
            # surrounding the central opening of the mechanism.
            .push([ne_nw_loc]).rect(ne_nw_len, t)   # NE ↔ NW bridge
            .push([nw_sw_loc]).rect(nw_sw_len, t)   # NW ↔ SW bridge
            .push([sw_se_loc]).rect(sw_se_len, t)   # SW ↔ SE bridge
            .push([se_ne_loc]).rect(se_ne_len, t)   # SE ↔ NE bridge

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
        comparison against the reference model observation (3.01 mm rigid
        segment) for the baseline parameter set.

        Returns:
            Plain JSON-serialisable dict.
        """
        ne, nw, sw, se = self._inner_corner_positions()
        bridge_len = self._central_hinge_bridge_length()
        s = self._square_side_length()
        ic_x = ne[0]   # inner corner x-offset from origin (approximately)
        ic_y = nw[1]   # inner corner y-offset from origin (approximately)

        return {
            "design": "rotating_square",
            "version": 1,

            "baseline_parameters": {
                "cell_size_mm":          _BASELINE_CELL_SIZE,
                "rotation_angle_deg":    _BASELINE_ROTATION_ANGLE_DEG,
                "hinge_thickness_mm":    _BASELINE_HINGE_THICKNESS,
            },

            "current_parameters": self.params.to_dict(),

            "derived_geometry": {
                "quadrant_offset_mm":          round(self._quadrant_offset(), 4),
                "rigid_square_side_mm":        round(s, 4),
                "vertex_circumradius_mm":      round(self._vertex_radius(), 4),
                "inner_corner_ne_mm":          [round(ne[0], 4), round(ne[1], 4)],
                "inner_corner_nw_mm":          [round(nw[0], 4), round(nw[1], 4)],
                "inner_corner_sw_mm":          [round(sw[0], 4), round(sw[1], 4)],
                "inner_corner_se_mm":          [round(se[0], 4), round(se[1], 4)],
                "central_hinge_bridge_len_mm": round(bridge_len, 4),
                "hinge_slenderness_ratio":     round(
                    self.params.hinge_thickness / bridge_len, 4
                ) if bridge_len > 0 else None,
            },

            "reference_proportions_at_baseline": {
                "rigid_segment_mm": _BASELINE_RIGID_SEGMENT_MM,
                "computed_rigid_segment_mm": round(s, 4),
                "match_note": (
                    f"Computed s = {s:.3f} mm vs reference {_BASELINE_RIGID_SEGMENT_MM} mm "
                    f"({abs(s - _BASELINE_RIGID_SEGMENT_MM) / _BASELINE_RIGID_SEGMENT_MM * 100:.1f}% "
                    f"difference; excellent agreement)."
                ),
            },

            "geometry_notes": [
                "All 4 rigid squares rotate by the same +rotation_angle_deg (uniform, not alternating).",
                "s = (cell_size/2) / (cos θ + sin θ): ensures outer corners touch sub-cell boundary.",
                "4 central hinge bridges form a diamond ring connecting the 4 inner corners.",
                "hinge_thickness is the bridge cross-sectional width, not the gap between squares.",
                "Outer corners of each square touch the cell boundary; tiling provides outer hinge material.",
                "No screw holes; no anatomical plate outline (version-1 raw auxetic envelope).",
            ],
        }
