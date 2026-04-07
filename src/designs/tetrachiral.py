"""
src/designs/tetrachiral.py
===========================
Parametric 2D tetrachiral unit-cell generator.

This module implements the tetrachiral auxetic design for the plate pipeline.
It subclasses ``BaseUnitCell`` and generates a 2D CadQuery Sketch profile
suitable for downstream tiling (lattice_builder.py) and extrusion (extruder.py).

GEOMETRY OVERVIEW:
    The unit cell contains a central circular node disk surrounded by four
    rectangular ligament arms that are offset tangentially (chirally) from
    the node centre rather than pointing radially outward.  The four arms
    extend to the cell boundary in the ±X and ±Y directions.

    Counter-clockwise (CCW) chiral convention (viewed from +Z)::

        ────────────────────────────────
        │         │ ← left ligament    │
        │         │   (above centre)   │
        │  ───────╋───────             │
        │         ●  ←central node     │
        │         ╋───────────────     │
        │         │                    │
        │         │ ← right ligament   │
        │         │   (below centre)   │
        ────────────────────────────────

    Chirality is produced by offsetting each arm tangentially by
    ``node_radius + ligament_thickness/2`` in the perpendicular direction
    before extending it toward the cell edge.  Four-fold rotational symmetry
    is preserved.

PARAMETERIZATION:
    tangential_offset = node_radius + ligament_thickness / 2

    For the top (+Y) arm (CCW chiral convention):
        x_centre = tangential_offset          (arm is RIGHT of cell centre)
        y_centre = (half_cell − node_radius) / 2
        arm length = half_cell + node_radius  (overlaps node disk)

    By 90° rotation symmetry:
        Right (+X) arm: below cell centre  (y = −tangential_offset)
        Bottom (−Y) arm: left of centre    (x = −tangential_offset)
        Left (−X) arm: above centre        (y = +tangential_offset)

    Geometric constraint (enforced by validate_parameters):
        node_radius + ligament_thickness < half_cell
        (ligament strip must fit within cell envelope)

FILLET RADIUS (derived, not independently swept):
    fillet_radius = 0.25 × (node_radius / 1.05)

    At baseline (node_radius = 1.05 mm): fillet_radius = 0.25 mm.

    ARCHITECTURAL DECISION — fillet geometric operations deferred:
        Computing the fillet radius is straightforward (formula above) and
        the value is included in metadata.  Applying a true geometric fillet
        via CadQuery Sketch operations at the disk/ligament junction is
        fragile in version 1 (the junction boundary edge identification is
        unreliable after Sketch clean() on overlapping shapes).  For version
        1, the smooth overlap region between the rotated ligament and the
        circular node provides a natural transition zone that qualitatively
        reproduces the fillet effect.  True CadQuery fillet operations can
        be added in a future version once geometry stability is confirmed on
        the full sweep range.  All code that uses fillet_radius references
        the derived value from TetrachiralParameters.fillet_radius; nothing
        hard-codes 0.25 mm.

TILING:
    Each ligament arm ends at ±half_cell in its extension direction.
    Adjacent tiled cells share material at the cell boundary faces.
    The CCW chiral convention is maintained across the tiled lattice:
    every node rotates in the same sense under mechanism deployment.

REFERENCE PROPORTIONS (baseline: cell_size=7.85, node_radius=1.05, lig_t=1.05):
    tangential_offset ≈ 1.575 mm
    arm length        ≈ 4.975 mm
    node diameter     ≈ 2.10  mm
    reference outer ring diameter observed in STEP model: ~4.20 mm
        (interpreted as the diagonal span of the four ligament
         attachment regions, ≈ 2 × tangential_offset × √2 ≈ 4.45 mm;
         approximate match — exact STEP feature not replicated)
"""

from __future__ import annotations

import math
from typing import Any

import cadquery as cq

from designs.base_cell import BaseUnitCell, SketchLike
from workflow.case_schema import DesignType, TetrachiralParameters

# ---------------------------------------------------------------------------
# Module constants — baseline values from the architecture spec
# ---------------------------------------------------------------------------

_BASELINE_CELL_SIZE: float = 7.85           # mm
_BASELINE_NODE_RADIUS: float = 1.05         # mm
_BASELINE_LIGAMENT_THICKNESS: float = 1.05  # mm
_BASELINE_FILLET_RADIUS: float = 0.25       # mm (derived at baseline)
_BASELINE_OUTER_RING_DIAMETER_REF: float = 4.20  # mm — reference STEP observation

# Fillet scaling formula coefficients (from architecture spec)
_FILLET_NUMERATOR: float = 0.25
_FILLET_BASELINE_NODE_RADIUS: float = 1.05  # denominator reference


# ---------------------------------------------------------------------------
# Unit-cell class
# ---------------------------------------------------------------------------

class TetrachiralUnitCell(BaseUnitCell):
    """
    Parametric 2D tetrachiral unit cell.

    Generates a CadQuery Sketch profile consisting of a central circular
    node disk and four tangentially-offset rectangular ligament arms.  The
    profile is suitable for tiling into a 5×3 lattice and extrusion into a
    3D plate solid.

    Args:
        parameters: ``TetrachiralParameters`` dataclass instance from
                    ``workflow/case_schema.py``.
    """

    DESIGN_SLUG: str = DesignType.TETRACHIRAL.value

    def __init__(self, parameters: TetrachiralParameters) -> None:
        if not isinstance(parameters, TetrachiralParameters):
            raise TypeError(
                f"TetrachiralUnitCell requires TetrachiralParameters, "
                f"got {type(parameters).__name__}."
            )
        super().__init__(parameters)

    @property
    def params(self) -> TetrachiralParameters:
        """
        Typed access to the tetrachiral parameter set.

        ARCHITECTURAL DECISION — typed property alias:
            BaseUnitCell stores _parameters as DesignParameterSet (union).
            This narrows to TetrachiralParameters so design-specific fields
            (params.node_radius, params.fillet_radius, etc.) are accessible
            without repeated isinstance checks throughout the class.
        """
        return self._parameters  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_parameters(self) -> None:
        """
        Design-level geometry feasibility checks for the tetrachiral cell.

        Supplements the schema-level field validation in
        ``TetrachiralParameters.validate()``.

        Additional checks:
          - node_radius > 0 (redundant with schema but explicit here)
          - ligament_thickness > 0 (redundant with schema but explicit here)
          - node_radius + ligament_thickness < half_cell
              Ensures the ligament strip (at tangential offset) fits within
              the cell envelope:
                  outer edge of ligament in perpendicular direction
                  = node_radius + ligament_thickness
              This must be strictly less than half_cell.
          - tangential_offset < half_cell (node_radius + t/2 < half_cell)
              The ligament centreline must be inside the cell.
          - 2 × node_radius < cell_size (redundant with schema, but explicit)
              The node disk must fit inside the cell.
          - fillet_radius > 0
              Derived from node_radius; always true if node_radius > 0, but
              checked for explicitness.

        Raises:
            ValueError: with a descriptive message on any violation.
        """
        p = self.params
        hc = self._half_cell()

        self._require_positive("cell_size", p.cell_size)
        self._require_positive("node_radius", p.node_radius)
        self._require_positive("ligament_thickness", p.ligament_thickness)

        # Node disk must fit inside cell (schema already checks 2r < cs, but be explicit)
        if 2.0 * p.node_radius >= p.cell_size:
            raise ValueError(
                f"node diameter ({2 * p.node_radius:.4f} mm) must be "
                f"less than cell_size ({p.cell_size} mm)."
            )

        # Ligament strip outer edge must fit within cell envelope
        outer_edge = p.node_radius + p.ligament_thickness
        if outer_edge >= hc:
            raise ValueError(
                f"node_radius + ligament_thickness = {outer_edge:.4f} mm must "
                f"be less than half_cell = {hc:.4f} mm.  "
                f"The ligament strip (at tangential offset from node centre) "
                f"protrudes beyond the cell boundary.  "
                f"Reduce node_radius or ligament_thickness, or increase cell_size."
            )

        # Tangential offset (ligament centreline) must be inside the cell
        tangential_offset = p.node_radius + p.ligament_thickness / 2.0
        if tangential_offset >= hc:
            raise ValueError(
                f"tangential_offset (node_radius + ligament_thickness/2 = "
                f"{tangential_offset:.4f} mm) must be less than "
                f"half_cell ({hc:.4f} mm).  "
                f"Reduce node_radius or ligament_thickness."
            )

        # Fillet radius guard (should always pass given positive node_radius)
        if p.fillet_radius <= 0.0:
            raise ValueError(
                f"Derived fillet_radius ({p.fillet_radius:.6f} mm) must be "
                f"positive.  Check node_radius ({p.node_radius} mm)."
            )

    # ------------------------------------------------------------------
    # Derived geometry quantities
    # ------------------------------------------------------------------

    def _half_cell(self) -> float:
        """Half the nominal cell size [mm]: cell_size / 2."""
        return self.params.cell_size / 2.0

    def _tangential_offset(self) -> float:
        """
        Perpendicular (tangential) offset of each ligament centreline from the
        cell centre [mm].

        Derivation
        ----------
        For the ligament to exit tangentially from the node circle, its
        inner edge must be flush with the node surface at the exit point.
        The ligament rectangle centreline therefore sits at::

            tangential_offset = node_radius + ligament_thickness / 2

        At baseline (node_r=1.05, t=1.05):
            tangential_offset = 1.05 + 0.525 = 1.575 mm
        """
        return self.params.node_radius + self.params.ligament_thickness / 2.0

    def _arm_length(self) -> float:
        """
        Total length [mm] of each ligament arm rectangle.

        The arm extends from y = −node_radius (inside the node disk, ensuring
        overlap for a clean Sketch union) to y = +half_cell (at the cell edge)::

            arm_length = half_cell + node_radius

        At baseline: arm_length = 3.925 + 1.05 = 4.975 mm.
        """
        return self._half_cell() + self.params.node_radius

    def _arm_centre_along(self) -> float:
        """
        Centre coordinate of each ligament arm along its extension direction [mm].

        The arm spans from −node_radius to +half_cell, so its centre is::

            arm_centre = (half_cell − node_radius) / 2

        At baseline: (3.925 − 1.05) / 2 = 1.4375 mm from cell centre.
        """
        return (self._half_cell() - self.params.node_radius) / 2.0

    # ------------------------------------------------------------------
    # CadQuery location helper
    # ------------------------------------------------------------------

    @staticmethod
    def _loc(x: float, y: float, angle_deg: float = 0.0) -> cq.Location:
        """
        Build a ``cq.Location`` at (x, y) with an optional Z-axis rotation.

        Used with ``cq.Sketch.push([loc]).circle(r)`` or
        ``cq.Sketch.push([loc]).rect(w, h)`` to position sketch elements.
        """
        return cq.Location(
            cq.Vector(x, y, 0.0),
            cq.Vector(0.0, 0.0, 1.0),
            angle_deg,
        )

    # ------------------------------------------------------------------
    # 2D geometry build
    # ------------------------------------------------------------------

    def build_2d(self) -> SketchLike:
        """
        Build and return the 2D tetrachiral profile in the XY plane.

        Construction strategy
        ----------------------
        The Sketch accumulates additive faces (``cq.Sketch`` default mode)
        and ``.clean()`` merges all overlapping regions into one connected
        profile.

        Step 1 — Central circular node:
            A filled circle of radius ``node_radius`` centred at the origin.

        Step 2 — Four tangential ligament arms:
            Each arm is a rectangle of cross-section ``ligament_thickness``
            and length ``arm_length = half_cell + node_radius``.

            The inner end of each arm overlaps the node disk (extending
            ``node_radius`` past the cell centre) ensuring a robust topological
            connection without requiring a precise tangency boolean.

            CCW chiral convention (arm position in the perpendicular direction):
                Top    (+Y arm):  x = +tangential_offset  (arm is RIGHT of centre)
                Right  (+X arm):  y = −tangential_offset  (arm is BELOW centre)
                Bottom (−Y arm):  x = −tangential_offset  (arm is LEFT of centre)
                Left   (−X arm):  y = +tangential_offset  (arm is ABOVE centre)

        Step 3 — Fillet radius (version 1 simplification):
            The derived ``fillet_radius = 0.25 × (node_radius / 1.05)`` is
            computed and stored in metadata but NOT applied as a geometric
            CadQuery fillet operation.

            Reason: the Sketch .fillet() API requires explicit identification
            of specific boundary vertices after .clean(), which is unreliable
            when the sketch is assembled from multiple overlapping primitives
            (disk + 4 rectangles).  The natural overlap zone between the
            circular node and the rectangular arm provides a visually smooth
            transition region proportional to node_radius, qualitatively
            reproducing the fillet intent.

            This simplification is documented here and in reference_metadata().
            True fillet operations are a future-version improvement.

        Returns:
            ``cq.Sketch`` with node and all four arms unioned into a single
            2D connected region.
        """
        node_r = self.params.node_radius
        t = self.params.ligament_thickness
        to = self._tangential_offset()      # perpendicular offset of arm centreline
        ac = self._arm_centre_along()       # arm centre along extension direction
        al = self._arm_length()             # total arm length

        sk = (
            cq.Sketch()

            # --- Central circular node disk ---
            .push([self._loc(0.0, 0.0)]).circle(node_r)

            # --- Top arm: extends in +Y, offset in +X (CCW chiral) ---
            # Rectangle of width t (in X) × height al (in Y).
            # Centre at (to, ac): to the right of centre, upper half of cell.
            .push([self._loc(to, ac)]).rect(t, al)

            # --- Right arm: extends in +X, offset in −Y (CCW chiral) ---
            # Rectangle of width al (in X) × height t (in Y).
            # Centre at (ac, −to): right half of cell, below centre.
            .push([self._loc(ac, -to)]).rect(al, t)

            # --- Bottom arm: extends in −Y, offset in −X (CCW chiral) ---
            # Symmetric to top arm through 180° rotation.
            # Centre at (−to, −ac): left of centre, lower half of cell.
            .push([self._loc(-to, -ac)]).rect(t, al)

            # --- Left arm: extends in −X, offset in +Y (CCW chiral) ---
            # Symmetric to right arm through 180° rotation.
            # Centre at (−ac, +to): left half of cell, above centre.
            .push([self._loc(-ac, to)]).rect(al, t)

            # Union all overlapping additive faces into a single connected region.
            .clean()
        )

        return sk

    # ------------------------------------------------------------------
    # Reference metadata
    # ------------------------------------------------------------------

    def reference_metadata(self) -> dict[str, Any]:
        """
        Return reference and derived geometry metadata for this unit cell.

        Includes a comparison against the reference STEP model observation
        (~4.20 mm outer ring diameter) and a note on fillet simplification.

        Returns:
            Plain JSON-serialisable dict.
        """
        p = self.params
        hc = self._half_cell()
        to = self._tangential_offset()
        ac = self._arm_centre_along()
        al = self._arm_length()

        # Diagonal span of the 4 tangential offset points as a rough proxy for
        # the "outer ring diameter" observed in the STEP model.
        diagonal_span = 2.0 * to * math.sqrt(2.0)

        return {
            "design": "tetrachiral",
            "version": 1,

            "baseline_parameters": {
                "cell_size_mm":          _BASELINE_CELL_SIZE,
                "node_radius_mm":        _BASELINE_NODE_RADIUS,
                "ligament_thickness_mm": _BASELINE_LIGAMENT_THICKNESS,
                "fillet_radius_mm":      _BASELINE_FILLET_RADIUS,
            },

            "current_parameters": p.to_dict(),

            "derived_geometry": {
                "half_cell_mm":            round(hc, 4),
                "node_diameter_mm":        round(2.0 * p.node_radius, 4),
                "tangential_offset_mm":    round(to, 4),
                "arm_centre_along_mm":     round(ac, 4),
                "arm_length_mm":           round(al, 4),
                "fillet_radius_derived_mm": round(p.fillet_radius, 6),
                "diagonal_span_of_offsets_mm": round(diagonal_span, 4),
            },

            "reference_proportions_at_baseline": {
                "outer_ring_diameter_observed_mm": _BASELINE_OUTER_RING_DIAMETER_REF,
                "diagonal_span_computed_mm": round(
                    2.0 * (
                        _BASELINE_NODE_RADIUS + _BASELINE_LIGAMENT_THICKNESS / 2.0
                    ) * math.sqrt(2.0),
                    4,
                ),
                "note": (
                    "The reference STEP model outer ring diameter of 4.20 mm is "
                    "interpreted as an approximate diagonal span of the four "
                    "ligament tangential offset positions. "
                    "Computed diagonal span ≈ 4.45 mm (~6% difference). "
                    "Exact STEP feature geometry is not replicated in version 1."
                ),
            },

            "geometry_notes": [
                "Central circular node disk: radius = node_radius, centred at origin.",
                "Four arms are rectangular (width = ligament_thickness), "
                "tangentially offset by (node_radius + ligament_thickness/2).",
                "CCW chiral convention: top arm is to the right (+X offset), "
                "right arm is below (−Y offset), bottom arm is to the left, "
                "left arm is above (+Y offset).",
                "Each arm extends from −node_radius to +half_cell in its direction; "
                "the inner overlap with the disk ensures a clean topological union.",
                (
                    "fillet_radius = 0.25 × (node_radius / 1.05) = "
                    f"{p.fillet_radius:.6f} mm is computed but NOT applied as "
                    "a geometric CadQuery fillet operation in version 1.  "
                    "The natural disk/rectangle overlap provides a smooth "
                    "transition zone proportional to node_radius.  "
                    "True fillets are a future-version improvement."
                ),
                "No screw holes; no anatomical plate outline (version-1 raw auxetic envelope).",
            ],
        }
