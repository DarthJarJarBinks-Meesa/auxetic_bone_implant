"""
src/designs/base_cell.py
=========================
Abstract base class for all parametric 2D auxetic unit-cell generators.

This module defines the shared interface, common validation hooks, metadata
behaviour, and utility methods that every concrete unit-cell class must
implement and may reuse.

ARCHITECTURAL DECISION — base class scope is narrow:
    This class knows about exactly one thing: a single parametric 2D unit
    cell in the XY plane.  It does NOT know about:
      - 5×3 tiling or lattice assembly  (→ geometry/lattice_builder.py)
      - 3D extrusion                    (→ geometry/extruder.py)
      - meshing, materials, or solver   (→ simulation/)
      - fatigue proxy or reporting      (→ analysis/)
    Keeping the scope narrow makes it easy to add new designs without
    touching any downstream modules.

ARCHITECTURAL DECISION — ABC via abc.ABC rather than raise-NotImplementedError:
    Using @abc.abstractmethod gives Python's import machinery the ability to
    raise TypeError at class *instantiation* if a subclass forgets to
    implement a required method.  This catches errors earlier and produces
    a clearer message than a runtime AttributeError or a custom guard.

ARCHITECTURAL DECISION — DesignParameterSet accepted in constructor, not
individual kwargs:
    The constructor takes the typed parameter dataclass from case_schema.py
    rather than unpacked keyword arguments.  This keeps the parameter
    contract centralised in case_schema.py and means that any change to
    design parameters only requires updating the dataclass and its
    from_dict / to_dict — not every constructor signature.

ARCHITECTURAL DECISION — SketchLike union, not a single concrete type:
    CadQuery geometry can be meaningfully returned as either a cq.Workplane
    (the primary geometry carrier) or a cq.Sketch (the newer sketch API).
    Defining a union alias for both keeps the interface flexible as CadQuery
    evolves and lets subclasses choose the API that best suits their geometry
    construction strategy.  Downstream callers (lattice_builder, extruder)
    must handle both types.

Concrete subclasses reside in:
    src/designs/reentrant.py
    src/designs/rotating_square.py
    src/designs/tetrachiral.py
"""

from __future__ import annotations

import abc
from typing import Any, Union

import cadquery as cq

from workflow.case_schema import (
    DesignParameterSet,
    DesignType,
    ReentrantParameters,
    RotatingSquareParameters,
    TetrachiralParameters,
)

# ---------------------------------------------------------------------------
# Type alias for 2D geometry return values
# ---------------------------------------------------------------------------

# ARCHITECTURAL DECISION — SketchLike union covers both CadQuery geometry APIs:
#   cq.Workplane is the classic CadQuery surface/solid carrier; cq.Sketch is
#   the newer constrained-sketch API introduced in CadQuery 2.2+.  Either can
#   be passed to an extrude step.  Using a Union alias rather than committing
#   to one type lets subclasses choose whichever API produces cleaner code for
#   their specific geometry (e.g. rotating square with precise angular offsets
#   may suit Workplane more than Sketch).
SketchLike = Union[cq.Workplane, cq.Sketch]
"""
Type alias for the 2D geometry object returned by ``build_2d()``.

Either ``cq.Workplane`` or ``cq.Sketch`` is acceptable.  Downstream
callers (lattice_builder, extruder) must handle both variants.
"""


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseUnitCell(abc.ABC):
    """
    Abstract base class for all parametric 2D auxetic unit-cell generators.

    Represents one unit cell in the XY plane whose nominal square bounding
    envelope has side length ``cell_size``.  Concrete subclasses implement
    ``build_2d()``, ``validate_parameters()``, and ``reference_metadata()``.

    The unit cell is a pure 2D geometry object.  Tiling into a 5×3 lattice
    and extrusion into a 3D solid are handled by separate geometry modules
    (``lattice_builder.py`` and ``extruder.py``) and are outside this class.

    Usage pattern::

        cell = ReentrantUnitCell(params)
        cell.validate()                  # validate parameters before use
        sketch = cell.build_2d()         # produce 2D CadQuery geometry
        meta  = cell.to_metadata_dict()  # record design metadata for case log

    Args:
        parameters: Typed design parameter dataclass from ``case_schema.py``.
                    Must be one of ``ReentrantParameters``,
                    ``RotatingSquareParameters``, or ``TetrachiralParameters``.

    Raises:
        TypeError:  if ``parameters`` is not a supported design parameter type.
        ValueError: if parameter validation fails at construction time.
    """

    def __init__(self, parameters: DesignParameterSet) -> None:
        # ARCHITECTURAL DECISION — type-check at construction, not later:
        #   Catching an unsupported parameter type here (rather than inside
        #   build_2d) gives a clear error at the point of object creation,
        #   not deep inside geometry generation where the root cause is
        #   harder to trace.
        if not isinstance(
            parameters,
            (ReentrantParameters, RotatingSquareParameters, TetrachiralParameters),
        ):
            raise TypeError(
                f"BaseUnitCell requires a supported DesignParameterSet "
                f"(ReentrantParameters, RotatingSquareParameters, or "
                f"TetrachiralParameters).  Got: {type(parameters).__name__}."
            )
        self._parameters: DesignParameterSet = parameters

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def parameters(self) -> DesignParameterSet:
        """The typed design parameter dataclass for this unit cell."""
        return self._parameters

    @property
    def design_type(self) -> DesignType:
        """The ``DesignType`` enum member for this unit cell."""
        return self._parameters.design_type

    @property
    def cell_size(self) -> float:
        """
        The nominal square envelope side length [mm] of this unit cell.

        This is the ``cell_size`` field from the parameter dataclass and
        represents the repeating unit dimension for lattice tiling.
        """
        return self._parameters.cell_size

    # ------------------------------------------------------------------
    # Required abstract methods — subclasses MUST implement these
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def build_2d(self) -> SketchLike:
        """
        Build and return the 2D CadQuery geometry for one unit cell.

        The geometry must lie in the XY plane, centred on or anchored to
        the origin, and fit within the nominal ``cell_size × cell_size``
        square bounding box.

        Returns:
            A ``cq.Workplane`` or ``cq.Sketch`` representing the 2D closed
            wire(s) or face(s) of one unit cell, suitable for downstream
            extrusion.

        Raises:
            ValueError: if the current parameters produce degenerate or
                        self-intersecting geometry.
        """

    @abc.abstractmethod
    def validate_parameters(self) -> None:
        """
        Perform design-specific parameter validation beyond the dataclass
        schema checks.

        This is where subclasses encode geometric feasibility rules that
        cannot be expressed as simple field-level constraints (e.g. checking
        that derived feature sizes do not conflict with each other).

        Called automatically by ``validate()``.

        Raises:
            ValueError: with a descriptive message if any design-specific
                        constraint is violated.
        """

    @abc.abstractmethod
    def reference_metadata(self) -> dict[str, Any]:
        """
        Return a dictionary of baseline/reference information about this
        design and its geometric assumptions.

        This is design-specific context intended for case logs, reports, and
        debugging — not for geometry generation.  Subclasses should include
        relevant notes such as whether baseline proportions were used, known
        geometric constraints, and any version-1 simplifications.

        Returns:
            Plain JSON-serialisable dict.
        """

    # ------------------------------------------------------------------
    # Shared concrete methods
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """
        Run the full validation chain for this unit cell.

        Calls (in order):
          1. ``self._parameters.validate()`` — dataclass field-level checks
             defined in ``case_schema.py``.
          2. ``self.validate_parameters()`` — design-specific feasibility
             checks defined by the concrete subclass.

        ARCHITECTURAL DECISION — two-level validation chain:
            Dataclass-level validation (case_schema.py) covers individual
            field bounds (positive values, angle ranges, thickness < cell
            size).  Design-level validation here covers inter-parameter
            relationships and geometric feasibility that require knowledge
            of the specific topology (e.g. checking that hinge + rigid
            segment does not overflow the cell).  Separating the two levels
            keeps each check at the right layer of abstraction.

        Raises:
            ValueError: if any validation check fails.
        """
        # Level 1: dataclass schema validation
        self._parameters.validate()
        # Level 2: design-specific feasibility checks
        self.validate_parameters()

    def build_and_validate_2d(self) -> SketchLike:
        """
        Build the 2D geometry and perform lightweight post-build sanity checks.

        This is the recommended entry point for geometry modules that need
        a safe, validated 2D result.  It calls ``validate()`` first, then
        ``build_2d()``, then checks the result is non-None and is a
        recognised CadQuery type.

        ARCHITECTURAL DECISION — lightweight checks only:
            Deep geometric validity (self-intersection, manifoldness) is
            expensive and partially unreliable through the CadQuery API
            at this abstraction level.  Those checks belong in
            ``geometry/validators.py`` which has fuller context about the
            assembled lattice.  This method only guards against the most
            common trivial failures (None return, wrong type).

        Returns:
            Validated 2D CadQuery geometry.

        Raises:
            ValueError: if ``validate()`` fails or the built geometry is
                        None or an unrecognised type.
        """
        self.validate()
        result = self.build_2d()

        if result is None:
            raise ValueError(
                f"{self.__class__.__name__}.build_2d() returned None for "
                f"parameters: {self._parameters.to_dict()}.  "
                f"Check that the geometry construction logic produces a "
                f"valid CadQuery object."
            )
        if not isinstance(result, (cq.Workplane, cq.Sketch)):
            raise ValueError(
                f"{self.__class__.__name__}.build_2d() must return a "
                f"cq.Workplane or cq.Sketch, got: {type(result).__name__}."
            )

        return result

    def bounding_box_size(self) -> tuple[float, float]:
        """
        Return the nominal 2D bounding box of this unit cell as (width, height)
        in mm.

        ARCHITECTURAL DECISION — returns (cell_size, cell_size) in version 1:
            All version-1 designs use a square nominal envelope, so the
            bounding box is always (cell_size, cell_size).  Subclasses with
            non-square cells can override this method without affecting the
            base class or any other design.

        Returns:
            Tuple (width_mm, height_mm) of the nominal bounding box.
        """
        return (self.cell_size, self.cell_size)

    def to_metadata_dict(self) -> dict[str, Any]:
        """
        Return a compact, JSON-serialisable metadata dictionary for this
        unit cell instance.

        Combines:
          - design type string
          - serialised parameters (from ``parameters.to_dict()``)
          - design-specific reference metadata (from ``reference_metadata()``)

        Useful for writing case metadata files and for logging.

        Returns:
            Plain dict with string keys and JSON-serialisable values.
        """
        return {
            "design_type": self.design_type.value,
            "parameters": self._parameters.to_dict(),
            "bounding_box_mm": list(self.bounding_box_size()),
            "reference_metadata": self.reference_metadata(),
        }

    def parameter_signature(self) -> str:
        """
        Return a compact, human-readable string identifying this unit cell's
        design type and key parameter values.

        Useful for log messages, filenames, and debugging output.

        ARCHITECTURAL DECISION — delegates to case_schema signature logic:
            Rather than re-implementing parameter serialisation here, this
            method formats the design type plus a compact repr of the
            parameters dict, excluding derived values for conciseness.
            It is intentionally short — uniqueness is NOT guaranteed and
            collision-proof hashing is handled by ``workflow/hashing.py``.

        Returns:
            Short string, e.g.
            ``"reentrant(cs=7.85, wt=1.5, ra=70.0)"``
        """
        params = self._parameters.to_dict()
        # Remove non-independent or redundant keys for display brevity.
        params.pop("design_type", None)
        params.pop("fillet_radius_derived", None)

        # Abbreviate keys for compact display.
        abbrev = {
            "cell_size": "cs",
            "wall_thickness": "wt",
            "reentrant_angle_deg": "ra",
            "rotation_angle_deg": "rot",
            "hinge_thickness": "ht",
            "node_radius": "nr",
            "ligament_thickness": "lt",
        }
        parts = ", ".join(
            f"{abbrev.get(k, k)}={v}" for k, v in params.items()
        )
        return f"{self.design_type.value}({parts})"

    # ------------------------------------------------------------------
    # Protected validation helpers for subclass reuse
    # ------------------------------------------------------------------

    @staticmethod
    def _require_positive(name: str, value: float) -> None:
        """
        Assert that ``value`` is strictly positive.

        Args:
            name:  Human-readable parameter name for the error message.
            value: The numeric value to check.

        Raises:
            ValueError: if ``value`` <= 0.
        """
        if value <= 0.0:
            raise ValueError(
                f"Parameter '{name}' must be strictly positive, got {value}."
            )

    @staticmethod
    def _require_angle_range(
        name: str,
        value: float,
        min_deg: float,
        max_deg: float,
    ) -> None:
        """
        Assert that an angle ``value`` falls within the open interval
        (min_deg, max_deg).

        Args:
            name:    Human-readable parameter name for the error message.
            value:   The angle in degrees to check.
            min_deg: Exclusive lower bound.
            max_deg: Exclusive upper bound.

        Raises:
            ValueError: if ``value`` is outside (min_deg, max_deg).
        """
        if not (min_deg < value < max_deg):
            raise ValueError(
                f"Parameter '{name}' must be in the open interval "
                f"({min_deg}°, {max_deg}°), got {value}°."
            )

    @staticmethod
    def _require_less_than(name: str, value: float, upper: float) -> None:
        """
        Assert that ``value`` is strictly less than ``upper``.

        Useful for checking that a sub-feature (e.g. wall thickness) does
        not exceed the cell envelope.

        Args:
            name:  Human-readable parameter name for the error message.
            value: The numeric value to check.
            upper: The exclusive upper bound.

        Raises:
            ValueError: if ``value`` >= ``upper``.
        """
        if value >= upper:
            raise ValueError(
                f"Parameter '{name}' must be less than {upper}, got {value}."
            )

    @staticmethod
    def _require_greater_than(name: str, value: float, lower: float) -> None:
        """
        Assert that ``value`` is strictly greater than ``lower``.

        Args:
            name:  Human-readable parameter name for the error message.
            value: The numeric value to check.
            lower: The exclusive lower bound.

        Raises:
            ValueError: if ``value`` <= ``lower``.
        """
        if value <= lower:
            raise ValueError(
                f"Parameter '{name}' must be greater than {lower}, got {value}."
            )

    # ------------------------------------------------------------------
    # Geometry utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _new_workplane(
        plane: str = "XY",
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> cq.Workplane:
        """
        Return a fresh CadQuery Workplane anchored to the specified plane
        and origin.

        ARCHITECTURAL DECISION — XY plane as the canonical build plane:
            All unit-cell geometry is built in XY (Z = 0) per the global
            architecture assumption.  The ``plane`` parameter is accepted
            for flexibility but callers should not override it without a
            clear reason; downstream extrusion assumes XY input.

        Args:
            plane:  CadQuery plane name (default ``"XY"``).
            origin: Workplane origin in 3D space (default origin).

        Returns:
            Fresh ``cq.Workplane`` instance.
        """
        return cq.Workplane(plane, origin=origin)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"design={self.design_type.value!r}, "
            f"sig={self.parameter_signature()!r})"
        )
