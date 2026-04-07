"""
src/workflow/case_schema.py
============================
Canonical typed datamodels for the auxetic plate pipeline.

This file is the single source of truth for case and result schemas passed
between geometry, simulation, analysis, reporting, and workflow modules.

ARCHITECTURAL DECISION — single schema file:
    All datamodels live here rather than being split across modules so that
    every module imports from one place.  This prevents circular imports,
    keeps the shared contract visible, and makes it easy to grep for any
    field name.  If this file grows very large in a future version it can be
    split into a `case_schema/` sub-package, but the public import path
    (`from workflow.case_schema import ...`) should remain stable.

Units convention (used throughout):
  - Lengths / displacements : mm
  - Forces                  : N
  - Stresses / moduli       : MPa
  - Density                 : g/cm³
  - Angles                  : degrees

Version 1 supports three designs only:
  - Re-entrant honeycomb
  - Rotating square
  - Tetrachiral

Do NOT add screw-hole fields, anatomical plate outlines, or density as a
geometry parameter here.  Those are explicitly out of scope for version 1.
"""

from __future__ import annotations

# ARCHITECTURAL DECISION — math imported but not used directly in this file:
#   It is kept here as an explicit reminder that derived geometry formulae
#   (e.g. fillet_radius) stay as lightweight Python expressions rather than
#   being imported from the geometry module, which would create a circular
#   dependency (geometry → case_schema → geometry).
import math  # noqa: F401  (available for formula documentation; used by callers)

from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

# ARCHITECTURAL DECISION — all enums inherit from (str, Enum):
#   This means every enum member IS a plain string at runtime, so:
#     - YAML/JSON serialisation requires no custom encoder (value is already str)
#     - Comparisons against raw config strings work without .value lookup
#     - Readability in log output is clean ("completed" not "<CaseStatus.COMPLETED: ...>")
#   The trade-off is that `isinstance(x, str)` returns True for enum members,
#   which can mask type errors in loose code.  We accept that trade-off for
#   serialisation simplicity at this project scale.

@unique
class DesignType(str, Enum):
    """
    Supported auxetic unit-cell design types for version 1.

    EXTENSIBILITY NOTE: To add a new design in a future version (e.g.
    HALF_CIRCLE, TRI_ANTICHIRAL), add a member here and implement the
    corresponding parameter dataclass and geometry module.  No other changes
    to this file are required — the dispatch table in
    ``design_parameters_from_dict`` picks it up automatically once the
    from_dict entry is added there too.
    """

    REENTRANT = "reentrant"
    ROTATING_SQUARE = "rotating_square"
    TETRACHIRAL = "tetrachiral"


@unique
class LoadCaseType(str, Enum):
    """
    Supported mechanical load case categories.

    EXTENSIBILITY NOTE: 4-point bending and combined loading are deferred to
    a future version.  Add members here when those load cases are implemented
    in ``src/simulation/loadcases.py``.
    """

    AXIAL_COMPRESSION = "axial_compression"
    AXIAL_TENSION = "axial_tension"
    BENDING = "bending"

    # ARCHITECTURAL DECISION — named CYCLIC rather than FATIGUE:
    #   Version 1 only computes a fatigue-risk *proxy*, not a validated fatigue
    #   life.  Using "cyclic" accurately describes what is applied (a cyclic
    #   load pattern) without implying a validated fatigue outcome.  If a full
    #   fatigue solver is integrated in a future version, a FATIGUE member can
    #   be added without breaking existing CYCLIC cases.
    CYCLIC = "cyclic"


@unique
class CaseStatus(str, Enum):
    """
    Lifecycle status of a single pipeline case.

    Transitions follow a linear happy path:
        PENDING → RUNNING → COMPLETED
    with two terminal states:
        RUNNING → FAILED
        PENDING → SKIPPED  (e.g. if the result already exists in cache)

    ARCHITECTURAL DECISION — status lives on CaseResult, not CaseDefinition:
        CaseDefinition is immutable input; CaseResult is mutable output.
        Keeping status on the result means a definition can be re-queued
        without mutation, and definitions can be safely hashed/cached.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@unique
class SolverStatus(str, Enum):
    """
    Status of the external solver (CalculiX) invocation.

    Kept separate from CaseStatus so that solver-specific diagnostics
    (e.g. TIMEOUT vs generic FAILED) do not pollute the top-level case
    lifecycle enum.
    """

    NOT_RUN = "not_run"
    SUCCESS = "success"
    FAILED = "failed"

    # ARCHITECTURAL DECISION — TIMEOUT as a distinct state (not a sub-variant
    # of FAILED):  This makes it easy to filter timed-out cases separately in
    # reporting and to set per-solver timeout budgets in config without
    # changing result interpretation logic.
    TIMEOUT = "timeout"


@unique
class MeshStatus(str, Enum):
    """
    Status of the gmsh meshing step.

    DEGENERATE covers cases where meshing technically succeeded but produced
    a mesh that is unusable (e.g. inverted elements, zero-volume regions).
    This is distinct from FAILED (gmsh threw an exception or returned a
    non-zero error code).
    """

    NOT_RUN = "not_run"
    SUCCESS = "success"
    FAILED = "failed"
    DEGENERATE = "degenerate"


# ---------------------------------------------------------------------------
# Base design parameter dataclass
# ---------------------------------------------------------------------------

# ARCHITECTURAL DECISION — dataclasses, not Pydantic:
#   Pydantic would add automatic validation-on-construction and a richer
#   serialisation API, but it is an external dependency.  The architecture
#   spec requires standard-library only.  We replicate the key behaviour
#   (validate(), to_dict(), from_dict()) manually.  If Pydantic is approved
#   in a future version, the interface here maps cleanly onto BaseModel
#   without changing callers.

@dataclass
class BaseDesignParameters:
    """
    Shared base for all parametric unit-cell parameter sets.

    Not meant to be instantiated directly; use a concrete subclass.
    Subclasses MUST override ``validate()`` and call ``super().validate()``
    so that base-level checks always run.

    ARCHITECTURAL DECISION — abstract-ish base via convention, not ABC:
        Using ``abc.ABC`` would prevent accidental instantiation but adds
        minor complexity.  Since this project uses typed dispatch everywhere
        (``DesignParameterSet`` union, ``design_parameters_from_dict``),
        accidental direct instantiation of the base is caught early at the
        dispatch layer.  ABCs can be added in a future refactor if the class
        hierarchy grows significantly.
    """

    design_type: DesignType
    cell_size: float  # mm — nominal square envelope of one unit cell

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """
        Validate shared base parameters.

        Raises:
            ValueError: if any parameter is out of range.
        """
        if self.cell_size <= 0.0:
            raise ValueError(
                f"cell_size must be positive, got {self.cell_size}"
            )
        if not isinstance(self.design_type, DesignType):
            raise ValueError(
                f"design_type must be a DesignType enum member, "
                f"got {self.design_type!r}"
            )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Return a JSON-serialisable dictionary of all parameters.

        ARCHITECTURAL DECISION — enum serialised as .value (str), not the
        enum object itself:  This keeps ``to_dict()`` output directly writable
        to JSON/YAML without a custom encoder.  ``from_dict()`` reconstructs
        the enum via ``DesignType(value)``.

        Subclasses should call ``super().to_dict()`` and ``.update()`` the
        result with their own fields, rather than rebuilding from scratch.
        """
        return {
            "design_type": self.design_type.value,
            "cell_size": self.cell_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseDesignParameters":
        """
        Deserialise from a plain dictionary.

        Concrete subclasses override this; the base raises NotImplementedError
        to enforce that callers always use a concrete subclass or the
        ``design_parameters_from_dict`` helper function.
        """
        raise NotImplementedError(
            "from_dict() must be implemented by concrete subclasses."
        )


# ---------------------------------------------------------------------------
# Re-entrant honeycomb parameters
# ---------------------------------------------------------------------------

@dataclass
class ReentrantParameters(BaseDesignParameters):
    """
    Parametric design parameters for the re-entrant honeycomb unit cell.

    Reference baseline (from architecture spec):
        cell_size            = 7.85 mm
        wall_thickness       = 1.50 mm
        reentrant_angle_deg  = 70.0 °

    The wall_thickness is the full effective wall thickness.
    Derived geometric details (rib lengths, internal angles) are computed
    by the geometry module from these three independent parameters.
    """

    # ARCHITECTURAL DECISION — design_type locked via field(init=False):
    #   Setting ``init=False`` with a fixed default means the field is set
    #   automatically at construction and cannot be passed as a constructor
    #   argument.  This prevents accidentally constructing a
    #   ReentrantParameters with design_type=TETRACHIRAL, which would
    #   silently corrupt all dispatch logic downstream.
    #   The same pattern is used on all three concrete parameter classes.
    design_type: DesignType = field(default=DesignType.REENTRANT, init=False)

    cell_size: float = 7.85            # mm — reference baseline default
    wall_thickness: float = 1.5        # mm — full effective wall thickness
    reentrant_angle_deg: float = 70.0  # degrees

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Validate all re-entrant honeycomb parameters."""
        super().validate()

        if self.wall_thickness <= 0.0:
            raise ValueError(
                f"wall_thickness must be positive, got {self.wall_thickness}"
            )
        if self.wall_thickness >= self.cell_size:
            # A wall thicker than the cell envelope is geometrically impossible.
            raise ValueError(
                f"wall_thickness ({self.wall_thickness}) must be less than "
                f"cell_size ({self.cell_size})"
            )
        if not (0.0 < self.reentrant_angle_deg < 90.0):
            # At 0° the re-entrant feature collapses; at 90° it becomes a
            # straight-walled (non-auxetic) honeycomb.
            raise ValueError(
                f"reentrant_angle_deg must be in (0, 90) degrees, "
                f"got {self.reentrant_angle_deg}"
            )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "wall_thickness": self.wall_thickness,
                "reentrant_angle_deg": self.reentrant_angle_deg,
            }
        )
        return base

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReentrantParameters":
        """
        Deserialise from a plain dictionary (e.g. loaded from YAML).

        ARCHITECTURAL DECISION — explicit float() casts on all numeric fields:
            YAML loaders may return ints where floats are expected (e.g.
            ``cell_size: 7`` is parsed as int 7, not float 7.0).  Explicit
            casting here prevents downstream type errors without requiring
            callers to pre-coerce their data.  The same pattern is used in
            all ``from_dict`` implementations in this file.
        """
        return cls(
            cell_size=float(data["cell_size"]),
            wall_thickness=float(data["wall_thickness"]),
            reentrant_angle_deg=float(data["reentrant_angle_deg"]),
        )


# ---------------------------------------------------------------------------
# Rotating square parameters
# ---------------------------------------------------------------------------

@dataclass
class RotatingSquareParameters(BaseDesignParameters):
    """
    Parametric design parameters for the rotating-square unit cell.

    Reference baseline (from architecture spec):
        cell_size           = 7.85 mm
        rotation_angle_deg  = 22.5 °
        hinge_thickness     = 0.40 mm

    hinge_thickness is a true independent geometry parameter, not derived.
    Rigid segment lengths and internal offsets are computed by the geometry
    module from these three independent parameters.
    """

    # See design_type locking decision note in ReentrantParameters.
    design_type: DesignType = field(
        default=DesignType.ROTATING_SQUARE, init=False
    )

    cell_size: float = 7.85           # mm — reference baseline default
    rotation_angle_deg: float = 22.5  # degrees — square rotation angle
    hinge_thickness: float = 0.40     # mm — hinge connector thickness

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Validate all rotating-square parameters."""
        super().validate()

        if self.hinge_thickness <= 0.0:
            raise ValueError(
                f"hinge_thickness must be positive, got {self.hinge_thickness}"
            )
        if self.hinge_thickness >= self.cell_size:
            raise ValueError(
                f"hinge_thickness ({self.hinge_thickness}) must be less than "
                f"cell_size ({self.cell_size})"
            )
        if not (0.0 < self.rotation_angle_deg < 45.0):
            # The rotating-square topology requires rotation in (0°, 45°).
            # At 0° there is no rotation (no auxetic effect).
            # At 45° adjacent squares touch, eliminating the hinge gap entirely.
            raise ValueError(
                f"rotation_angle_deg must be in (0, 45) degrees for a "
                f"rotating-square topology, got {self.rotation_angle_deg}"
            )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "rotation_angle_deg": self.rotation_angle_deg,
                "hinge_thickness": self.hinge_thickness,
            }
        )
        return base

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RotatingSquareParameters":
        """Deserialise from a plain dictionary.

        See float() casting note in ReentrantParameters.from_dict.
        """
        return cls(
            cell_size=float(data["cell_size"]),
            rotation_angle_deg=float(data["rotation_angle_deg"]),
            hinge_thickness=float(data["hinge_thickness"]),
        )


# ---------------------------------------------------------------------------
# Tetrachiral parameters
# ---------------------------------------------------------------------------

@dataclass
class TetrachiralParameters(BaseDesignParameters):
    """
    Parametric design parameters for the tetrachiral unit cell.

    Reference baseline (from architecture spec):
        cell_size           = 7.85 mm
        node_radius         = 1.05 mm
        ligament_thickness  = 1.05 mm
        fillet_radius       = 0.25 mm  (when node_radius = 1.05 mm)

    fillet_radius is a DERIVED helper parameter, not independently set:
        fillet_radius = 0.25 * (node_radius / 1.05)

    It is exposed as a read-only @property and included in serialisation
    under the key "fillet_radius_derived" for traceability.
    """

    # See design_type locking decision note in ReentrantParameters.
    design_type: DesignType = field(
        default=DesignType.TETRACHIRAL, init=False
    )

    cell_size: float = 7.85           # mm — reference baseline default
    node_radius: float = 1.05         # mm — radius of the chiral node circle
    ligament_thickness: float = 1.05  # mm — thickness of connecting ligaments

    # ARCHITECTURAL DECISION — fillet_radius as @property, not a stored field:
    #   fillet_radius is strictly derived from node_radius via the formula
    #   mandated by the architecture spec.  Storing it as a dataclass field
    #   would risk it drifting out of sync if node_radius is updated after
    #   construction (e.g. in a sweep loop that mutates parameters).
    #   As a property it is always recomputed and always consistent.
    #   It appears in to_dict() as "fillet_radius_derived" (a traceability
    #   audit entry) but is explicitly ignored by from_dict() so that
    #   round-tripping serialisation never overrides the formula.

    @property
    def fillet_radius(self) -> float:
        """
        Derived fillet radius [mm], proportional to node_radius.

        Formula (from architecture spec):
            fillet_radius = 0.25 * (node_radius / 1.05)

        This value is recomputed on every access and cannot be set directly.
        """
        return 0.25 * (self.node_radius / 1.05)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Validate all tetrachiral parameters."""
        super().validate()

        if self.node_radius <= 0.0:
            raise ValueError(
                f"node_radius must be positive, got {self.node_radius}"
            )
        if self.ligament_thickness <= 0.0:
            raise ValueError(
                f"ligament_thickness must be positive, "
                f"got {self.ligament_thickness}"
            )
        # Node diameter must not exceed cell envelope — a generous upper bound.
        # The geometry module will apply tighter constraints during build.
        if 2.0 * self.node_radius >= self.cell_size:
            raise ValueError(
                f"node diameter ({2 * self.node_radius:.4f} mm) must be "
                f"less than cell_size ({self.cell_size} mm)"
            )
        if self.ligament_thickness >= self.cell_size:
            raise ValueError(
                f"ligament_thickness ({self.ligament_thickness}) must be "
                f"less than cell_size ({self.cell_size})"
            )
        # Guard against a degenerate fillet; should not occur with positive
        # node_radius, but check explicitly for defensive clarity.
        if self.fillet_radius <= 0.0:
            raise ValueError(
                f"Derived fillet_radius must be positive; "
                f"check node_radius ({self.node_radius})"
            )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "node_radius": self.node_radius,
                "ligament_thickness": self.ligament_thickness,
                # ARCHITECTURAL DECISION — include derived value in output:
                #   "fillet_radius_derived" is written to to_dict() for
                #   traceability (reports, logs, audit trails).  The "_derived"
                #   suffix signals to readers that editing this field in saved
                #   YAML/JSON has no effect — only node_radius controls it.
                #   from_dict() explicitly ignores this key.
                "fillet_radius_derived": self.fillet_radius,
            }
        )
        return base

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TetrachiralParameters":
        """
        Deserialise from a plain dictionary.

        ``fillet_radius_derived`` in the input dict is intentionally ignored
        because fillet_radius is always recomputed from node_radius.
        See the fillet_radius property decision note above.
        """
        return cls(
            cell_size=float(data["cell_size"]),
            node_radius=float(data["node_radius"]),
            ligament_thickness=float(data["ligament_thickness"]),
        )


# ---------------------------------------------------------------------------
# Type alias for the union of all supported parameter sets
# ---------------------------------------------------------------------------

# ARCHITECTURAL DECISION — explicit union type alias, not BaseDesignParameters:
#   Using BaseDesignParameters as the type everywhere would lose concrete type
#   information and break static-analysis narrowing (e.g. mypy cannot infer
#   that p.fillet_radius is accessible without knowing p is
#   TetrachiralParameters).  The union alias preserves narrowing after
#   isinstance() checks.  When a new design is added, only this alias and the
#   dispatch table in design_parameters_from_dict need updating.
DesignParameterSet = (
    ReentrantParameters | RotatingSquareParameters | TetrachiralParameters
)
"""
Union of all concrete design parameter types.

Use this alias in function signatures and CaseDefinition instead of
repeating the full union every time.  When a new design is added in a
future version, extend this alias alongside DesignType and the dispatch
table in ``design_parameters_from_dict``.
"""


# ---------------------------------------------------------------------------
# Material definition
# ---------------------------------------------------------------------------

@dataclass
class MaterialDefinition:
    """
    Material property record used for simulation assignment.

    Version 1 supports two materials:
        - Ti-5Al-2.5Fe
        - Ti-6Al-4V

    Properties are stored in SI-adjacent engineering units:
        elastic_modulus_mpa  — Young's modulus [MPa]
        poissons_ratio       — dimensionless
        density_g_per_cm3    — mass density [g/cm³]
        yield_strength_mpa   — 0.2% proof stress [MPa]
        fatigue_limit_mpa    — endurance / fatigue limit [MPa]

    ARCHITECTURAL DECISION — fatigue_limit_mpa is Optional[float], not float:
        Fatigue data for titanium alloys is loading-condition-dependent and
        may not be well-established for a specific material lot or surface
        condition.  Marking it Optional forces downstream code
        (fatigue_model.py) to handle the absent case explicitly rather than
        silently using a hard-coded fallback.  If None, the fatigue-risk proxy
        should be flagged as low-confidence in its output.

    NOTE: fatigue_limit_mpa is used only as a threshold reference for the
    fatigue-risk *proxy* score.  It is NOT a validated fatigue life
    prediction input and must not be described as such in any report output.
    """

    name: str
    elastic_modulus_mpa: float
    poissons_ratio: float
    density_g_per_cm3: float
    yield_strength_mpa: float
    fatigue_limit_mpa: float | None = None   # placeholder; see NOTE above
    notes: str | None = None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Validate material property ranges."""
        if not self.name:
            raise ValueError("Material name must not be empty.")
        if self.elastic_modulus_mpa <= 0.0:
            raise ValueError(
                f"elastic_modulus_mpa must be positive, "
                f"got {self.elastic_modulus_mpa}"
            )
        if not (0.0 < self.poissons_ratio < 0.5):
            # Thermodynamic stability requires ν ∈ (−1, 0.5) for isotropic
            # materials; we restrict further to (0, 0.5) for titanium alloys.
            raise ValueError(
                f"poissons_ratio must be in (0, 0.5), "
                f"got {self.poissons_ratio}"
            )
        if self.density_g_per_cm3 <= 0.0:
            raise ValueError(
                f"density_g_per_cm3 must be positive, "
                f"got {self.density_g_per_cm3}"
            )
        if self.yield_strength_mpa <= 0.0:
            raise ValueError(
                f"yield_strength_mpa must be positive, "
                f"got {self.yield_strength_mpa}"
            )
        fatigue = self.fatigue_limit_mpa
        if fatigue is not None and fatigue <= 0.0:
            raise ValueError(
                f"fatigue_limit_mpa must be positive if provided, "
                f"got {fatigue}"
            )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "elastic_modulus_mpa": self.elastic_modulus_mpa,
            "poissons_ratio": self.poissons_ratio,
            "density_g_per_cm3": self.density_g_per_cm3,
            "yield_strength_mpa": self.yield_strength_mpa,
            "fatigue_limit_mpa": self.fatigue_limit_mpa,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MaterialDefinition":
        """Deserialise from a plain dictionary."""
        return cls(
            name=str(data["name"]),
            elastic_modulus_mpa=float(data["elastic_modulus_mpa"]),
            poissons_ratio=float(data["poissons_ratio"]),
            density_g_per_cm3=float(data["density_g_per_cm3"]),
            yield_strength_mpa=float(data["yield_strength_mpa"]),
            fatigue_limit_mpa=(
                float(data["fatigue_limit_mpa"])
                if data.get("fatigue_limit_mpa") is not None
                else None
            ),
            notes=data.get("notes"),
        )


# ---------------------------------------------------------------------------
# Load case definition
# ---------------------------------------------------------------------------

@dataclass
class LoadCaseDefinition:
    """
    Descriptor for a single mechanical load case.

    Flexible enough to cover:
        - Axial compression  (force_n is the magnitude; sign convention
                              is defined per load case type in loadcases.py)
        - Axial tension      (force_n positive magnitude)
        - Three-point bending (span stored in metadata["span_mm"])
        - Cyclic / fatigue-risk proxy (R-ratio stored in metadata["r_ratio"])

    ARCHITECTURAL DECISION — extras in metadata: dict[str, Any]:
        Rather than subclassing LoadCaseDefinition for each load type (which
        would complicate the union types in CaseDefinition), load-case-specific
        parameters (bending span, cyclic R-ratio, etc.) are carried in the
        open ``metadata`` dict.  The simulation/loadcases.py module is
        responsible for knowing which keys to expect for each LoadCaseType
        and must raise clearly if required keys are missing.  This keeps the
        schema stable while allowing load case diversity.

    NOTE: Fatigue cases in version 1 produce a risk proxy only, not a
    validated fatigue life prediction.
    """

    load_case_type: LoadCaseType
    name: str
    force_n: float | None = None           # N — applied force magnitude
    description: str | None = None
    boundary_condition_label: str | None = None

    # ARCHITECTURAL DECISION — mutable default via field(default_factory=dict):
    #   Mutable defaults (list, dict) must use default_factory in dataclasses
    #   to avoid the classic shared-mutable-default bug where all instances
    #   share the same dict object.  This pattern is used consistently for all
    #   dict and list fields throughout this file.
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Validate load case fields."""
        if not self.name:
            raise ValueError("LoadCaseDefinition.name must not be empty.")
        if not isinstance(self.load_case_type, LoadCaseType):
            raise ValueError(
                f"load_case_type must be a LoadCaseType enum, "
                f"got {self.load_case_type!r}"
            )
        if self.force_n is not None and self.force_n == 0.0:
            # A zero force is almost certainly a configuration error.
            raise ValueError(
                "force_n must be non-zero if provided (use positive for "
                "tension/compression magnitude; sign convention is defined "
                "per load case type in loadcases.py)."
            )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "load_case_type": self.load_case_type.value,
            "name": self.name,
            "force_n": self.force_n,
            "description": self.description,
            "boundary_condition_label": self.boundary_condition_label,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoadCaseDefinition":
        """Deserialise from a plain dictionary."""
        lc_val = str(data["load_case_type"])
        options = [LoadCaseType.AXIAL_COMPRESSION, LoadCaseType.AXIAL_TENSION, LoadCaseType.BENDING, LoadCaseType.CYCLIC]
        lctype = next((e for e in options if e.value == lc_val), None)
        if lctype is None:
            raise ValueError(f"Unknown load case type: {lc_val}")
        return cls(
            load_case_type=lctype,
            name=str(data["name"]),
            force_n=(
                float(data["force_n"])
                if data.get("force_n") is not None
                else None
            ),
            description=data.get("description"),
            boundary_condition_label=data.get("boundary_condition_label"),
            metadata=dict(data.get("metadata", {})),
        )


# ---------------------------------------------------------------------------
# Case file paths
# ---------------------------------------------------------------------------

@dataclass
class CasePaths:
    """
    Canonical filesystem paths for a single pipeline case run.

    ARCHITECTURAL DECISION — strings, not pathlib.Path:
        Using plain strings keeps this model trivially JSON/YAML serialisable
        without a custom encoder.  It also avoids implicit Path behaviour
        differences between OS platforms in a pipeline that may run on Linux
        (HPC) or macOS (development).  Callers that need Path objects should
        wrap at the call site: ``Path(case_paths.mesh_directory)``.
        If this produces repeated boilerplate, a companion property layer can
        be added without changing the stored field types.

    ARCHITECTURAL DECISION — CasePaths is a separate dataclass:
        Separating path information from CaseDefinition and CaseResult allows
        the orchestrator to construct paths before a case runs (for status
        checking / cache lookup) and allows the case runner to write artifacts
        without needing the full case definition in scope.  It also makes it
        trivial to relocate a run directory (e.g. archiving) by updating one
        object rather than hunting fields across two schemas.

    Typical layout (from architecture spec)::

        runs/
        └── case_000001/
            ├── case_config.yaml        <- case_config_file
            ├── metadata.json           <- metadata_file
            ├── status.txt              <- status_file
            ├── geometry/               <- geometry_directory
            ├── mesh/                   <- mesh_directory
            ├── solver/                 <- solver_directory
            └── results/                <- results_directory
    """

    run_directory: str
    geometry_directory: str
    mesh_directory: str
    solver_directory: str
    results_directory: str
    status_file: str
    metadata_file: str
    case_config_file: str

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_run_directory(cls, run_dir: str) -> "CasePaths":
        """
        Construct canonical CasePaths by convention from a run directory path.

        ARCHITECTURAL DECISION — centralise path conventions in this factory:
            All sub-path construction happens once here.  No other module
            should hard-code sub-directory names.  If the layout changes
            (e.g. "mesh" renamed to "meshes"), this is the only place that
            needs updating.

        Args:
            run_dir: Root directory for this case, e.g. ``runs/case_000001``.

        Returns:
            CasePaths with all sub-paths derived by convention.
        """
        # Local import: os is stdlib.  Kept local to avoid importing it at
        # module level for a single method's benefit.
        import os

        return cls(
            run_directory=run_dir,
            geometry_directory=os.path.join(run_dir, "geometry"),
            mesh_directory=os.path.join(run_dir, "mesh"),
            solver_directory=os.path.join(run_dir, "solver"),
            results_directory=os.path.join(run_dir, "results"),
            status_file=os.path.join(run_dir, "status.txt"),
            metadata_file=os.path.join(run_dir, "metadata.json"),
            case_config_file=os.path.join(run_dir, "case_config.yaml"),
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, str]:
        return {
            "run_directory": self.run_directory,
            "geometry_directory": self.geometry_directory,
            "mesh_directory": self.mesh_directory,
            "solver_directory": self.solver_directory,
            "results_directory": self.results_directory,
            "status_file": self.status_file,
            "metadata_file": self.metadata_file,
            "case_config_file": self.case_config_file,
        }


# ---------------------------------------------------------------------------
# Case definition
# ---------------------------------------------------------------------------

@dataclass
class CaseDefinition:
    """
    Full specification of one runnable pipeline case.

    ARCHITECTURAL DECISION — one case = one combination of design + material
    + load case + plate_thickness:
        This granularity means cases can be run individually, parallelised,
        requeued on failure, and result-cached without any coupling between
        them.  The orchestrator (orchestrator.py) generates the full Cartesian
        product of parameter combinations and hands each to a case runner.

    ARCHITECTURAL DECISION — CaseDefinition is immutable input data:
        It carries no mutable state (status, results, paths).  Status lives on
        CaseResult; paths live on CasePaths.  This separation means a
        definition can be serialised, hashed, reloaded, and requeued without
        risk of stale state being re-used from a previous run.

    ARCHITECTURAL DECISION — lattice_repeats_x/y stored here, not hard-coded
    in the geometry module:
        Version 1 always uses 5×3, but storing the repeats on CaseDefinition
        makes the schema honest about what controls lattice size.  A future
        version supporting non-standard lattice sizes can sweep these without
        changing the geometry module's interface.
    """

    case_id: str
    design_parameters: DesignParameterSet
    plate_thickness: float              # mm — extrusion depth in Z
    material: MaterialDefinition
    load_case: LoadCaseDefinition
    lattice_repeats_x: int = 5          # version 1: always 5
    lattice_repeats_y: int = 3          # version 1: always 3
    reference_step_path: str | None = None  # reference geometry only; not used by pipeline
    notes: str | None = None

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def design_type(self) -> DesignType:
        """Shortcut to the design type enum from the embedded parameter set."""
        return self.design_parameters.design_type

    @property
    def case_label(self) -> str:
        """
        Human-readable case identifier composed from key discriminators.

        Example: ``reentrant__Ti6Al4V__axial_compression__t5.0``

        ARCHITECTURAL DECISION — case_label as a property, not a stored field:
            Storing it would risk the label drifting from the actual parameter
            values if a sweep loop mutates parameters after construction.
            As a property it is always consistent with the current field
            values and requires no synchronisation logic.
        """
        return (
            f"{self.design_type.value}"
            f"__{self.material.name.replace(' ', '_')}"
            f"__{self.load_case.load_case_type.value}"
            f"__t{self.plate_thickness:.1f}"
        )

    def parameter_signature(self) -> str:
        """
        Return a short deterministic string encoding the independent design
        parameters and plate thickness.

        Used for logging and as a human-readable prefix of cache keys.

        ARCHITECTURAL DECISION — parameter_signature is NOT a unique hash:
            It is intentionally readable so logs and filenames are scannable
            by eye.  Collision-proof uniqueness is the responsibility of
            src/workflow/hashing.py, which should hash the full to_dict()
            output.  parameter_signature is a label prefix only.

        ARCHITECTURAL DECISION — derived values stripped from signature:
            Only independent parameters appear.  Derived values
            (fillet_radius_derived) are excluded so that two configs that are
            parametrically identical produce the same signature regardless of
            any floating-point rounding in derived fields.

        Example:
            ``reentrant|cell_size7.85|wall_thickness1.5|reentrant_angle_deg70.0|t4.0``
        """
        params = self.design_parameters.to_dict()
        # Remove design_type — it already appears as the prefix.
        params.pop("design_type", None)
        # Remove derived values — signature must reflect only independent params.
        params.pop("fillet_radius_derived", None)
        parts_list: list[str] = [str(self.design_parameters.design_type.value)]
        for k, v in params.items():
            parts_list.append(f"{k}{v}")
        parts_list.append(f"t{self.plate_thickness:.4f}")
        return "|".join(parts_list)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """
        Validate the full case definition and all embedded sub-models.

        ARCHITECTURAL DECISION — cascading validation:
            CaseDefinition.validate() calls validate() on each sub-model so
            that a single top-level call is sufficient to catch errors
            anywhere in the case tree.  Callers should always call
            case.validate() after constructing a case from deserialised data.

        Raises:
            ValueError: if any component fails validation.
        """
        if not self.case_id:
            raise ValueError("case_id must not be empty.")
        if self.plate_thickness <= 0.0:
            raise ValueError(
                f"plate_thickness must be positive, got {self.plate_thickness}"
            )
        if self.lattice_repeats_x <= 0:
            raise ValueError(
                f"lattice_repeats_x must be positive, "
                f"got {self.lattice_repeats_x}"
            )
        if self.lattice_repeats_y <= 0:
            raise ValueError(
                f"lattice_repeats_y must be positive, "
                f"got {self.lattice_repeats_y}"
            )
        # Cascade into sub-models; any ValueError propagates to the caller.
        self.design_parameters.validate()
        self.material.validate()
        self.load_case.validate()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            # case_label included as a convenience for human readers of saved
            # YAML/JSON.  It is recomputed from fields on load, never stored.
            "case_label": self.case_label,
            "design_parameters": self.design_parameters.to_dict(),
            "plate_thickness": self.plate_thickness,
            "material": self.material.to_dict(),
            "load_case": self.load_case.to_dict(),
            "lattice_repeats_x": self.lattice_repeats_x,
            "lattice_repeats_y": self.lattice_repeats_y,
            "reference_step_path": self.reference_step_path,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CaseDefinition":
        return cls(
            case_id=str(data["case_id"]),
            design_parameters=design_parameters_from_dict(data["design_parameters"]),
            plate_thickness=float(data["plate_thickness"]),
            material=MaterialDefinition.from_dict(data["material"]),
            load_case=LoadCaseDefinition.from_dict(data["load_case"]),
            lattice_repeats_x=int(data.get("lattice_repeats_x", 5)),
            lattice_repeats_y=int(data.get("lattice_repeats_y", 3)),
            reference_step_path=data.get("reference_step_path"),
            notes=data.get("notes"),
        )


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

@dataclass
class MetricSet:
    """
    Extracted engineering metrics from one solved case.

    All stress / modulus values : MPa
    All displacement values     : mm
    Stiffness                   : N/mm

    ARCHITECTURAL DECISION — all metric fields are Optional[float], not float:
        Not every load case produces every metric (e.g. effective_modulus_mpa
        is not meaningful for a fatigue-proxy-only run; hotspot_stress_mpa
        may not be extractable from all mesh topologies).  Using None rather
        than 0.0 or NaN makes absence of data explicit and forces downstream
        code to handle it rather than silently treating a missing result as
        a zero result.

    ARCHITECTURAL DECISION — fatigue_risk_score is a dimensionless proxy [0,1]:
        It is NOT a validated fatigue life prediction.  It is computed by
        fatigue_model.py from stress amplitude, max stress, and material
        threshold properties.  It must be labelled clearly as a proxy in all
        reports and plots generated by reporting.py.

    ARCHITECTURAL DECISION — stress_strain_points as list of (strain, stress)
    tuples rather than two separate lists:
        Paired tuples prevent index-desync bugs and map cleanly to a list of
        2-element lists in JSON.  If large datasets are needed in a future
        version, a numpy array or a CSV artifact file is more appropriate than
        this field.
    """

    max_von_mises_stress_mpa: float | None = None
    max_displacement_mm: float | None = None
    effective_stiffness_n_per_mm: float | None = None
    effective_modulus_mpa: float | None = None
    fatigue_risk_score: float | None = None    # proxy only; NOT validated
    hotspot_stress_mpa: float | None = None
    stress_strain_points: list[tuple[float, float]] = field(
        default_factory=list
    )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_von_mises_stress_mpa": self.max_von_mises_stress_mpa,
            "max_displacement_mm": self.max_displacement_mm,
            "effective_stiffness_n_per_mm": self.effective_stiffness_n_per_mm,
            "effective_modulus_mpa": self.effective_modulus_mpa,
            "fatigue_risk_score": self.fatigue_risk_score,
            "hotspot_stress_mpa": self.hotspot_stress_mpa,
            # list[tuple] serialises to list[list] in JSON — acceptable for
            # downstream consumption; re-tupling on load is trivial if needed.
            "stress_strain_points": self.stress_strain_points,
        }


@dataclass
class CaseResult:
    """
    Complete result record for one pipeline case.

    ARCHITECTURAL DECISION — CaseResult is mutable; CaseDefinition is not:
        Status transitions (PENDING → RUNNING → COMPLETED/FAILED) happen on
        CaseResult via the ``mark_*`` methods.  This keeps all mutable state
        in one place and leaves CaseDefinition as a stable, hashable input.

    ARCHITECTURAL DECISION — mark_failed / mark_completed as explicit methods:
        Centralising status transitions in named methods (rather than letting
        callers set .status and .success directly) ensures status and success
        always stay in sync.  A future version could add transition guards
        (e.g. refuse to mark COMPLETED if all metrics are None) here without
        changing any caller.

    ``artifacts`` maps a human-readable label to a file path, e.g.::

        {
            "mesh_file":    "runs/case_000001/mesh/model.msh",
            "solver_input": "runs/case_000001/solver/input.inp",
            "raw_results":  "runs/case_000001/results/raw_results.frd",
            "metrics_json": "runs/case_000001/results/extracted_metrics.json",
        }

    ``metadata`` carries any additional diagnostic information from the run
    (solver warnings, mesh quality stats, timing breakdowns, etc.).
    """

    case_id: str
    status: CaseStatus
    success: bool
    metrics: MetricSet = field(default_factory=MetricSet)
    error_message: str | None = None
    runtime_seconds: float | None = None
    solver_return_code: int | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def mark_failed(self, error_message: str) -> None:
        """
        Transition this result to a failed state.

        Args:
            error_message: Human-readable description of the failure reason.
        """
        self.status = CaseStatus.FAILED
        self.success = False
        self.error_message = error_message

    def mark_completed(self) -> None:
        """Transition this result to a successfully completed state."""
        self.status = CaseStatus.COMPLETED
        self.success = True

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "status": self.status.value,
            "success": self.success,
            "metrics": self.metrics.to_dict(),
            "error_message": self.error_message,
            "runtime_seconds": self.runtime_seconds,
            "solver_return_code": self.solver_return_code,
            "artifacts": self.artifacts,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def is_supported_design_parameter_set(obj: Any) -> bool:
    """
    Return True if ``obj`` is an instance of a supported design parameter
    dataclass.

    Useful for runtime type checks in loading and dispatch code where the
    static type of ``obj`` is unknown (e.g. after deserialisation from YAML).

    ARCHITECTURAL DECISION — check against the three concrete classes, not
    BaseDesignParameters:
        Checking the base class would return True for a directly-instantiated
        BaseDesignParameters, which is not a valid design parameter set.
        Checking the concrete union explicitly is the safe guard.
    """
    return isinstance(
        obj,
        (ReentrantParameters, RotatingSquareParameters, TetrachiralParameters),
    )


def design_parameters_from_dict(data: dict[str, Any]) -> DesignParameterSet:
    """
    Deserialise a design parameter set from a plain dictionary.

    The dictionary must contain a ``"design_type"`` key whose value matches
    a valid ``DesignType`` enum value string (e.g. ``"reentrant"``).

    ARCHITECTURAL DECISION — dispatch table, not if/elif chain:
        A dict keyed by DesignType makes adding a new design trivial: add one
        entry here and one DesignType member.  An if/elif chain requires
        modifying control flow and is easier to miss in a future edit.  The
        dispatch table also fails loudly (KeyError) if a DesignType is added
        to the enum without a corresponding from_dict entry being added here,
        which is the correct failure mode.

    Args:
        data: Plain dictionary, typically loaded from YAML or JSON.

    Returns:
        The appropriate concrete design parameter dataclass.

    Raises:
        ValueError: if ``design_type`` is missing or unsupported.
    """
    raw_type = data.get("design_type")
    if raw_type is None:
        raise ValueError(
            "design_parameters_from_dict: 'design_type' key is required."
        )

    if not raw_type:
        raise ValueError("Missing 'design_type' in CaseDefinition data.")

    raw_type_str = str(raw_type)
    dispatch: dict[str, Any] = {
        DesignType.REENTRANT.value: ReentrantParameters.from_dict,
        DesignType.ROTATING_SQUARE.value: RotatingSquareParameters.from_dict,
        DesignType.TETRACHIRAL.value: TetrachiralParameters.from_dict,
    }

    if raw_type_str not in dispatch:
        raise ValueError(f"Unknown design type: {raw_type_str}")

    return dispatch[raw_type_str](data)
