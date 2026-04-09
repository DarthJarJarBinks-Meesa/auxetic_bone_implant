"""
src/geometry/validators.py
===========================
Geometry validation module for the auxetic plate pipeline.

This module provides lightweight, structured validation checks at the
unit-cell, lattice, and 3D-solid stages of the pipeline.  It acts as a
screening layer before expensive downstream operations (meshing, solving).

PIPELINE POSITION:
    unit cell → lattice → solid  →  [THIS MODULE validates each stage]
                                 →  mesher.py

ARCHITECTURAL DECISION — structured ValidationResult, not bare exceptions:
    Most validation functions return a ``ValidationResult`` rather than
    raising on the first error.  This allows callers to collect the full
    picture of what is wrong with a geometry before deciding whether to
    abort or log-and-continue.  Only the ``require_*`` hard-fail wrappers
    raise ``GeometryValidationError``, and only when the caller explicitly
    wants a hard stop (e.g. immediately before meshing in a strict-mode run).

ARCHITECTURAL DECISION — bounding-box based checks only:
    Deep topological analysis (self-intersections, non-manifold edges,
    minimum wall thickness extraction) is expensive and partially unreliable
    via the CadQuery / OCC kernel in version 1.  Bounding-box checks are
    fast, reliable, and sufficient to catch the most common failure modes
    (zero-area geometry, degenerate extrusion, parameter combinations that
    collapse the cell topology).  Advanced topology checks are future work.

ARCHITECTURAL DECISION — graceful degradation on CadQuery introspection:
    CadQuery's API for extracting bounding boxes can raise or return
    unexpected values for degenerate or non-solid objects.  All bounding-box
    extraction is wrapped in try/except and produces a warning rather than
    crashing, so that partial metadata is still collected for logging even
    when the geometry is clearly invalid.

ARCHITECTURAL DECISION — minimum_feature_size_mm generates warnings, not errors:
    A feature size below the threshold is flagged as a warning, not an error,
    because the threshold is a manufacturability guideline (SLM Ti minimum
    wall), not a fundamental geometric impossibility.  Callers that want
    hard failure for thin features should check ``result.is_valid`` after
    inspecting the messages or use strict_mode in the orchestrator.

UNITS: mm throughout.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Union, Optional

import cadquery as cq

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Geometry2D = Union[cq.Workplane, cq.Sketch]
"""Type alias for 2D planar CadQuery geometry."""

Solid3D = cq.Workplane
"""Type alias for 3D solid CadQuery geometry."""

# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class GeometryValidationError(Exception):
    """
    Raised by hard-fail ``require_*`` wrappers when geometry is invalid.

    Standard validation functions return ``ValidationResult`` and do not
    raise this exception.  Only callers that explicitly cannot proceed with
    invalid geometry (e.g. immediately before a mesh step) should use the
    ``require_*`` helpers.
    """

# ---------------------------------------------------------------------------
# Validation result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ValidationMessage:
    """
    A single validation diagnostic message.

    Attributes:
        level:   Severity string — one of ``"info"``, ``"warning"``,
                 ``"error"``.
        code:    Short machine-readable identifier (e.g. ``"ZERO_VOLUME"``).
        message: Human-readable description.
    """
    level: str    # "info" | "warning" | "error"
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"level": self.level, "code": self.code, "message": self.message}


@dataclass
class ValidationResult:
    """
    Structured outcome of a geometry validation check.

    ``is_valid`` is ``True`` iff no ``"error"`` level messages have been
    added.  Warnings and infos do not mark the result invalid.

    ARCHITECTURAL DECISION — is_valid is derived from messages, not stored:
        ``is_valid`` starts as ``True`` and is set to ``False`` the first
        time ``add_error()`` is called.  This prevents the flag from drifting
        out of sync with the message list.

    Attributes:
        is_valid:  ``True`` if no errors have been added.
        messages:  Ordered list of ``ValidationMessage`` entries.
        metadata:  Optional key/value pairs (bounding-box dims, etc.) for
                   logging and reporting.
    """
    is_valid: bool = True
    messages: list[ValidationMessage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Message builders
    # ------------------------------------------------------------------

    def add_info(self, code: str, message: str) -> None:
        """Append an informational message.  Does not affect ``is_valid``."""
        self.messages.append(ValidationMessage("info", code, message))
        logger.debug("[VALIDATION INFO] %s: %s", code, message)

    def add_warning(self, code: str, message: str) -> None:
        """Append a warning message.  Does not affect ``is_valid``."""
        self.messages.append(ValidationMessage("warning", code, message))
        logger.warning("[VALIDATION WARNING] %s: %s", code, message)

    def add_error(self, code: str, message: str) -> None:
        """
        Append an error message and mark the result as invalid.

        Once ``is_valid`` is set to ``False`` it remains ``False``.
        """
        self.messages.append(ValidationMessage("error", code, message))
        self.is_valid = False
        logger.error("[VALIDATION ERROR] %s: %s", code, message)

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def merge(self, other: ValidationResult) -> None:
        """
        Merge another ``ValidationResult`` into this one in place.

        All messages from ``other`` are appended; if ``other`` is invalid,
        this result becomes invalid too.  Metadata dicts are merged with
        ``other`` taking precedence on key collision.
        """
        self.messages.extend(other.messages)
        if not other.is_valid:
            self.is_valid = False
        self.metadata.update(other.metadata)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "is_valid": self.is_valid,
            "error_count":   sum(1 for m in self.messages if m.level == "error"),
            "warning_count": sum(1 for m in self.messages if m.level == "warning"),
            "info_count":    sum(1 for m in self.messages if m.level == "info"),
            "messages": [m.to_dict() for m in self.messages],
            "metadata": self.metadata,
        }

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def errors(self) -> list[ValidationMessage]:
        """Return only error-level messages."""
        return [m for m in self.messages if m.level == "error"]

    @property
    def warnings(self) -> list[ValidationMessage]:
        """Return only warning-level messages."""
        return [m for m in self.messages if m.level == "warning"]


# ---------------------------------------------------------------------------
# Bounding-box helpers
# ---------------------------------------------------------------------------

def _safe_bounding_box(obj: Any) -> Optional[tuple[float, float, float]]:
    """
    Attempt to extract ``(xsize, ysize, zsize)`` from a CadQuery object.

    Returns ``None`` on any failure rather than raising, so that downstream
    validators can emit a structured warning instead of crashing.

    """
    def _wp_bbox(o: Any) -> Any:
        # A CadQuery Workplane might have multiple items on the stack 
        # (e.g. the accumulated lattice tiles). Computing the bounding box 
        # of the compound guarantees we capture the full tiled extents.
        import cadquery as cq
        if hasattr(o, "vals") and o.vals():
            return cq.Compound.makeCompound(o.vals()).BoundingBox()
        return o.val().BoundingBox()

    for extractor in (
        _wp_bbox,
        lambda o: o.BoundingBox(),
    ):
        try:
            bb = extractor(obj)
            return (
                bb.xmax - bb.xmin,
                bb.ymax - bb.ymin,
                bb.zmax - bb.zmin,
            )
        except Exception:
            continue
    return None


def _bounding_box_metadata(obj: Any) -> dict[str, float]:
    """
    Return a metadata dict with bounding-box dimensions, or an empty dict
    if extraction fails.

    Returns:
        Dict with keys ``xsize_mm``, ``ysize_mm``, ``zsize_mm`` on success,
        or an empty dict on failure.
    """
    dims = _safe_bounding_box(obj)
    if dims is None:
        return {}
    return {
        "xsize_mm": float(f"{dims[0]:.6f}"),
        "ysize_mm": float(f"{dims[1]:.6f}"),
        "zsize_mm": float(f"{dims[2]:.6f}"),
    }


# ---------------------------------------------------------------------------
# Basic scalar validators
# ---------------------------------------------------------------------------

def validate_positive_scalar(name: str, value: float) -> ValidationResult:
    """
    Validate that a named scalar value is strictly positive.

    Args:
        name:  Human-readable parameter name for messages.
        value: Numeric value to check.

    Returns:
        ``ValidationResult`` — valid if ``value > 0``, error otherwise.
    """
    result = ValidationResult()
    if value <= 0.0:
        result.add_error(
            "NON_POSITIVE_SCALAR",
            f"'{name}' must be strictly positive, got {value}.",
        )
    else:
        result.add_info(
            "POSITIVE_SCALAR_OK",
            f"'{name}' = {value} is positive.",
        )
    return result


def validate_minimum_feature_size(
    value_mm: float,
    minimum_mm: float,
    context: str,
) -> ValidationResult:
    """
    Emit a warning if ``value_mm`` is below the minimum feature size threshold.

    ARCHITECTURAL DECISION — warning, not error:
        See module docstring.  The threshold is a manufacturability guideline.

    Args:
        value_mm:   The feature size to check [mm].
        minimum_mm: The minimum acceptable size [mm].
        context:    Human-readable description of what is being checked.

    Returns:
        ``ValidationResult`` — always valid; may contain a warning.
    """
    result = ValidationResult()
    if value_mm < minimum_mm:
        result.add_warning(
            "FEATURE_BELOW_MINIMUM",
            f"{context}: {value_mm:.4f} mm is below the minimum feature "
            f"size threshold of {minimum_mm:.4f} mm.  "
            f"This may cause meshing difficulties or manufacturability issues.",
        )
    return result


# ---------------------------------------------------------------------------
# Planar geometry validation
# ---------------------------------------------------------------------------

def validate_planar_geometry(
    planar_geometry: Any,
    minimum_feature_size_mm: Optional[float] = None,
) -> ValidationResult:
    """
    Validate a 2D CadQuery planar geometry object.

    Checks:
      - not ``None``
      - recognised CadQuery type (``cq.Workplane`` or ``cq.Sketch``)
      - bounding box extractable (warning if not)
      - X and Y extents are positive
      - if ``minimum_feature_size_mm`` supplied, warn when smallest XY
        extent is below it

    Args:
        planar_geometry:       2D CadQuery geometry to validate.
        minimum_feature_size_mm: Optional minimum feature size threshold [mm].

    Returns:
        ``ValidationResult`` with messages and bounding-box metadata.
    """
    result = ValidationResult()

    # --- None check ---
    if planar_geometry is None:
        result.add_error(
            "PLANAR_GEOMETRY_NONE",
            "Planar geometry is None.  The unit-cell build or lattice tiling "
            "step did not return a valid geometry object.",
        )
        return result  # no further checks possible

    # --- Type check ---
    if not isinstance(planar_geometry, (cq.Workplane, cq.Sketch)):
        result.add_error(
            "PLANAR_GEOMETRY_WRONG_TYPE",
            f"Planar geometry must be cq.Workplane or cq.Sketch, "
            f"got {type(planar_geometry).__name__}.",
        )
        return result

    # --- Bounding-box extraction ---
    bb_meta = _bounding_box_metadata(planar_geometry)
    result.metadata.update({"planar_bounding_box_mm": bb_meta})

    if not bb_meta:
        result.add_warning(
            "PLANAR_BBOX_UNAVAILABLE",
            "Could not extract bounding-box dimensions from the planar "
            "geometry.  Some downstream checks are skipped.  "
            "The geometry may be empty or degenerate.",
        )
        return result

    x_size = bb_meta["xsize_mm"]
    y_size = bb_meta["ysize_mm"]

    # --- Positive XY extents ---
    if x_size <= 0.0:
        result.add_error(
            "PLANAR_ZERO_X_EXTENT",
            f"Planar geometry has zero or negative X extent ({x_size:.6f} mm).  "
            f"The geometry is empty or degenerate in the X direction.",
        )
    if y_size <= 0.0:
        result.add_error(
            "PLANAR_ZERO_Y_EXTENT",
            f"Planar geometry has zero or negative Y extent ({y_size:.6f} mm).  "
            f"The geometry is empty or degenerate in the Y direction.",
        )

    if result.is_valid:
        result.add_info(
            "PLANAR_EXTENTS_OK",
            f"Planar geometry XY extents: {x_size:.4f} mm × {y_size:.4f} mm.",
        )

    # --- Minimum feature size check ---
    if minimum_feature_size_mm is not None and bb_meta:
        smallest_xy = min(x_size, y_size)
        feature_check = validate_minimum_feature_size(
            smallest_xy,
            minimum_feature_size_mm,
            "Smallest planar bounding-box dimension",
        )
        result.merge(feature_check)

    return result


# ---------------------------------------------------------------------------
# Solid geometry validation
# ---------------------------------------------------------------------------

def validate_solid_geometry(
    solid_geometry: Any,
    minimum_feature_size_mm: Optional[float] = None,
) -> ValidationResult:
    """
    Validate a 3D CadQuery solid geometry object.

    Checks:
      - not ``None``
      - recognised CadQuery type (``cq.Workplane``)
      - bounding box extractable
      - X, Y, and Z extents are all positive and non-trivial
      - if ``minimum_feature_size_mm`` supplied, warn on thin dimensions

    Args:
        solid_geometry:          3D CadQuery solid to validate.
        minimum_feature_size_mm: Optional minimum feature size threshold [mm].

    Returns:
        ``ValidationResult`` with messages and bounding-box metadata.
    """
    result = ValidationResult()

    # --- None check ---
    if solid_geometry is None:
        result.add_error(
            "SOLID_GEOMETRY_NONE",
            "Solid geometry is None.  The extrusion step did not return "
            "a valid 3D object.",
        )
        return result

    # --- Type check ---
    if not isinstance(solid_geometry, cq.Workplane):
        result.add_error(
            "SOLID_GEOMETRY_WRONG_TYPE",
            f"Solid geometry must be cq.Workplane, "
            f"got {type(solid_geometry).__name__}.",
        )
        return result

    # --- Bounding-box extraction ---
    bb_meta = _bounding_box_metadata(solid_geometry)
    result.metadata.update({"solid_bounding_box_mm": bb_meta})

    if not bb_meta:
        result.add_warning(
            "SOLID_BBOX_UNAVAILABLE",
            "Could not extract bounding-box dimensions from the solid "
            "geometry.  Some downstream checks are skipped.  "
            "The solid may be empty or degenerate.",
        )
        return result

    x_size = bb_meta["xsize_mm"]
    y_size = bb_meta["ysize_mm"]
    z_size = bb_meta["zsize_mm"]

    # --- Positive XYZ extents ---
    for axis, size in (("X", x_size), ("Y", y_size), ("Z", z_size)):
        if size <= 0.0:
            result.add_error(
                f"SOLID_ZERO_{axis}_EXTENT",
                f"Solid geometry has zero or negative {axis} extent "
                f"({size:.6f} mm).  The solid is degenerate.",
            )

    # --- Z extent sanity (plate_thickness proxy check) ---
    # Z size should be positive and meaningful.  A degenerate zero-Z solid
    # indicates extrusion failure.  This is already caught above, but
    # we add a more descriptive message for the Z case.
    if z_size <= 0.0:
        result.add_error(
            "SOLID_ZERO_THICKNESS",
            f"Solid Z extent is {z_size:.6f} mm.  "
            f"Extrusion likely failed or plate_thickness was zero.",
        )

    if result.is_valid:
        result.add_info(
            "SOLID_EXTENTS_OK",
            f"Solid bounding box: {x_size:.4f} × {y_size:.4f} × {z_size:.4f} mm.",
        )

    # --- Minimum feature size check across all three dimensions ---
    if minimum_feature_size_mm is not None and bb_meta:
        for axis, size in (("X", x_size), ("Y", y_size), ("Z", z_size)):
            feature_check = validate_minimum_feature_size(
                size,
                minimum_feature_size_mm,
                f"Solid {axis} bounding-box extent",
            )
            result.merge(feature_check)

    return result


# ---------------------------------------------------------------------------
# Unit-cell object validation
# ---------------------------------------------------------------------------

def validate_unit_cell_object(
    unit_cell: Any,
    build_and_check_2d: bool = False,
    minimum_feature_size_mm: Optional[float] = None,
) -> ValidationResult:
    """
    Validate a ``BaseUnitCell`` subclass instance.

    Checks:
      - not ``None``
      - exposes the expected interface (``validate()``, ``build_2d()``,
        ``to_metadata_dict()``)
      - ``validate()`` runs without raising
      - if ``build_and_check_2d=True``, builds the 2D geometry and validates
        it with ``validate_planar_geometry``

    ARCHITECTURAL DECISION — build_and_check_2d is opt-in:
        Building the 2D geometry can be expensive and may raise.  Callers
        that only want parameter validation (e.g. in sweep pre-filtering)
        can leave ``build_and_check_2d=False`` for speed.

    Args:
        unit_cell:               Any object intended to be a BaseUnitCell.
        build_and_check_2d:      If ``True``, also build and validate 2D geometry.
        minimum_feature_size_mm: Passed to planar validation if 2D is built.

    Returns:
        ``ValidationResult``.
    """
    result = ValidationResult()

    # --- None check ---
    if unit_cell is None:
        result.add_error(
            "UNIT_CELL_NONE",
            "unit_cell object is None.",
        )
        return result

    # --- Interface check (duck-typed) ---
    required_attrs = ["validate", "build_2d", "to_metadata_dict", "cell_size", "design_type"]
    missing = [attr for attr in required_attrs if not hasattr(unit_cell, attr)]
    if missing:
        result.add_error(
            "UNIT_CELL_MISSING_INTERFACE",
            f"unit_cell object {type(unit_cell).__name__!r} is missing "
            f"expected attributes: {missing}.  "
            f"Ensure it is a concrete BaseUnitCell subclass.",
        )
        return result

    # --- Parameter validation ---
    try:
        unit_cell.validate()
        result.add_info(
            "UNIT_CELL_PARAMS_VALID",
            f"unit_cell.validate() passed for {type(unit_cell).__name__} "
            f"(design={unit_cell.design_type.value}, "
            f"cell_size={unit_cell.cell_size} mm).",
        )
    except ValueError as exc:
        result.add_error(
            "UNIT_CELL_PARAMS_INVALID",
            f"unit_cell.validate() raised ValueError for "
            f"{type(unit_cell).__name__}: {exc}",
        )
        # If parameter validation failed, skip 2D build check
        return result

    # --- Optional 2D geometry build and check ---
    if build_and_check_2d:
        try:
            geometry_2d = unit_cell.build_2d()
        except Exception as exc:
            result.add_error(
                "UNIT_CELL_BUILD_2D_FAILED",
                f"unit_cell.build_2d() raised an exception for "
                f"{type(unit_cell).__name__}: {exc}",
            )
            return result

        planar_result = validate_planar_geometry(
            geometry_2d,
            minimum_feature_size_mm=minimum_feature_size_mm,
        )
        result.merge(planar_result)

    return result


# ---------------------------------------------------------------------------
# Lattice geometry validation
# ---------------------------------------------------------------------------

def validate_lattice_geometry(
    lattice_geometry: Any,
    expected_repeats_x: Optional[int] = None,
    expected_repeats_y: Optional[int] = None,
    cell_size: Optional[float] = None,
    minimum_feature_size_mm: Optional[float] = None,
) -> ValidationResult:
    """
    Validate a tiled 2D lattice geometry.

    Delegates to ``validate_planar_geometry`` and adds lattice-specific
    dimensional plausibility checks when ``expected_repeats_x``,
    ``expected_repeats_y``, and ``cell_size`` are provided.

    Args:
        lattice_geometry:    2D tiled lattice geometry.
        expected_repeats_x:  Expected number of unit cells in X (optional).
        expected_repeats_y:  Expected number of unit cells in Y (optional).
        cell_size:           Nominal unit-cell envelope [mm] (optional).
        minimum_feature_size_mm: Minimum feature size threshold [mm] (optional).

    Returns:
        ``ValidationResult`` with planar checks and optional lattice checks.
    """
    result = validate_planar_geometry(
        lattice_geometry,
        minimum_feature_size_mm=minimum_feature_size_mm,
    )

    # Record lattice configuration in metadata
    result.metadata["lattice_expected_repeats_x"] = expected_repeats_x
    result.metadata["lattice_expected_repeats_y"] = expected_repeats_y
    result.metadata["lattice_cell_size_mm"] = cell_size

    # --- Dimensional plausibility check ---
    # If all three parameters are provided, check that bounding-box dimensions
    # are in the right ballpark (within ±20% of expected nominal size).
    # This is a loose sanity check, not an exact assertion.
    if (
        expected_repeats_x is not None
        and expected_repeats_y is not None
        and cell_size is not None
        and cell_size > 0.0
        and result.is_valid  # skip if basic checks already failed
    ):
        bb_meta = result.metadata.get("planar_bounding_box_mm", {})
        if bb_meta:
            expected_x = expected_repeats_x * cell_size
            expected_y = expected_repeats_y * cell_size
            actual_x = bb_meta.get("xsize_mm", 0.0)
            actual_y = bb_meta.get("ysize_mm", 0.0)

            tolerance = 0.25  # allow ±25% for auxetic topology variations

            for axis, actual, expected in (
                ("X", actual_x, expected_x),
                ("Y", actual_y, expected_y),
            ):
                if expected > 0:
                    ratio = actual / expected
                    if not (1.0 - tolerance <= ratio <= 1.0 + tolerance):
                        result.add_warning(
                            f"LATTICE_SIZE_{axis}_UNEXPECTED",
                            f"Lattice {axis} extent ({actual:.4f} mm) differs from "
                            f"expected nominal ({expected:.4f} mm) by "
                            f"{abs(ratio - 1.0) * 100:.1f}%  "
                            f"(tolerance ±{tolerance*100:.0f}%).  "
                            f"Check the lattice tiling output.",
                        )
                    else:
                        result.add_info(
                            f"LATTICE_SIZE_{axis}_OK",
                            f"Lattice {axis} extent {actual:.4f} mm is within "
                            f"±{tolerance*100:.0f}% of expected {expected:.4f} mm.",
                        )

    return result


# ---------------------------------------------------------------------------
# Case-level geometry validation
# ---------------------------------------------------------------------------

def validate_case_geometry(
    case_definition: Any,
    planar_geometry: Optional[Any] = None,
    solid_geometry: Optional[Any] = None,
    minimum_feature_size_mm: Optional[float] = None,
) -> ValidationResult:
    """
    Validate geometry associated with a pipeline case definition.

    Combines plate-thickness validation, planar geometry validation, and
    solid geometry validation into one merged ``ValidationResult``.

    ARCHITECTURAL DECISION — duck-typed on case_definition:
        ``CaseDefinition`` is not imported explicitly to avoid cross-layer
        coupling.  Attributes ``plate_thickness``, ``lattice_repeats_x``, and
        ``lattice_repeats_y`` are accessed via ``getattr`` with safe defaults.

    Args:
        case_definition:         Object with plate_thickness attribute.
        planar_geometry:         2D lattice geometry (optional).
        solid_geometry:          3D solid geometry (optional).
        minimum_feature_size_mm: Minimum feature size threshold [mm] (optional).

    Returns:
        Merged ``ValidationResult``.
    """
    result = ValidationResult()

    # --- Plate thickness ---
    plate_thickness = getattr(case_definition, "plate_thickness", None)
    if plate_thickness is not None:
        thickness_check = validate_positive_scalar("plate_thickness", plate_thickness)
        result.merge(thickness_check)
        result.metadata["plate_thickness_mm"] = plate_thickness
    else:
        result.add_warning(
            "CASE_NO_PLATE_THICKNESS",
            "case_definition has no 'plate_thickness' attribute; "
            "plate thickness validation skipped.",
        )

    # --- Planar geometry ---
    if planar_geometry is not None:
        repeats_x = getattr(case_definition, "lattice_repeats_x", None)
        repeats_y = getattr(case_definition, "lattice_repeats_y", None)
        cell_size = getattr(
            getattr(case_definition, "design_parameters", None),
            "cell_size",
            None,
        )
        planar_result = validate_lattice_geometry(
            planar_geometry,
            expected_repeats_x=repeats_x,
            expected_repeats_y=repeats_y,
            cell_size=cell_size,
            minimum_feature_size_mm=minimum_feature_size_mm,
        )
        result.merge(planar_result)

    # --- Solid geometry ---
    if solid_geometry is not None:
        solid_result = validate_solid_geometry(
            solid_geometry,
            minimum_feature_size_mm=minimum_feature_size_mm,
        )
        result.merge(solid_result)

    return result


# ---------------------------------------------------------------------------
# Hard-fail wrappers
# ---------------------------------------------------------------------------

def require_valid_planar_geometry(
    planar_geometry: Any,
    minimum_feature_size_mm: Optional[float] = None,
    context: str = "",
) -> None:
    """
    Validate planar geometry and raise ``GeometryValidationError`` if invalid.

    Use this before steps that cannot recover from invalid 2D geometry
    (e.g. immediately before lattice tiling or extrusion).

    Args:
        planar_geometry:         2D geometry to validate.
        minimum_feature_size_mm: Optional minimum feature size threshold [mm].
        context:                 Optional context string for error messages.

    Raises:
        GeometryValidationError: if the validation result is not valid.
    """
    result = validate_planar_geometry(planar_geometry, minimum_feature_size_mm)
    if not result.is_valid:
        error_summaries = "; ".join(
            f"[{m.code}] {m.message}" for m in result.errors
        )
        ctx = f" ({context})" if context else ""
        raise GeometryValidationError(
            f"Planar geometry validation failed{ctx}: {error_summaries}"
        )


def require_valid_solid_geometry(
    solid_geometry: Any,
    minimum_feature_size_mm: Optional[float] = None,
    context: str = "",
) -> None:
    """
    Validate solid geometry and raise ``GeometryValidationError`` if invalid.

    Use this before meshing to ensure the 3D solid is non-degenerate.

    Args:
        solid_geometry:          3D geometry to validate.
        minimum_feature_size_mm: Optional minimum feature size threshold [mm].
        context:                 Optional context string for error messages.

    Raises:
        GeometryValidationError: if the validation result is not valid.
    """
    result = validate_solid_geometry(solid_geometry, minimum_feature_size_mm)
    if not result.is_valid:
        error_summaries = "; ".join(
            f"[{m.code}] {m.message}" for m in result.errors
        )
        ctx = f" ({context})" if context else ""
        raise GeometryValidationError(
            f"Solid geometry validation failed{ctx}: {error_summaries}"
        )
