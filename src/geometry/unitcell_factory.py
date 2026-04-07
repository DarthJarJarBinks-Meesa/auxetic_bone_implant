"""
src/geometry/unitcell_factory.py
==================================
Unit-cell factory for the auxetic plate pipeline.

This module is the canonical bridge between the workflow data-model layer
(``workflow/case_schema.py``) and the concrete 2D unit-cell generator classes
(``designs/``).  It centralises all design-class dispatch logic so that no
other module needs to know which parameter type maps to which class.

ARCHITECTURAL DECISION — factory pattern over direct imports in callers:
    If every module that needs a unit cell imports and instantiates
    ``ReentrantUnitCell`` directly, adding a new design requires editing every
    such module.  Routing through this factory means only two places change
    when a new design is added: (1) the ``_REGISTRY`` mapping here, and
    (2) ``DesignType`` enum + parameter class in ``case_schema.py``.

ARCHITECTURAL DECISION — dispatch table, not if/elif chain:
    The ``_REGISTRY`` dict keyed by ``DesignType`` is O(1) lookup and fails
    loudly (KeyError) on an unregistered design.  An if/elif chain is easier
    to miss a branch in when adding designs.  The dispatch table is checked
    by ``get_supported_design_types()`` so tests can assert completeness.

ARCHITECTURAL DECISION — no CadQuery imports here:
    This module performs only dispatch.  It never calls ``build_2d()``,
    creates geometry, or touches CadQuery.  Keeping it import-clean makes
    it fast to load in contexts where CadQuery may not be installed (e.g.
    config validation, case generation, reporting).

ARCHITECTURAL DECISION — loose typing on ``create_unit_cell_from_case``:
    Accepting ``Any`` for the case object avoids importing ``CaseDefinition``
    from ``workflow/case_schema.py`` here, which would create a cross-layer
    import that is fragile if the schema module is ever split.  Duck-typing
    on ``.design_parameters`` is sufficient and explicit enough.

Version 1 supports exactly three designs:
    REENTRANT, ROTATING_SQUARE, TETRACHIRAL
"""

from __future__ import annotations

from typing import Any

from designs.base_cell import BaseUnitCell
from designs.reentrant import ReentrantUnitCell
from designs.rotating_square import RotatingSquareUnitCell
from designs.tetrachiral import TetrachiralUnitCell
from workflow.case_schema import (
    DesignParameterSet,
    DesignType,
    ReentrantParameters,
    RotatingSquareParameters,
    TetrachiralParameters,
    design_parameters_from_dict,
)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class UnitCellFactoryError(Exception):
    """
    Raised when the unit-cell factory cannot create a requested design.

    Covers:
      - unsupported ``DesignType`` values
      - mismatched parameter object types
      - malformed input dicts missing ``design_type``
      - registry lookup failures
    """


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# ARCHITECTURAL DECISION — explicit dict literal, not dynamic discovery:
#   Automatic class discovery (e.g. scanning subclasses of BaseUnitCell) is
#   clever but hides the mapping.  An explicit dict is auditable, sortable,
#   and raises a clear KeyError if a DesignType is added to the enum without
#   a corresponding registry entry.
#
# EXTENSIBILITY NOTE: to add a new design, add one entry here.  No other
# change in this file is required.
_REGISTRY: dict[DesignType, type[BaseUnitCell]] = {
    DesignType.REENTRANT:       ReentrantUnitCell,
    DesignType.ROTATING_SQUARE: RotatingSquareUnitCell,
    DesignType.TETRACHIRAL:     TetrachiralUnitCell,
}

# Verify at import time that the registry covers every DesignType member.
# This ensures that adding a DesignType without updating _REGISTRY triggers
# an assertion error the moment this module is first imported, not later
# at runtime inside a long sweep.
_unregistered = [dt for dt in DesignType if dt not in _REGISTRY]
if _unregistered:
    raise RuntimeError(
        f"unitcell_factory._REGISTRY is missing entries for DesignType "
        f"member(s): {[dt.value for dt in _unregistered]}.  "
        f"Add the corresponding class to _REGISTRY."
    )


# ---------------------------------------------------------------------------
# Primary factory function
# ---------------------------------------------------------------------------

def create_unit_cell(parameters: DesignParameterSet) -> BaseUnitCell:
    """
    Instantiate and return the correct ``BaseUnitCell`` subclass for the
    given design parameter object.

    This is the primary entry point for all pipeline modules that need a
    unit-cell generator object.

    Args:
        parameters: A typed design parameter dataclass — one of
                    ``ReentrantParameters``, ``RotatingSquareParameters``,
                    or ``TetrachiralParameters``.

    Returns:
        The concrete ``BaseUnitCell`` subclass instance, ready for
        ``validate()`` and ``build_2d()``.

    Raises:
        UnitCellFactoryError: if ``parameters`` is not a recognised design
                              parameter type or the registry lookup fails.

    Example::

        params = ReentrantParameters(cell_size=7.85, wall_thickness=1.5,
                                     reentrant_angle_deg=70.0)
        cell = create_unit_cell(params)
        sketch = cell.build_and_validate_2d()
    """
    if not isinstance(
        parameters,
        (ReentrantParameters, RotatingSquareParameters, TetrachiralParameters),
    ):
        raise UnitCellFactoryError(
            f"create_unit_cell() requires a supported DesignParameterSet "
            f"(ReentrantParameters, RotatingSquareParameters, or "
            f"TetrachiralParameters).  Got: {type(parameters).__name__}.  "
            f"Use design_parameters_from_dict() to construct a parameter "
            f"object from a raw dict."
        )

    design_type = parameters.design_type

    try:
        cell_class = _REGISTRY[design_type]
    except KeyError:
        raise UnitCellFactoryError(
            f"No unit-cell class is registered for DesignType "
            f"'{design_type.value}'.  "
            f"Supported types: {[dt.value for dt in _REGISTRY]}."
        ) from None

    return cell_class(parameters)


# ---------------------------------------------------------------------------
# Secondary constructor from dict
# ---------------------------------------------------------------------------

def create_unit_cell_from_dict(data: dict[str, Any]) -> BaseUnitCell:
    """
    Parse a raw dictionary into design parameters and instantiate the correct
    unit-cell generator.

    Delegates parameter parsing to ``design_parameters_from_dict()`` from
    ``workflow/case_schema.py``, which dispatches on the ``"design_type"``
    key and returns the appropriate typed dataclass.

    Args:
        data: Plain dict containing ``"design_type"`` and all required
              parameter fields for that design.  Typically loaded from YAML
              or JSON (e.g. a ``case_config.yaml`` file).

    Returns:
        The concrete ``BaseUnitCell`` subclass instance.

    Raises:
        UnitCellFactoryError: wrapping any ``ValueError`` raised by
                              ``design_parameters_from_dict()`` (missing keys,
                              unsupported design type, bad values).

    Example::

        cell = create_unit_cell_from_dict({
            "design_type": "tetrachiral",
            "cell_size": 7.85,
            "node_radius": 1.05,
            "ligament_thickness": 1.05,
        })
    """
    try:
        parameters = design_parameters_from_dict(data)
    except ValueError as exc:
        raise UnitCellFactoryError(
            f"Failed to parse design parameters from dict: {exc}.  "
            f"Ensure 'design_type' is present and all required fields are "
            f"provided for the specified design."
        ) from exc

    return create_unit_cell(parameters)


# ---------------------------------------------------------------------------
# CaseDefinition convenience helper
# ---------------------------------------------------------------------------

def create_unit_cell_from_case(case_definition: Any) -> BaseUnitCell:
    """
    Extract design parameters from a case definition object and instantiate
    the correct unit-cell generator.

    Accepts any object that has a ``.design_parameters`` attribute whose
    value is a supported ``DesignParameterSet`` instance.  This includes
    ``CaseDefinition`` from ``workflow/case_schema.py`` but does not require
    importing it here (avoiding cross-layer coupling).

    ARCHITECTURAL DECISION — duck-typed on ``.design_parameters``:
        Importing ``CaseDefinition`` explicitly would couple this factory
        module to the full workflow layer.  Duck-typing on the attribute is
        sufficient, explicit, and avoids circular-import risk if the schema
        module is later restructured.

    Args:
        case_definition: Object with a ``.design_parameters`` attribute
                         (typically a ``CaseDefinition`` instance).

    Returns:
        The concrete ``BaseUnitCell`` subclass instance.

    Raises:
        UnitCellFactoryError: if ``case_definition`` has no
                              ``design_parameters`` attribute, or if the
                              attribute value is not a supported parameter type.

    Example::

        case = CaseDefinition(
            case_id="case_000001",
            design_parameters=ReentrantParameters(...),
            ...
        )
        cell = create_unit_cell_from_case(case)
    """
    if not hasattr(case_definition, "design_parameters"):
        raise UnitCellFactoryError(
            f"create_unit_cell_from_case() expects an object with a "
            f"'.design_parameters' attribute (e.g. CaseDefinition).  "
            f"Got: {type(case_definition).__name__} which has no such attribute."
        )

    return create_unit_cell(case_definition.design_parameters)


# ---------------------------------------------------------------------------
# Supported-design helper functions
# ---------------------------------------------------------------------------

def get_supported_design_types() -> tuple[DesignType, ...]:
    """
    Return a tuple of all ``DesignType`` values that have a registered
    unit-cell class.

    Useful for validation, testing, and sweep-config guards.

    Returns:
        Tuple of ``DesignType`` enum members in registry insertion order.
    """
    return tuple(_REGISTRY.keys())


def is_supported_design_type(design_type: DesignType | str) -> bool:
    """
    Return ``True`` if ``design_type`` is registered in the unit-cell factory.

    Accepts either a ``DesignType`` enum member or a raw string value
    (e.g. ``"reentrant"``).

    Args:
        design_type: ``DesignType`` enum member or its string ``.value``.

    Returns:
        ``True`` if supported, ``False`` otherwise.

    Example::

        is_supported_design_type("tetrachiral")   # True
        is_supported_design_type("half_circle")   # False (v1 out of scope)
    """
    if isinstance(design_type, str):
        try:
            design_type = DesignType(design_type)
        except ValueError:
            return False
    return design_type in _REGISTRY


def get_unit_cell_class(design_type: DesignType) -> type[BaseUnitCell]:
    """
    Return the concrete ``BaseUnitCell`` subclass registered for a given
    ``DesignType``, without instantiating it.

    Useful for introspection, testing, and documentation generation.

    Args:
        design_type: A ``DesignType`` enum member.

    Returns:
        The concrete unit-cell class (not an instance).

    Raises:
        UnitCellFactoryError: if ``design_type`` is not in the registry.

    Example::

        cls = get_unit_cell_class(DesignType.ROTATING_SQUARE)
        # cls is RotatingSquareUnitCell
    """
    try:
        return _REGISTRY[design_type]
    except KeyError:
        raise UnitCellFactoryError(
            f"No unit-cell class is registered for DesignType "
            f"'{design_type.value}'.  "
            f"Supported types: {[dt.value for dt in _REGISTRY]}."
        ) from None
