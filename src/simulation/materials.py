"""
src/simulation/materials.py
=============================
Material library module for the auxetic plate pipeline.

This module loads material definitions from ``config/materials.yaml``,
validates them at a practical level, and exposes typed ``MaterialRecord``
objects for downstream simulation and fatigue-proxy modules.

PIPELINE POSITION:
    config/materials.yaml  →  [THIS MODULE]  →  solver_exporter.py
                                              →  fatigue_model.py

ARCHITECTURAL DECISION — separate rich dataclasses from case_schema.py:
    ``case_schema.py`` defines ``MaterialDefinition`` as the minimal schema
    needed across the pipeline (case definitions, results, reports).
    This module defines richer dataclasses (``MechanicalProperties``,
    ``FatigueProperties``, ``MaterialRecord``) that carry the full YAML
    structure including UTS, placeholder flags, S-N curve availability,
    and simulation defaults.  ``MaterialRecord.to_case_schema_material()``
    bridges to ``MaterialDefinition`` for modules that only need the
    minimal representation.  This separation keeps ``case_schema.py`` lean
    and avoids coupling the schema layer to YAML implementation details.

ARCHITECTURAL DECISION — MaterialLibrary as the canonical runtime object:
    All simulation and fatigue modules should obtain materials through
    ``MaterialLibrary.get(name)`` rather than parsing YAML themselves.
    The library validates all entries on load, so downstream code can
    trust the material objects it receives without re-validating.

ARCHITECTURAL DECISION — fatigue values are placeholder by default:
    ``FatigueProperties.fatigue_limit_is_placeholder`` defaults to ``True``
    and must be explicitly set to ``False`` in YAML only when the fatigue
    value has been replaced with validated, traceable test data.
    The fatigue-proxy module checks this flag and attaches a low-confidence
    label to results when it remains True.

UNITS (consistent with base_config.yaml and case_schema.py):
    Stress / modulus : MPa
    Density          : g/cm³
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from workflow.case_schema import MaterialDefinition

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class MaterialLibraryError(Exception):
    """
    Raised when the material library cannot be loaded or a lookup fails.

    Covers:
      - missing or malformed YAML file
      - missing required fields in a material entry
      - invalid numeric property values
      - requests for undefined material names
    """


# ---------------------------------------------------------------------------
# Typed sub-dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MechanicalProperties:
    """
    Core mechanical properties for simulation assignment.

    All stress/modulus values are in MPa; density is in g/cm³.
    """

    elastic_modulus_mpa: float
    poissons_ratio: float
    density_g_per_cm3: float
    yield_strength_mpa: float
    ultimate_tensile_strength_mpa: float

    def validate(self) -> None:
        """
        Validate numeric ranges for mechanical properties.

        Raises:
            ValueError: with a descriptive message on any violation.
        """
        if self.elastic_modulus_mpa <= 0.0:
            raise ValueError(
                f"elastic_modulus_mpa must be positive, "
                f"got {self.elastic_modulus_mpa}."
            )
        if not (0.0 < self.poissons_ratio < 0.5):
            # Thermodynamic stability: ν ∈ (−1, 0.5); restricted to (0, 0.5)
            # for the titanium alloys in scope.
            raise ValueError(
                f"poissons_ratio must be in (0, 0.5), "
                f"got {self.poissons_ratio}."
            )
        if self.density_g_per_cm3 <= 0.0:
            raise ValueError(
                f"density_g_per_cm3 must be positive, "
                f"got {self.density_g_per_cm3}."
            )
        if self.yield_strength_mpa <= 0.0:
            raise ValueError(
                f"yield_strength_mpa must be positive, "
                f"got {self.yield_strength_mpa}."
            )
        if self.ultimate_tensile_strength_mpa <= 0.0:
            raise ValueError(
                f"ultimate_tensile_strength_mpa must be positive, "
                f"got {self.ultimate_tensile_strength_mpa}."
            )
        if self.ultimate_tensile_strength_mpa < self.yield_strength_mpa:
            # UTS should be ≥ yield strength; flag reversed values as they
            # almost certainly indicate a data-entry error.
            raise ValueError(
                f"ultimate_tensile_strength_mpa ({self.ultimate_tensile_strength_mpa}) "
                f"must be ≥ yield_strength_mpa ({self.yield_strength_mpa}).  "
                f"Check the materials.yaml entry."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "elastic_modulus_mpa":            self.elastic_modulus_mpa,
            "poissons_ratio":                 self.poissons_ratio,
            "density_g_per_cm3":              self.density_g_per_cm3,
            "yield_strength_mpa":             self.yield_strength_mpa,
            "ultimate_tensile_strength_mpa":  self.ultimate_tensile_strength_mpa,
        }


@dataclass
class FatigueProperties:
    """
    Fatigue-related properties for the fatigue-risk proxy model.

    ARCHITECTURAL DECISION — all fatigue values are provisional by default:
        ``fatigue_limit_is_placeholder`` defaults to ``True``.  Downstream
        fatigue-proxy code must check this flag and label results accordingly.
        Only set it to ``False`` when the value has been replaced with
        validated, traceable test data.

    NOTE: these values are used only as proxy reference thresholds.  They
    are NOT validated fatigue design allowables.
    """

    fatigue_limit_mpa: float | None = None       # MPa; proxy reference only
    fatigue_limit_is_placeholder: bool = True    # always True unless replaced
    sn_curve_available: bool = False             # no S-N curve data in version 1
    mean_stress_correction: str | None = None    # e.g. "goodman_placeholder"
    notes: str | None = None

    def validate(self) -> None:
        """Validate fatigue property values where they are defined."""
        if self.fatigue_limit_mpa is not None and self.fatigue_limit_mpa <= 0.0:
            raise ValueError(
                f"fatigue_limit_mpa must be positive if provided, "
                f"got {self.fatigue_limit_mpa}."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fatigue_limit_mpa":             self.fatigue_limit_mpa,
            "fatigue_limit_is_placeholder":  self.fatigue_limit_is_placeholder,
            "sn_curve_available":            self.sn_curve_available,
            "mean_stress_correction":        self.mean_stress_correction,
            "notes":                         self.notes,
        }


@dataclass
class MaterialRecord:
    """
    Full material record loaded from ``config/materials.yaml``.

    This is the rich runtime representation used within the simulation layer.
    Use ``to_case_schema_material()`` to obtain the minimal ``MaterialDefinition``
    expected by ``case_schema.py`` and downstream schema-level code.

    Attributes:
        name:                Material name (must match the YAML mapping key).
        enabled:             Whether the material is active in sweep generation.
        category:            Descriptive category string (e.g. ``titanium_alloy``).
        notes:               Optional human-readable notes.
        mechanical:          ``MechanicalProperties`` instance.
        fatigue:             ``FatigueProperties`` instance.
        simulation_defaults: Raw dict of per-material simulation flags from YAML.
    """

    name: str
    enabled: bool
    category: str
    notes: str | None
    mechanical: MechanicalProperties
    fatigue: FatigueProperties
    simulation_defaults: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """
        Run full validation on this material record.

        Raises:
            MaterialLibraryError: wrapping any ValueError from sub-object
                                  validation, with material name context.
        """
        if not self.name:
            raise MaterialLibraryError("MaterialRecord.name must not be empty.")
        try:
            self.mechanical.validate()
        except ValueError as exc:
            raise MaterialLibraryError(
                f"Material '{self.name}' mechanical properties invalid: {exc}"
            ) from exc
        try:
            self.fatigue.validate()
        except ValueError as exc:
            raise MaterialLibraryError(
                f"Material '{self.name}' fatigue properties invalid: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Conversion to case_schema MaterialDefinition
    # ------------------------------------------------------------------

    def to_case_schema_material(self) -> MaterialDefinition:
        """
        Convert to the minimal ``MaterialDefinition`` dataclass from
        ``workflow/case_schema.py``.

        ARCHITECTURAL DECISION — bridge method, not shared base class:
            ``MaterialDefinition`` is the schema-layer minimal representation.
            ``MaterialRecord`` is the simulation-layer rich representation.
            Bridging via a method keeps each class independent and avoids
            making the schema layer depend on YAML-parsing details.

        Returns:
            ``MaterialDefinition`` with properties mapped from this record.
        """
        return MaterialDefinition(
            name=self.name,
            elastic_modulus_mpa=self.mechanical.elastic_modulus_mpa,
            poissons_ratio=self.mechanical.poissons_ratio,
            density_g_per_cm3=self.mechanical.density_g_per_cm3,
            yield_strength_mpa=self.mechanical.yield_strength_mpa,
            fatigue_limit_mpa=self.fatigue.fatigue_limit_mpa,
            notes=(
                f"{'[PLACEHOLDER FATIGUE] ' if self.fatigue.fatigue_limit_is_placeholder else ''}"
                f"{self.notes or ''}"
            ).strip() or None,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of this material record."""
        return {
            "name":                self.name,
            "enabled":             self.enabled,
            "category":            self.category,
            "notes":               self.notes,
            "mechanical":          self.mechanical.to_dict(),
            "fatigue":             self.fatigue.to_dict(),
            "simulation_defaults": self.simulation_defaults,
        }


# ---------------------------------------------------------------------------
# Material library container
# ---------------------------------------------------------------------------

@dataclass
class MaterialLibrary:
    """
    Runtime container for all material records loaded from the YAML library.

    Provides named lookup, enabled-material filtering, and default-material
    access.  This is the canonical material access object for all simulation
    and fatigue-proxy modules.

    Attributes:
        metadata:         Raw metadata section from the YAML.
        units:            Raw units section from the YAML.
        default_material: Name of the default material for unspecified cases.
        materials:        Mapping of material name → ``MaterialRecord``.
    """

    metadata: dict[str, Any]
    units: dict[str, Any]
    default_material: str
    materials: dict[str, MaterialRecord]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """
        Validate the library: check default material exists, validate each
        record, and warn about disabled materials.

        Raises:
            MaterialLibraryError: if the library is structurally invalid.
        """
        if not self.materials:
            raise MaterialLibraryError(
                "MaterialLibrary contains no material entries.  "
                "Check config/materials.yaml."
            )
        if self.default_material not in self.materials:
            raise MaterialLibraryError(
                f"default_material '{self.default_material}' is not defined "
                f"in the materials mapping.  "
                f"Available materials: {list(self.materials.keys())}."
            )
        for name, record in self.materials.items():
            record.validate()
            if not record.enabled:
                logger.warning(
                    "Material '%s' is disabled (enabled: false) and will be "
                    "excluded from enabled-material queries.",
                    name,
                )

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get(self, name: str) -> MaterialRecord:
        """
        Return the ``MaterialRecord`` for a given material name.

        Args:
            name: Material name as it appears in ``config/materials.yaml``.

        Returns:
            ``MaterialRecord`` instance.

        Raises:
            MaterialLibraryError: if the name is not found in the library.
        """
        if name not in self.materials:
            raise MaterialLibraryError(
                f"Material '{name}' is not defined in the material library.  "
                f"Available: {list(self.materials.keys())}."
            )
        return self.materials[name]

    def get_enabled_materials(self) -> list[MaterialRecord]:
        """
        Return all ``MaterialRecord`` instances with ``enabled = True``.

        Returns:
            List of enabled material records (order matches YAML insertion order).
        """
        return [r for r in self.materials.values() if r.enabled]

    def get_default(self) -> MaterialRecord:
        """
        Return the default material record specified by ``default_material``.

        Returns:
            ``MaterialRecord`` for the configured default material.

        Raises:
            MaterialLibraryError: if the default is not found (shouldn't
                                  happen after ``validate()`` passes).
        """
        return self.get(self.default_material)

    def names(self) -> list[str]:
        """Return the list of all material names (enabled and disabled)."""
        return list(self.materials.keys())

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the full library."""
        return {
            "metadata":         self.metadata,
            "units":            self.units,
            "default_material": self.default_material,
            "materials":        {k: v.to_dict() for k, v in self.materials.items()},
        }


# ---------------------------------------------------------------------------
# YAML parsing helpers
# ---------------------------------------------------------------------------

def _parse_mechanical_properties(data: Mapping[str, Any]) -> MechanicalProperties:
    """
    Parse the ``mechanical`` sub-section of a material YAML entry.

    Args:
        data: Dict corresponding to the ``mechanical:`` key in materials.yaml.

    Returns:
        ``MechanicalProperties`` instance.

    Raises:
        MaterialLibraryError: if required fields are missing or non-numeric.
    """
    required = [
        "elastic_modulus_mpa",
        "poissons_ratio",
        "density_g_per_cm3",
        "yield_strength_mpa",
        "ultimate_tensile_strength_mpa",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise MaterialLibraryError(
            f"Missing required mechanical property field(s): {missing}."
        )
    try:
        return MechanicalProperties(
            elastic_modulus_mpa=float(data["elastic_modulus_mpa"]),
            poissons_ratio=float(data["poissons_ratio"]),
            density_g_per_cm3=float(data["density_g_per_cm3"]),
            yield_strength_mpa=float(data["yield_strength_mpa"]),
            ultimate_tensile_strength_mpa=float(data["ultimate_tensile_strength_mpa"]),
        )
    except (TypeError, ValueError) as exc:
        raise MaterialLibraryError(
            f"Non-numeric value in mechanical properties: {exc}"
        ) from exc


def _parse_fatigue_properties(data: Mapping[str, Any]) -> FatigueProperties:
    """
    Parse the ``fatigue`` sub-section of a material YAML entry.

    All fatigue fields are optional; sensible defaults are applied when
    absent.  ``fatigue_limit_is_placeholder`` defaults to ``True`` unless
    explicitly set to ``false`` in the YAML.

    Args:
        data: Dict corresponding to the ``fatigue:`` key in materials.yaml.

    Returns:
        ``FatigueProperties`` instance.
    """
    fatigue_limit_raw = data.get("fatigue_limit_mpa")
    fatigue_limit: float | None = None
    if fatigue_limit_raw is not None:
        try:
            fatigue_limit = float(fatigue_limit_raw)
        except (TypeError, ValueError) as exc:
            raise MaterialLibraryError(
                f"Non-numeric fatigue_limit_mpa value: {fatigue_limit_raw!r}: {exc}"
            ) from exc

    return FatigueProperties(
        fatigue_limit_mpa=fatigue_limit,
        # ARCHITECTURAL DECISION — default is True; must be explicitly false:
        fatigue_limit_is_placeholder=bool(data.get("fatigue_limit_is_placeholder", True)),
        sn_curve_available=bool(data.get("sn_curve_available", False)),
        mean_stress_correction=data.get("mean_stress_correction"),
        notes=data.get("notes"),
    )


def _parse_material_record(name: str, data: Mapping[str, Any]) -> MaterialRecord:
    """
    Parse one material entry from the ``materials:`` section of materials.yaml.

    Args:
        name: Material name string (the YAML mapping key).
        data: Dict of the material entry.

    Returns:
        ``MaterialRecord`` instance (not yet validated).

    Raises:
        MaterialLibraryError: if required sub-sections are missing.
    """
    if "mechanical" not in data:
        raise MaterialLibraryError(
            f"Material '{name}' is missing the required 'mechanical' sub-section."
        )

    try:
        mechanical = _parse_mechanical_properties(data["mechanical"])
    except MaterialLibraryError as exc:
        raise MaterialLibraryError(
            f"Material '{name}' mechanical parsing failed: {exc}"
        ) from exc

    fatigue_data = data.get("fatigue", {})
    try:
        fatigue = _parse_fatigue_properties(fatigue_data)
    except MaterialLibraryError as exc:
        raise MaterialLibraryError(
            f"Material '{name}' fatigue parsing failed: {exc}"
        ) from exc

    return MaterialRecord(
        name=str(data.get("name", name)),
        enabled=bool(data.get("enabled", True)),
        category=str(data.get("category", "unknown")),
        notes=data.get("notes"),
        mechanical=mechanical,
        fatigue=fatigue,
        simulation_defaults=dict(data.get("simulation_defaults", {})),
    )


def _parse_material_library(raw: dict[str, Any]) -> MaterialLibrary:
    """
    Parse a fully loaded materials YAML dict into a ``MaterialLibrary``.

    Args:
        raw: Top-level dict from ``yaml.safe_load`` of materials.yaml.

    Returns:
        ``MaterialLibrary`` (not yet validated).

    Raises:
        MaterialLibraryError: on structural or parsing errors.
    """
    for required_key in ("metadata", "units", "default_material", "materials"):
        if required_key not in raw:
            raise MaterialLibraryError(
                f"materials.yaml is missing the required top-level key: "
                f"'{required_key}'."
            )

    materials_raw = raw["materials"]
    if not isinstance(materials_raw, dict) or not materials_raw:
        raise MaterialLibraryError(
            "materials.yaml [materials] must be a non-empty mapping."
        )

    records: dict[str, MaterialRecord] = {}
    for mat_name, mat_data in materials_raw.items():
        if not isinstance(mat_data, dict):
            raise MaterialLibraryError(
                f"Material entry '{mat_name}' must be a dict, "
                f"got {type(mat_data).__name__}."
            )
        records[mat_name] = _parse_material_record(mat_name, mat_data)

    return MaterialLibrary(
        metadata=dict(raw.get("metadata", {})),
        units=dict(raw.get("units", {})),
        default_material=str(raw["default_material"]),
        materials=records,
    )


# ---------------------------------------------------------------------------
# Loader functions
# ---------------------------------------------------------------------------

def load_material_library(
    materials_yaml_path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> MaterialLibrary:
    """
    Load, parse, and validate the material library from YAML.

    Resolution order for the YAML file:
      1. ``materials_yaml_path`` if provided directly.
      2. The path configured in ``base_config.yaml [paths.materials_config]``,
         resolved via ``load_pipeline_config(project_root)``.
      3. Fallback: ``<project_root>/config/materials.yaml`` by convention.

    Args:
        materials_yaml_path: Explicit path to materials.yaml (optional).
        project_root:        Project root directory for config resolution
                             (optional; auto-detected if None).

    Returns:
        Validated ``MaterialLibrary`` instance.

    Raises:
        MaterialLibraryError: if the file cannot be found, parsed, or validated.
    """
    resolved_path: Path

    if materials_yaml_path is not None:
        resolved_path = Path(materials_yaml_path).resolve()
    else:
        # Resolve via config loader.
        try:
            from utils.config_loader import load_pipeline_config
            cfg = load_pipeline_config(project_root)
            resolved_path = cfg.resolve_project_path(
                cfg.paths.get("materials_config", "config/materials.yaml")
            )
        except Exception as exc:
            raise MaterialLibraryError(
                f"Could not resolve materials.yaml path via config loader: {exc}.  "
                f"Pass materials_yaml_path explicitly."
            ) from exc

    if not resolved_path.exists():
        raise MaterialLibraryError(
            f"materials.yaml not found at: {resolved_path}"
        )

    logger.info("Loading material library from: %s", resolved_path)

    try:
        with resolved_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise MaterialLibraryError(
            f"Failed to parse materials.yaml: {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise MaterialLibraryError(
            f"materials.yaml must be a YAML mapping at the top level, "
            f"got {type(raw).__name__}."
        )

    library = _parse_material_library(raw)
    library.validate()

    logger.info(
        "Material library loaded: %d material(s), default='%s'.",
        len(library.materials),
        library.default_material,
    )
    return library


def load_material_record(
    name: str,
    materials_yaml_path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> MaterialRecord:
    """
    Load the material library and return the named ``MaterialRecord``.

    A convenience wrapper for callers that need a single material without
    keeping the full library object in scope.

    Args:
        name:                Material name (e.g. ``"Ti-6Al-4V"``).
        materials_yaml_path: Explicit YAML path (optional).
        project_root:        Project root for config resolution (optional).

    Returns:
        ``MaterialRecord`` for the requested material.

    Raises:
        MaterialLibraryError: if the library cannot be loaded or the name
                              is not found.
    """
    library = load_material_library(materials_yaml_path, project_root)
    return library.get(name)


def get_default_material(
    materials_yaml_path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> MaterialRecord:
    """
    Load the material library and return the configured default material.

    Args:
        materials_yaml_path: Explicit YAML path (optional).
        project_root:        Project root for config resolution (optional).

    Returns:
        ``MaterialRecord`` for the default material.

    Raises:
        MaterialLibraryError: on load or lookup failure.
    """
    library = load_material_library(materials_yaml_path, project_root)
    return library.get_default()


# ---------------------------------------------------------------------------
# Compatibility helpers (bridge to case_schema.py)
# ---------------------------------------------------------------------------

def material_record_from_name(
    name: str,
    library: MaterialLibrary,
) -> MaterialRecord:
    """
    Look up a ``MaterialRecord`` by name from an already-loaded library.

    Equivalent to ``library.get(name)``; provided for explicit naming at
    call sites.

    Args:
        name:    Material name.
        library: Already-loaded ``MaterialLibrary``.

    Returns:
        ``MaterialRecord`` instance.

    Raises:
        MaterialLibraryError: if the name is not found.
    """
    return library.get(name)


def material_definition_from_name(
    name: str,
    library: MaterialLibrary,
) -> MaterialDefinition:
    """
    Look up a material by name and return its ``MaterialDefinition``
    (the minimal schema-layer representation from ``case_schema.py``).

    Use this when building ``CaseDefinition`` objects or anywhere that
    expects the schema-layer type rather than the full ``MaterialRecord``.

    Args:
        name:    Material name.
        library: Already-loaded ``MaterialLibrary``.

    Returns:
        ``MaterialDefinition`` instance.

    Raises:
        MaterialLibraryError: if the name is not found.
    """
    return library.get(name).to_case_schema_material()
