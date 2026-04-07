"""
src/simulation/loadcases.py
=============================
Load-case library module for the auxetic plate pipeline.

This module loads load-case definitions from ``config/loadcases.yaml``,
validates them at a practical level, and exposes typed ``LoadCaseRecord``
objects for downstream solver export, workflow orchestration, and
fatigue-proxy modules.

PIPELINE POSITION:
    config/loadcases.yaml  →  [THIS MODULE]  →  solver_exporter.py
                                              →  fatigue_model.py
                                              →  workflow/orchestrator.py

ARCHITECTURAL DECISION — separate rich dataclasses from case_schema.py:
    ``case_schema.py`` defines ``LoadCaseDefinition`` as the minimal schema
    needed across the pipeline (case definitions, results, reports).
    This module defines the richer ``LoadCaseRecord`` that carries the full
    YAML structure: boundary condition specs, cyclic parameters, metadata
    extras, and enabled/disabled state.
    ``LoadCaseRecord.to_case_schema_loadcase()`` bridges to
    ``LoadCaseDefinition`` for modules that only need the minimal form.

ARCHITECTURAL DECISION — boundary conditions as descriptive labels only:
    ``BoundaryConditionSpec`` stores human-readable BC descriptors from the
    YAML (``boundary_condition_label``, ``loading_direction``, etc.).  It
    does NOT contain CalculiX *BOUNDARY or *CLOAD syntax.  Translation to
    solver syntax is the responsibility of ``solver_exporter.py``.  This
    keeps the YAML and this module solver-agnostic.

ARCHITECTURAL DECISION — cyclic proxy is flagged, not a separate type:
    The CYCLIC load case type uses ``mean_force_n`` / ``amplitude_force_n``
    rather than a single ``force_n``.  Version 1 does NOT run a time-
    stepping FE simulation for cyclic cases; instead, the fatigue-risk proxy
    module uses the stress field from a companion static case combined with
    the cyclic amplitudes.  The ``proxy_only`` field in ``metadata`` carries
    this flag explicitly so downstream code can detect it without parsing
    the description string.

ARCHITECTURAL DECISION — future placeholders skipped silently:
    The ``future_placeholders`` section in loadcases.yaml contains disabled
    entries for four-point bending, torsion, etc.  These are stored as a
    raw dict and not parsed into ``LoadCaseRecord`` objects.  The loader
    warns if a future placeholder has ``enabled: true`` (which should not
    happen in version 1) but does not abort.

UNITS (consistent with base_config.yaml and case_schema.py):
    Force     : N
    Frequency : Hz
    Cycles    : count (dimensionless)
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from workflow.case_schema import LoadCaseDefinition, LoadCaseType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class LoadCaseLibraryError(Exception):
    """
    Raised when the load-case library cannot be loaded or a lookup fails.

    Covers:
      - missing or malformed YAML file
      - unrecognised ``load_case_type`` string
      - missing required fields in a load-case entry
      - invalid numeric property values
      - requests for undefined load-case keys
    """


# ---------------------------------------------------------------------------
# Typed sub-dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BoundaryConditionSpec:
    """
    Descriptive boundary condition specification for a load case.

    Contains human-readable labels interpreted by ``solver_exporter.py``
    when building CalculiX input decks.  Does NOT contain solver syntax.

    Attributes:
        label:                        Primary BC label for solver dispatch.
        loading_direction:            Direction of applied force (e.g. ``"negative_x"``).
        fixed_region:                 Mesh region label for zero-displacement BC.
        load_application_region:      Mesh region label for load application.
        support_span_fraction_of_length: Outer support span as fraction of plate length (bending).
        support_left_fraction:        Left support position as fraction of length.
        support_right_fraction:       Right support position as fraction of length.
        load_at_midspan:              Whether load is applied at the midpoint (bending).
        notes:                        Optional human-readable notes.
    """

    label: str
    loading_direction: str | None = None
    fixed_region: str | None = None
    load_application_region: str | None = None
    support_span_fraction_of_length: float | None = None
    support_left_fraction: float | None = None
    support_right_fraction: float | None = None
    load_at_midspan: bool | None = None
    notes: str | None = None

    def validate(self) -> None:
        """
        Validate boundary condition spec fields.

        Raises:
            ValueError: if the support span fraction is outside (0, 1].
        """
        if not self.label:
            raise ValueError(
                "BoundaryConditionSpec.label must not be empty."
            )
        supportspan = self.support_span_fraction_of_length
        if supportspan is not None:
            if not (0.0 < supportspan <= 1.0):
                raise ValueError(
                    f"support_span_fraction_of_length must be in (0, 1], "
                    f"got {supportspan}."
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label":                           self.label,
            "loading_direction":               self.loading_direction,
            "fixed_region":                    self.fixed_region,
            "load_application_region":         self.load_application_region,
            "support_span_fraction_of_length": self.support_span_fraction_of_length,
            "support_left_fraction":           self.support_left_fraction,
            "support_right_fraction":          self.support_right_fraction,
            "load_at_midspan":                 self.load_at_midspan,
            "notes":                           self.notes,
        }


@dataclass
class LoadCaseRecord:
    """
    Full load-case record loaded from ``config/loadcases.yaml``.

    This is the rich runtime representation used within the simulation layer.
    Use ``to_case_schema_loadcase()`` to obtain the minimal
    ``LoadCaseDefinition`` expected by ``case_schema.py``.

    For CYCLIC cases:
        ``mean_force_n`` and ``amplitude_force_n`` are used instead of
        ``force_n``.  Both represent force magnitudes in N.  The fatigue
        proxy module reads these to compute stress amplitude.

    Attributes:
        key:               YAML mapping key (e.g. ``"axial_compression"``).
        enabled:           Whether this case is active in sweep generation.
        load_case_type:    ``LoadCaseType`` enum member.
        name:              Human-readable display name.
        description:       Optional verbose description.
        force_n:           Applied force magnitude [N] (static cases).
        mean_force_n:      Mean force magnitude [N] (cyclic cases).
        amplitude_force_n: Cyclic force amplitude [N] (cyclic cases).
        frequency_hz:      Cyclic loading frequency [Hz].
        cycle_count:       Reference cycle count (proxy normalisation only).
        boundary_conditions: Parsed BC spec (optional).
        metadata:          Open dict for load-case extras from YAML.
    """

    key: str
    enabled: bool
    load_case_type: LoadCaseType
    name: str
    description: str | None
    force_n: float | None = None
    mean_force_n: float | None = None
    amplitude_force_n: float | None = None
    frequency_hz: float | None = None
    cycle_count: int | None = None
    boundary_conditions: BoundaryConditionSpec | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """
        Validate this load-case record.

        Checks are type-specific:
          - AXIAL_COMPRESSION / AXIAL_TENSION / BENDING: require positive force_n.
          - CYCLIC: require positive mean_force_n and/or amplitude_force_n.
          - frequency_hz positive if set.
          - cycle_count positive if set.
          - boundary_conditions validated if present.

        Raises:
            LoadCaseLibraryError: wrapping any ValueError, with the case key
                                  as context.
        """
        try:
            self._validate_impl()
        except ValueError as exc:
            raise LoadCaseLibraryError(
                f"Load case '{self.key}' validation failed: {exc}"
            ) from exc

    def _validate_impl(self) -> None:
        """Inner validation logic (raises ValueError, not LoadCaseLibraryError)."""
        if not self.key:
            raise ValueError("key must not be empty.")
        if not self.name:
            raise ValueError("name must not be empty.")

        # Static force cases
        if self.load_case_type in (
            LoadCaseType.AXIAL_COMPRESSION,
            LoadCaseType.AXIAL_TENSION,
            LoadCaseType.BENDING,
        ):
            force = self.force_n
            if force is not None and force <= 0.0:
                raise ValueError(
                    f"force_n must be positive for {self.load_case_type.value} "
                    f"case, got {force} N."
                )

        # Cyclic proxy case
        if self.load_case_type == LoadCaseType.CYCLIC:
            mean_f = self.mean_force_n
            if mean_f is not None and mean_f <= 0.0:
                raise ValueError(
                    f"mean_force_n must be positive for cyclic proxy case, "
                    f"got {mean_f} N."
                )
            amp_f = self.amplitude_force_n
            if amp_f is not None and amp_f <= 0.0:
                raise ValueError(
                    f"amplitude_force_n must be positive for cyclic proxy case, "
                    f"got {amp_f} N."
                )

        # Shared optional fields
        freq = self.frequency_hz
        if freq is not None and freq <= 0.0:
            raise ValueError(
                f"frequency_hz must be positive if provided, "
                f"got {freq} Hz."
            )
        cycles = self.cycle_count
        if cycles is not None and cycles <= 0:
            raise ValueError(
                f"cycle_count must be a positive integer if provided, "
                f"got {cycles}."
            )

        # BC spec validation
        bc = self.boundary_conditions
        if bc is not None:
            bc.validate()

    # ------------------------------------------------------------------
    # Conversion to case_schema LoadCaseDefinition
    # ------------------------------------------------------------------

    def to_case_schema_loadcase(self) -> LoadCaseDefinition:
        """
        Convert to the minimal ``LoadCaseDefinition`` from ``case_schema.py``.

        ARCHITECTURAL DECISION — bridge method:
            ``LoadCaseDefinition`` is the schema-layer minimal representation.
            Extra fields (BC spec details, cyclic parameters, proxy flags)
            are stored in the ``metadata`` dict so downstream modules that
            only consume the schema type can still access them.

        The ``force_n`` field is set from:
          - ``force_n`` for static cases.
          - ``mean_force_n`` for CYCLIC cases (representative magnitude).

        Returns:
            ``LoadCaseDefinition`` instance.
        """
        representative_force = self.force_n
        if self.load_case_type == LoadCaseType.CYCLIC:
            representative_force = self.mean_force_n

        # Collect extra fields into metadata for the schema representation.
        schema_meta: dict[str, Any] = dict(self.metadata)
        if self.mean_force_n is not None:
            schema_meta["mean_force_n"] = self.mean_force_n
        if self.amplitude_force_n is not None:
            schema_meta["amplitude_force_n"] = self.amplitude_force_n
        if self.frequency_hz is not None:
            schema_meta["frequency_hz"] = self.frequency_hz
        if self.cycle_count is not None:
            schema_meta["cycle_count"] = self.cycle_count
            
        bc = self.boundary_conditions
        if bc is not None:
            schema_meta["boundary_condition_spec"] = bc.to_dict()

        return LoadCaseDefinition(
            load_case_type=self.load_case_type,
            name=self.name,
            force_n=representative_force,
            description=self.description,
            boundary_condition_label=(
                bc.label if bc is not None else None
            ),
            metadata=schema_meta,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of this load-case record."""
        bc = self.boundary_conditions
        return {
            "key":               self.key,
            "enabled":           self.enabled,
            "load_case_type":    self.load_case_type.value,
            "name":              self.name,
            "description":       self.description,
            "force_n":           self.force_n,
            "mean_force_n":      self.mean_force_n,
            "amplitude_force_n": self.amplitude_force_n,
            "frequency_hz":      self.frequency_hz,
            "cycle_count":       self.cycle_count,
            "boundary_conditions": (
                bc.to_dict() if bc is not None else None
            ),
            "metadata":          self.metadata,
        }


# ---------------------------------------------------------------------------
# Load-case library container
# ---------------------------------------------------------------------------

@dataclass
class LoadCaseLibrary:
    """
    Runtime container for all load-case records loaded from the YAML library.

    Provides named lookup, enabled-case filtering, and default-list access.
    This is the canonical load-case access object for simulation and
    fatigue-proxy modules.

    Attributes:
        metadata:            Raw metadata section from the YAML.
        units:               Raw units section from the YAML.
        defaults:            Raw defaults section (default forces, cycle counts, etc.).
        loadcases:           Mapping of case key → ``LoadCaseRecord``.
        future_placeholders: Raw dict of disabled future load-case stubs.
    """

    metadata: dict[str, Any]
    units: dict[str, Any]
    defaults: dict[str, Any]
    loadcases: dict[str, LoadCaseRecord]
    future_placeholders: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """
        Validate the library: validate each load-case record and warn about
        disabled entries.

        Raises:
            LoadCaseLibraryError: if the library has no entries or a record
                                  fails validation.
        """
        if not self.loadcases:
            raise LoadCaseLibraryError(
                "LoadCaseLibrary contains no load-case entries.  "
                "Check config/loadcases.yaml."
            )
        for key, record in self.loadcases.items():
            record.validate()
            if not record.enabled:
                logger.warning(
                    "Load case '%s' is disabled (enabled: false) and will be "
                    "excluded from enabled-loadcase queries.",
                    key,
                )

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get(self, name: str) -> LoadCaseRecord:
        """
        Return the ``LoadCaseRecord`` for a given load-case key.

        Args:
            name: Load-case key as defined in loadcases.yaml (e.g.
                  ``"axial_compression"``).

        Returns:
            ``LoadCaseRecord`` instance.

        Raises:
            LoadCaseLibraryError: if the key is not found.
        """
        if name not in self.loadcases:
            raise LoadCaseLibraryError(
                f"Load case '{name}' is not defined in the load-case library.  "
                f"Available: {list(self.loadcases.keys())}."
            )
        return self.loadcases[name]

    def get_enabled_loadcases(self) -> list[LoadCaseRecord]:
        """
        Return all ``LoadCaseRecord`` instances with ``enabled = True``.

        Returns:
            List of enabled records (order matches YAML insertion order).
        """
        return [r for r in self.loadcases.values() if r.enabled]

    def get_default_enabled_keys(self) -> list[str]:
        """
        Return the list of default enabled load-case keys from the
        ``defaults.default_enabled_loadcases`` section of the YAML.

        Falls back to the keys of all enabled records if the section is absent.

        Returns:
            List of load-case key strings.
        """
        default_keys = self.defaults.get("default_enabled_loadcases")
        if isinstance(default_keys, list):
            return [str(k) for k in default_keys]
        # Fallback: return all enabled keys
        return [r.key for r in self.get_enabled_loadcases()]

    def names(self) -> list[str]:
        """Return the list of all load-case keys (enabled and disabled)."""
        return list(self.loadcases.keys())

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the full library."""
        return {
            "metadata":            self.metadata,
            "units":               self.units,
            "defaults":            self.defaults,
            "loadcases":           {k: v.to_dict() for k, v in self.loadcases.items()},
            "future_placeholders": self.future_placeholders,
        }


# ---------------------------------------------------------------------------
# YAML parsing helpers
# ---------------------------------------------------------------------------

def _parse_load_case_type(value: str) -> LoadCaseType:
    """
    Convert a load-case type string from YAML to a ``LoadCaseType`` enum.

    Args:
        value: String value from YAML (e.g. ``"axial_compression"``).

    Returns:
        ``LoadCaseType`` enum member.

    Raises:
        LoadCaseLibraryError: if the string is not a valid ``LoadCaseType``.
    """
    try:
        return LoadCaseType(value)
    except ValueError:
        supported = [lct.value for lct in LoadCaseType]
        raise LoadCaseLibraryError(
            f"Unrecognised load_case_type '{value}'.  "
            f"Supported values: {supported}."
        ) from None


def _parse_boundary_conditions(
    data: Mapping[str, Any] | None,
    fallback_label: str | None = None,
) -> BoundaryConditionSpec | None:
    """
    Parse the boundary condition section of a load-case YAML entry.

    Accepts either a nested ``metadata:`` dict (as structured in
    loadcases.yaml) or ``None``.

    ARCHITECTURAL DECISION — BC fields may be flat in ``metadata`` or
    embedded in a sub-section:
        The loadcases.yaml stores BC helper fields inside ``metadata:``
        (e.g. ``load_application_region``, ``support_span_fraction_of_length``).
        The top-level ``boundary_condition_label`` is a separate field on the
        load-case entry.  This parser accepts either structure and normalises
        into ``BoundaryConditionSpec``.

    Args:
        data:           The ``metadata`` dict from the YAML load-case entry,
                        or ``None`` if absent.
        fallback_label: The ``boundary_condition_label`` from the YAML root
                        entry, used if ``data`` does not contain a label.

    Returns:
        ``BoundaryConditionSpec`` or ``None`` if no BC information is available.
    """
    if not data and not fallback_label:
        return None

    data = data or {}
    label = str(data.get("label", fallback_label or "unlabeled"))

    support_span_raw = data.get("support_span_fraction_of_length")
    support_span: float | None = None
    if support_span_raw is not None:
        try:
            support_span = float(support_span_raw)
        except (TypeError, ValueError):
            support_span = None

    support_left_raw = data.get("support_left_fraction")
    support_left: float | None = None
    if support_left_raw is not None:
        try:
            support_left = float(support_left_raw)
        except (TypeError, ValueError):
            support_left = None

    support_right_raw = data.get("support_right_fraction")
    support_right: float | None = None
    if support_right_raw is not None:
        try:
            support_right = float(support_right_raw)
        except (TypeError, ValueError):
            support_right = None

    load_at_midspan_raw = data.get("load_at_midspan")
    load_at_midspan: bool | None = None
    if load_at_midspan_raw is not None:
        load_at_midspan = bool(load_at_midspan_raw)

    return BoundaryConditionSpec(
        label=label,
        loading_direction=data.get("loading_direction"),
        fixed_region=data.get("fixed_region"),
        load_application_region=data.get("load_application_region"),
        support_span_fraction_of_length=support_span,
        support_left_fraction=support_left,
        support_right_fraction=support_right,
        load_at_midspan=load_at_midspan,
        notes=data.get("version_1_note") or data.get("notes"),
    )


def _safe_float(value: Any, field_name: str) -> float | None:
    """Parse a float from YAML or return None; raise on non-numeric values."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise LoadCaseLibraryError(
            f"Non-numeric value for '{field_name}': {value!r}: {exc}"
        ) from exc


def _safe_int(value: Any, field_name: str) -> int | None:
    """Parse an int from YAML or return None; raise on non-numeric values."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise LoadCaseLibraryError(
            f"Non-integer value for '{field_name}': {value!r}: {exc}"
        ) from exc


def _parse_loadcase_record(key: str, data: Mapping[str, Any]) -> LoadCaseRecord:
    """
    Parse one load-case entry from the ``loadcases:`` section of loadcases.yaml.

    Args:
        key:  YAML mapping key (e.g. ``"axial_compression"``).
        data: Dict of the load-case entry.

    Returns:
        ``LoadCaseRecord`` instance (not yet validated).

    Raises:
        LoadCaseLibraryError: if required fields are missing or malformed.
    """
    for required in ("load_case_type", "name"):
        if required not in data:
            raise LoadCaseLibraryError(
                f"Load case '{key}' is missing required field '{required}'."
            )

    load_case_type = _parse_load_case_type(str(data["load_case_type"]))

    # Parse forces — use top-level field names matching the YAML schema
    force_n = _safe_float(data.get("force_n"), "force_n")
    mean_force_n = _safe_float(data.get("mean_force_n"), "mean_force_n")
    amplitude_force_n = _safe_float(data.get("amplitude_force_n"), "amplitude_force_n")
    frequency_hz = _safe_float(data.get("frequency_hz"), "frequency_hz")
    cycle_count = _safe_int(data.get("cycle_count"), "cycle_count")

    # Parse boundary conditions.
    # loadcases.yaml stores BC helper info in the ``metadata`` sub-dict.
    # The top-level ``boundary_condition_label`` is the primary label.
    bc_label = data.get("boundary_condition_label")
    raw_metadata = dict(data.get("metadata", {}))
    bc_spec = _parse_boundary_conditions(raw_metadata, fallback_label=bc_label)

    # Add loading_direction to BC spec if at the top level
    if bc_spec is not None and bc_spec.loading_direction is None:
        bc_spec.loading_direction = data.get("loading_direction")

    return LoadCaseRecord(
        key=key,
        enabled=bool(data.get("enabled", True)),
        load_case_type=load_case_type,
        name=str(data["name"]),
        description=data.get("description"),
        force_n=force_n,
        mean_force_n=mean_force_n,
        amplitude_force_n=amplitude_force_n,
        frequency_hz=frequency_hz,
        cycle_count=cycle_count,
        boundary_conditions=bc_spec,
        metadata=raw_metadata,
    )


def _parse_loadcase_library(raw: dict[str, Any]) -> LoadCaseLibrary:
    """
    Parse a fully loaded loadcases YAML dict into a ``LoadCaseLibrary``.

    Args:
        raw: Top-level dict from ``yaml.safe_load`` of loadcases.yaml.

    Returns:
        ``LoadCaseLibrary`` (not yet validated).

    Raises:
        LoadCaseLibraryError: on structural or parsing errors.
    """
    for required_key in ("metadata", "units", "defaults", "loadcases"):
        if required_key not in raw:
            raise LoadCaseLibraryError(
                f"loadcases.yaml is missing the required top-level key: "
                f"'{required_key}'."
            )

    loadcases_raw = raw["loadcases"]
    if not isinstance(loadcases_raw, dict) or not loadcases_raw:
        raise LoadCaseLibraryError(
            "loadcases.yaml [loadcases] must be a non-empty mapping."
        )

    records: dict[str, LoadCaseRecord] = {}
    for lc_key, lc_data in loadcases_raw.items():
        if not isinstance(lc_data, dict):
            raise LoadCaseLibraryError(
                f"Load case entry '{lc_key}' must be a dict, "
                f"got {type(lc_data).__name__}."
            )
        records[lc_key] = _parse_loadcase_record(lc_key, lc_data)

    # Warn if any future_placeholder has accidentally been enabled
    future_raw = dict(raw.get("future_placeholders", {}))
    for placeholder_key, placeholder_data in future_raw.items():
        if isinstance(placeholder_data, dict) and placeholder_data.get("enabled", False):
            logger.warning(
                "Future placeholder load case '%s' has enabled: true in loadcases.yaml.  "
                "This is not implemented in version 1.  Set enabled: false.",
                placeholder_key,
            )

    return LoadCaseLibrary(
        metadata=dict(raw.get("metadata", {})),
        units=dict(raw.get("units", {})),
        defaults=dict(raw.get("defaults", {})),
        loadcases=records,
        future_placeholders=future_raw,
    )


# ---------------------------------------------------------------------------
# Loader functions
# ---------------------------------------------------------------------------

def load_loadcase_library(
    loadcases_yaml_path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> LoadCaseLibrary:
    """
    Load, parse, and validate the load-case library from YAML.

    Resolution order for the YAML file:
      1. ``loadcases_yaml_path`` if provided directly.
      2. The path configured in ``base_config.yaml [paths.loadcases_config]``,
         resolved via ``load_pipeline_config(project_root)``.
      3. Fallback convention: ``<project_root>/config/loadcases.yaml``.

    Args:
        loadcases_yaml_path: Explicit path to loadcases.yaml (optional).
        project_root:        Project root directory for config resolution
                             (optional; auto-detected if None).

    Returns:
        Validated ``LoadCaseLibrary`` instance.

    Raises:
        LoadCaseLibraryError: if the file cannot be found, parsed, or validated.
    """
    resolved_path: Path

    if loadcases_yaml_path is not None:
        resolved_path = Path(loadcases_yaml_path).resolve()
    else:
        try:
            from utils.config_loader import load_pipeline_config
            cfg = load_pipeline_config(project_root)
            resolved_path = cfg.resolve_project_path(
                cfg.paths.get("loadcases_config", "config/loadcases.yaml")
            )
        except Exception as exc:
            raise LoadCaseLibraryError(
                f"Could not resolve loadcases.yaml path via config loader: {exc}.  "
                f"Pass loadcases_yaml_path explicitly."
            ) from exc

    if not resolved_path.exists():
        raise LoadCaseLibraryError(
            f"loadcases.yaml not found at: {resolved_path}"
        )

    logger.info("Loading load-case library from: %s", resolved_path)

    try:
        with resolved_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise LoadCaseLibraryError(
            f"Failed to parse loadcases.yaml: {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise LoadCaseLibraryError(
            f"loadcases.yaml must be a YAML mapping at the top level, "
            f"got {type(raw).__name__}."
        )

    library = _parse_loadcase_library(raw)
    library.validate()

    logger.info(
        "Load-case library loaded: %d case(s), %d enabled.",
        len(library.loadcases),
        len(library.get_enabled_loadcases()),
    )
    return library


def load_loadcase_record(
    name: str,
    loadcases_yaml_path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> LoadCaseRecord:
    """
    Load the library and return a single named ``LoadCaseRecord``.

    Args:
        name:                Load-case key (e.g. ``"axial_compression"``).
        loadcases_yaml_path: Explicit YAML path (optional).
        project_root:        Project root for config resolution (optional).

    Returns:
        ``LoadCaseRecord`` for the requested load case.

    Raises:
        LoadCaseLibraryError: if the library cannot be loaded or the key
                              is not found.
    """
    library = load_loadcase_library(loadcases_yaml_path, project_root)
    return library.get(name)


def get_enabled_loadcases(
    loadcases_yaml_path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> list[LoadCaseRecord]:
    """
    Load the library and return all enabled ``LoadCaseRecord`` instances.

    Args:
        loadcases_yaml_path: Explicit YAML path (optional).
        project_root:        Project root for config resolution (optional).

    Returns:
        List of enabled ``LoadCaseRecord`` objects.

    Raises:
        LoadCaseLibraryError: on load failure.
    """
    library = load_loadcase_library(loadcases_yaml_path, project_root)
    return library.get_enabled_loadcases()


# ---------------------------------------------------------------------------
# Compatibility helpers (bridge to case_schema.py)
# ---------------------------------------------------------------------------

def loadcase_record_from_name(
    name: str,
    library: LoadCaseLibrary,
) -> LoadCaseRecord:
    """
    Look up a ``LoadCaseRecord`` by key from an already-loaded library.

    Equivalent to ``library.get(name)``; provided for explicit naming.

    Args:
        name:    Load-case key.
        library: Already-loaded ``LoadCaseLibrary``.

    Returns:
        ``LoadCaseRecord`` instance.

    Raises:
        LoadCaseLibraryError: if the key is not found.
    """
    return library.get(name)


def loadcase_definition_from_name(
    name: str,
    library: LoadCaseLibrary,
) -> LoadCaseDefinition:
    """
    Look up a load case by key and return its ``LoadCaseDefinition``
    (the minimal schema-layer representation from ``case_schema.py``).

    Use this when building ``CaseDefinition`` objects or anywhere that
    expects the schema-layer type rather than the full ``LoadCaseRecord``.

    Args:
        name:    Load-case key.
        library: Already-loaded ``LoadCaseLibrary``.

    Returns:
        ``LoadCaseDefinition`` instance.

    Raises:
        LoadCaseLibraryError: if the key is not found.
    """
    return library.get(name).to_case_schema_loadcase()
