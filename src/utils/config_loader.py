"""
src/utils/config_loader.py
===========================
YAML configuration loader for the auxetic plate pipeline.

This module is the single configuration entry point for the entire pipeline.
It loads, structurally validates, and exposes all five config files through
a single ``PipelineConfig`` dataclass.

Config files loaded (relative to project root):
    config/base_config.yaml
    config/materials.yaml
    config/loadcases.yaml
    config/sweep_config.yaml
    config/meshing.yaml

ARCHITECTURAL DECISION — one loader, five files:
    All config access in the pipeline should go through ``PipelineConfig``
    rather than each module loading its own YAML.  This centralises path
    resolution, structural validation, and config access in one place and
    prevents modules from silently ignoring missing sections or inventing
    fallback values.

ARCHITECTURAL DECISION — standard-library only (+ PyYAML):
    No omegaconf, dynaconf, pydantic, or other config framework.  PyYAML
    is the only external dependency.  The loader itself is intentionally
    thin; business-logic validation of individual numeric ranges is the
    responsibility of the modules that consume the values (e.g.
    MaterialDefinition.validate() in case_schema.py).

ARCHITECTURAL DECISION — ``pathlib.Path`` used consistently:
    All path handling uses Path objects internally.  Only ``to_dict()``
    converts them back to strings for JSON serialisation.

ARCHITECTURAL DECISION — project root auto-detection:
    If no project_root is supplied, the loader walks upward from this
    file's location looking for a ``config/`` directory.  This makes the
    loader usable from any sub-directory without requiring callers to know
    the absolute project path.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Version-1 supported designs — kept here so loader validation does not need
# to import from case_schema.py (which would couple utils to workflow).
_SUPPORTED_DESIGNS: frozenset[str] = frozenset(
    {"reentrant", "rotating_square", "tetrachiral"}
)

# Required top-level section keys per config file.
_REQUIRED_BASE_CONFIG_SECTIONS: tuple[str, ...] = (
    "project",
    "paths",
    "units",
    "geometry_defaults",
    "lattice_defaults",
    "extrusion_defaults",
    "reference_geometry",
    "execution",
    "solver",
    "fatigue_proxy",
    "reporting",
    "ranking_defaults",
    "validation_defaults",
)

_REQUIRED_MATERIALS_SECTIONS: tuple[str, ...] = (
    "metadata",
    "units",
    "default_material",
    "materials",
)

_REQUIRED_LOADCASES_SECTIONS: tuple[str, ...] = (
    "metadata",
    "units",
    "defaults",
    "loadcases",
)

_REQUIRED_SWEEP_SECTIONS: tuple[str, ...] = (
    "metadata",
    "strategy",
    "global_defaults",
    "materials",
    "loadcases",
    "plate_thickness",
    "design_sweeps",
    "staged_execution",
    "filters",
)

_REQUIRED_MESHING_SECTIONS: tuple[str, ...] = (
    "metadata",
    "backend",
    "global_defaults",
    "presets",
    "feature_refinement",
    "quality_controls",
    "export",
    "staged_meshing",
)

# Config filenames — only the names, not full paths.
_CONFIG_FILENAMES: dict[str, str] = {
    "base":      "base_config.yaml",
    "materials": "materials.yaml",
    "loadcases": "loadcases.yaml",
    "sweep":     "sweep_config.yaml",
    "meshing":   "meshing.yaml",
}


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class ConfigError(Exception):
    """
    Raised when configuration loading or structural validation fails.

    Covers:
      - missing config files
      - malformed or unreadable YAML
      - missing required top-level sections
      - violated architecture-level assumptions (e.g. unsupported design type)
    """


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _read_yaml_file(path: Path) -> dict[str, Any]:
    """
    Read and parse a YAML file into a plain Python dict.

    Args:
        path: Absolute or relative path to the YAML file.

    Returns:
        Parsed dict.  Returns an empty dict if the file contains no content.

    Raises:
        ConfigError: if the file does not exist or cannot be parsed.
    """
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    if not path.is_file():
        raise ConfigError(f"Config path is not a file: {path}")

    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(
            f"Failed to parse YAML file {path}: {exc}"
        ) from exc

    if data is None:
        logger.warning("Config file %s is empty; treating as empty dict.", path)
        return {}

    if not isinstance(data, dict):
        raise ConfigError(
            f"Expected a YAML mapping at the top level of {path}, "
            f"got {type(data).__name__}."
        )

    return data


def _require_keys(
    data: Mapping[str, Any],
    required: Sequence[str],
    context: str,
) -> None:
    """
    Assert that all required keys are present in a mapping.

    Args:
        data:     The dict / mapping to check.
        required: Sequence of key names that must be present.
        context:  Human-readable label for error messages (e.g. filename).

    Raises:
        ConfigError: listing all missing keys at once.
    """
    missing = [k for k in required if k not in data]
    if missing:
        raise ConfigError(
            f"Config validation failed for {context}. "
            f"Missing required sections/keys: {missing}"
        )


def _resolve_path(base_dir: Path, value: str | Path) -> Path:
    """
    Resolve a path value that may be absolute or relative.

    If ``value`` is already absolute, return it unchanged.
    If it is relative, resolve it relative to ``base_dir``.

    Args:
        base_dir: Anchor directory for relative resolution
                  (usually the project root).
        value:    Path string or Path object to resolve.

    Returns:
        Resolved absolute Path.
    """
    p = Path(value)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _find_project_root(start: Path) -> Path:
    """
    Walk upward from ``start`` to find the project root.

    The project root is identified as the lowest ancestor directory that
    contains a ``config/`` sub-directory.

    ARCHITECTURAL DECISION — walk upward until ``config/`` is found:
        This allows the loader to be called from any sub-package within
        ``src/`` without requiring the caller to know the absolute root.
        If no suitable directory is found within a reasonable depth (20
        levels), a ConfigError is raised rather than silently using a
        wrong directory.

    Args:
        start: Directory from which to begin the upward search.

    Returns:
        Path to the project root.

    Raises:
        ConfigError: if no project root is found.
    """
    current = start.resolve()
    for _ in range(20):  # safety cap on traversal depth
        if (current / "config").is_dir():
            return current
        parent = current.parent
        if parent == current:
            # Reached filesystem root without finding config/
            break
        current = parent

    raise ConfigError(
        f"Could not locate project root (directory containing 'config/') "
        f"by walking upward from: {start}. "
        f"Pass project_root explicitly to load_pipeline_config()."
    )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge ``override`` into ``base``.

    Values in ``override`` take precedence.  Nested dicts are merged
    recursively; all other types are replaced outright.

    ARCHITECTURAL DECISION — simple recursive merge, not full OmegaConf:
        This is intentionally lightweight.  It handles the common case of
        merging a user override dict into a base config.  It does not
        support list merging, interpolation, or type coercion.

    Args:
        base:     Base dictionary (lower priority).
        override: Override dictionary (higher priority).

    Returns:
        A new dict representing the merged result.
    """
    result = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _paths_to_strings(obj: Any) -> Any:
    """
    Recursively convert all ``pathlib.Path`` objects in a nested structure
    to plain strings.

    Used by ``PipelineConfig.to_dict()`` to produce a JSON-serialisable
    representation.
    """
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _paths_to_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_paths_to_strings(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Structural validation helpers
# ---------------------------------------------------------------------------

def _validate_base_config(data: dict[str, Any]) -> None:
    """Validate the structural content of base_config.yaml."""
    _require_keys(data, _REQUIRED_BASE_CONFIG_SECTIONS, "base_config.yaml")

    # Validate solver tool stack names
    solver_section = data.get("solver", {})
    _require_keys(
        solver_section,
        ["geometry_engine", "mesher", "solver_backend"],
        "base_config.yaml [solver]",
    )

    expected_tools = {
        "geometry_engine": "cadquery",
        "mesher": "gmsh",
        "solver_backend": "calculix",
    }
    for key, expected in expected_tools.items():
        actual = str(solver_section.get(key, "")).lower()
        if actual != expected:
            raise ConfigError(
                f"base_config.yaml [solver.{key}] must be '{expected}', "
                f"got '{actual}'.  Update the config to match the selected "
                f"tool stack."
            )


def _validate_materials_config(data: dict[str, Any]) -> None:
    """Validate the structural content of materials.yaml."""
    _require_keys(data, _REQUIRED_MATERIALS_SECTIONS, "materials.yaml")

    materials = data.get("materials", {})
    if not isinstance(materials, dict) or len(materials) == 0:
        raise ConfigError(
            "materials.yaml [materials] must be a non-empty mapping of "
            "material name to property dict."
        )


def _validate_loadcases_config(data: dict[str, Any]) -> None:
    """Validate the structural content of loadcases.yaml."""
    _require_keys(data, _REQUIRED_LOADCASES_SECTIONS, "loadcases.yaml")

    loadcases = data.get("loadcases", {})
    if not isinstance(loadcases, dict) or len(loadcases) == 0:
        raise ConfigError(
            "loadcases.yaml [loadcases] must be a non-empty mapping of "
            "load case key to definition dict."
        )


def _validate_sweep_config(data: dict[str, Any]) -> None:
    """
    Validate the structural content of sweep_config.yaml and enforce
    version-1 architecture constraints.
    """
    _require_keys(data, _REQUIRED_SWEEP_SECTIONS, "sweep_config.yaml")

    # Validate enabled designs are within the supported set.
    global_defaults = data.get("global_defaults", {})
    enabled = global_defaults.get("enabled_designs", [])
    unsupported = [d for d in enabled if d not in _SUPPORTED_DESIGNS]
    if unsupported:
        raise ConfigError(
            f"sweep_config.yaml [global_defaults.enabled_designs] contains "
            f"unsupported design(s): {unsupported}.  "
            f"Version-1 supported designs: {sorted(_SUPPORTED_DESIGNS)}."
        )

    # Validate that density is not swept as a design parameter.
    design_sweeps = data.get("design_sweeps", {})
    for design_name, design_data in design_sweeps.items():
        if not isinstance(design_data, dict):
            continue
        sweep_params = design_data.get("sweep_parameters", {})
        if isinstance(sweep_params, dict) and "density" in sweep_params:
            raise ConfigError(
                f"sweep_config.yaml [design_sweeps.{design_name}.sweep_parameters] "
                f"contains 'density'.  Density is not a geometric sweep parameter "
                f"in version 1 and must be removed."
            )

    # Validate that tetrachiral does not independently sweep fillet_radius.
    tetrachiral_sweeps = (
        design_sweeps.get("tetrachiral", {})
        .get("sweep_parameters", {})
    )
    if isinstance(tetrachiral_sweeps, dict) and "fillet_radius" in tetrachiral_sweeps:
        raise ConfigError(
            "sweep_config.yaml [design_sweeps.tetrachiral.sweep_parameters] "
            "contains 'fillet_radius'.  fillet_radius is a derived parameter "
            "(= 0.25 * node_radius / 1.05) and must not be swept independently."
        )


def _validate_meshing_config(data: dict[str, Any]) -> None:
    """Validate the structural content of meshing.yaml."""
    _require_keys(data, _REQUIRED_MESHING_SECTIONS, "meshing.yaml")

    # Validate that at least the three standard presets are defined.
    presets = data.get("presets", {})
    required_presets = ["coarse", "default", "refined"]
    missing = [p for p in required_presets if p not in presets]
    if missing:
        raise ConfigError(
            f"meshing.yaml [presets] is missing required preset(s): {missing}.  "
            f"Expected at minimum: {required_presets}."
        )


# ---------------------------------------------------------------------------
# PipelineConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    Loaded and validated configuration bundle for the auxetic plate pipeline.

    Holds the raw parsed dicts from all five config files alongside the
    resolved project root and config directory.  Provides convenience
    accessor methods that downstream modules should prefer over direct
    dict access, so that key-path changes in YAML only require updates here.

    ARCHITECTURAL DECISION — raw dicts, not sub-dataclasses:
        Storing raw dicts rather than typed sub-configs keeps this class
        lean and decoupled.  Typed validation of individual values is the
        responsibility of the consuming modules (e.g. MaterialDefinition,
        LoadCaseDefinition).  This loader performs only structural checks.

    ARCHITECTURAL DECISION — convenience methods, not property explosion:
        Frequently accessed paths and nested values are exposed as explicit
        methods with clear names.  Modules should call
        ``cfg.get_material_definition("Ti-6Al-4V")`` rather than
        ``cfg.materials_config["materials"]["Ti-6Al-4V"]`` everywhere.
        This makes YAML key renames a single-file fix.
    """

    project_root: Path
    config_dir: Path
    base_config: dict[str, Any]
    materials_config: dict[str, Any]
    loadcases_config: dict[str, Any]
    sweep_config: dict[str, Any]
    meshing_config: dict[str, Any]

    # ------------------------------------------------------------------
    # Top-level section accessors
    # ------------------------------------------------------------------

    @property
    def project(self) -> dict[str, Any]:
        """Return the ``project`` section of base_config."""
        return self.base_config.get("project", {})

    @property
    def paths(self) -> dict[str, Any]:
        """Return the ``paths`` section of base_config."""
        return self.base_config.get("paths", {})

    @property
    def units(self) -> dict[str, Any]:
        """Return the ``units`` section of base_config."""
        return self.base_config.get("units", {})

    @property
    def geometry_defaults(self) -> dict[str, Any]:
        """Return the ``geometry_defaults`` section of base_config."""
        return self.base_config.get("geometry_defaults", {})

    @property
    def lattice_defaults(self) -> dict[str, Any]:
        """Return the ``lattice_defaults`` section of base_config."""
        return self.base_config.get("lattice_defaults", {})

    @property
    def extrusion_defaults(self) -> dict[str, Any]:
        """Return the ``extrusion_defaults`` section of base_config."""
        return self.base_config.get("extrusion_defaults", {})

    @property
    def solver(self) -> dict[str, Any]:
        """Return the ``solver`` section of base_config."""
        return self.base_config.get("solver", {})

    @property
    def reporting(self) -> dict[str, Any]:
        """Return the ``reporting`` section of base_config."""
        return self.base_config.get("reporting", {})

    @property
    def fatigue_proxy(self) -> dict[str, Any]:
        """Return the ``fatigue_proxy`` section of base_config."""
        return self.base_config.get("fatigue_proxy", {})

    @property
    def execution(self) -> dict[str, Any]:
        """Return the ``execution`` section of base_config."""
        return self.base_config.get("execution", {})

    @property
    def ranking_defaults(self) -> dict[str, Any]:
        """Return the ``ranking_defaults`` section of base_config."""
        return self.base_config.get("ranking_defaults", {})

    @property
    def validation_defaults(self) -> dict[str, Any]:
        """Return the ``validation_defaults`` section of base_config."""
        return self.base_config.get("validation_defaults", {})

    # ------------------------------------------------------------------
    # Design helpers
    # ------------------------------------------------------------------

    def get_enabled_designs(self) -> list[str]:
        """
        Return the list of designs enabled for sweep generation.

        Source: ``sweep_config.global_defaults.enabled_designs``
        """
        return list(
            self.sweep_config
            .get("global_defaults", {})
            .get("enabled_designs", [])
        )

    def get_design_baseline(self, design_name: str) -> dict[str, Any]:
        """
        Return the baseline parameter dict for a specific design.

        Source: ``base_config.geometry_defaults.<design_name>_baseline``

        Args:
            design_name: One of ``reentrant``, ``rotating_square``,
                         ``tetrachiral``.

        Returns:
            Baseline parameter dict.

        Raises:
            ConfigError: if the design or its baseline is not found.
        """
        key = f"{design_name}_baseline"
        baseline = self.geometry_defaults.get(key)
        if baseline is None:
            raise ConfigError(
                f"No baseline found in base_config.yaml "
                f"[geometry_defaults.{key}] for design '{design_name}'.  "
                f"Expected key: '{key}'."
            )
        return dict(baseline)

    def get_design_sweep_config(self, design_name: str) -> dict[str, Any]:
        """
        Return the full sweep configuration block for a specific design.

        Source: ``sweep_config.design_sweeps.<design_name>``

        Args:
            design_name: One of ``reentrant``, ``rotating_square``,
                         ``tetrachiral``.

        Returns:
            Design sweep dict (includes ``baseline``, ``sweep_parameters``,
            ``first_pass_parameters``, ``notes``).

        Raises:
            ConfigError: if the design is not found in design_sweeps.
        """
        sweeps = self.sweep_config.get("design_sweeps", {})
        if design_name not in sweeps:
            raise ConfigError(
                f"Design '{design_name}' not found in "
                f"sweep_config.yaml [design_sweeps].  "
                f"Available: {list(sweeps.keys())}."
            )
        return dict(sweeps[design_name])

    def get_reference_step_path(self, design_name: str) -> Path | None:
        """
        Return the resolved reference STEP file path for a design.

        Returns ``None`` if no path is configured or the file does not exist.
        A warning is logged if the file is configured but not found on disk.

        Source: ``base_config.reference_geometry.files.<design_name>.path``

        NOTE: Reference STEP files are for dimensional reference only and are
        not used as the parametric geometry engine.
        """
        ref_geom = self.base_config.get("reference_geometry", {})
        files = ref_geom.get("files", {})
        entry = files.get(design_name)
        if entry is None:
            return None

        raw_path = entry.get("path")
        if raw_path is None:
            return None

        resolved = _resolve_path(self.project_root, raw_path)
        if not resolved.exists():
            logger.warning(
                "Reference STEP file configured for design '%s' does not "
                "exist on disk: %s",
                design_name,
                resolved,
            )
        return resolved

    # ------------------------------------------------------------------
    # Material helpers
    # ------------------------------------------------------------------

    def get_enabled_materials(self) -> list[str]:
        """
        Return the list of material names enabled for sweep generation.

        Source: ``sweep_config.materials.enabled_materials``
        """
        return list(
            self.sweep_config
            .get("materials", {})
            .get("enabled_materials", [])
        )

    def get_material_definition(self, name: str) -> dict[str, Any]:
        """
        Return the raw material property dict for a given material name.

        Source: ``materials_config.materials.<name>``

        Args:
            name: Exact material name (e.g. ``Ti-6Al-4V``).

        Returns:
            Material definition dict.

        Raises:
            ConfigError: if the material is not found.
        """
        materials = self.materials_config.get("materials", {})
        if name not in materials:
            raise ConfigError(
                f"Material '{name}' not found in materials.yaml.  "
                f"Available: {list(materials.keys())}."
            )
        return dict(materials[name])

    def get_default_material_name(self) -> str:
        """
        Return the default material name from materials.yaml.

        Source: ``materials_config.default_material``
        """
        return str(self.materials_config.get("default_material", ""))

    # ------------------------------------------------------------------
    # Load case helpers
    # ------------------------------------------------------------------

    def get_enabled_loadcases(self) -> list[str]:
        """
        Return the list of load case keys enabled for sweep generation.

        Source: ``sweep_config.loadcases.enabled_loadcases``
        """
        return list(
            self.sweep_config
            .get("loadcases", {})
            .get("enabled_loadcases", [])
        )

    def get_loadcase_definition(self, name: str) -> dict[str, Any]:
        """
        Return the raw load case definition dict for a given key.

        Source: ``loadcases_config.loadcases.<name>``

        Args:
            name: Load case key (e.g. ``axial_compression``).

        Returns:
            Load case definition dict.

        Raises:
            ConfigError: if the load case key is not found.
        """
        loadcases = self.loadcases_config.get("loadcases", {})
        if name not in loadcases:
            raise ConfigError(
                f"Load case '{name}' not found in loadcases.yaml.  "
                f"Available: {list(loadcases.keys())}."
            )
        return dict(loadcases[name])

    # ------------------------------------------------------------------
    # Meshing helpers
    # ------------------------------------------------------------------

    def get_meshing_preset(self, name: str) -> dict[str, Any]:
        """
        Return the meshing preset dict for a given preset name.

        Source: ``meshing_config.presets.<name>``

        Args:
            name: Preset name; one of ``coarse``, ``default``, ``refined``.

        Returns:
            Preset configuration dict.

        Raises:
            ConfigError: if the preset is not found.
        """
        presets = self.meshing_config.get("presets", {})
        if name not in presets:
            raise ConfigError(
                f"Meshing preset '{name}' not found in meshing.yaml.  "
                f"Available: {list(presets.keys())}."
            )
        return dict(presets[name])

    def get_meshing_global_defaults(self) -> dict[str, Any]:
        """
        Return the global defaults section from meshing.yaml.

        Source: ``meshing_config.global_defaults``
        """
        return dict(self.meshing_config.get("global_defaults", {}))

    # ------------------------------------------------------------------
    # Path resolution helpers
    # ------------------------------------------------------------------

    def resolve_project_path(self, relative_or_absolute: str | Path) -> Path:
        """
        Resolve a path that may be absolute or relative to the project root.

        Args:
            relative_or_absolute: Path string or Path object.

        Returns:
            Resolved absolute Path.
        """
        return _resolve_path(self.project_root, relative_or_absolute)

    def get_runs_directory(self) -> Path:
        """Return the resolved runs output directory."""
        return _resolve_path(
            self.project_root,
            self.paths.get("runs_dir", "runs"),
        )

    def get_reports_directory(self) -> Path:
        """Return the resolved reports output directory."""
        return _resolve_path(
            self.project_root,
            self.paths.get("reports_dir", "reports"),
        )

    # ------------------------------------------------------------------
    # Plate thickness helpers
    # ------------------------------------------------------------------

    def get_plate_thickness_values(self, pass_type: str = "full") -> list[float]:
        """
        Return the plate thickness values for a given pass type.

        Args:
            pass_type: ``"full"`` for all sweep values, ``"first_pass"``
                       for the reduced screening set.

        Returns:
            List of plate thickness floats (mm).

        Raises:
            ConfigError: if an unrecognised pass_type is provided.
        """
        pt_section = self.sweep_config.get("plate_thickness", {})
        if pass_type == "full":
            values = pt_section.get("values_mm", [])
        elif pass_type == "first_pass":
            values = pt_section.get("first_pass_values_mm", [])
        else:
            raise ConfigError(
                f"Unknown pass_type '{pass_type}' for get_plate_thickness_values.  "
                f"Use 'full' or 'first_pass'."
            )
        return [float(v) for v in values]

    def get_default_plate_thickness(self) -> float:
        """
        Return the default plate thickness (mm) from extrusion_defaults.

        Source: ``base_config.extrusion_defaults.default_plate_thickness_mm``
        """
        return float(
            self.extrusion_defaults.get("default_plate_thickness_mm", 2.5)
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Return a JSON-serialisable representation of the full config bundle.

        All ``pathlib.Path`` objects are converted to plain strings.

        Useful for writing a resolved config snapshot alongside case outputs
        for traceability.
        """
        return _paths_to_strings(
            {
                "project_root": self.project_root,
                "config_dir": self.config_dir,
                "base_config": self.base_config,
                "materials_config": self.materials_config,
                "loadcases_config": self.loadcases_config,
                "sweep_config": self.sweep_config,
                "meshing_config": self.meshing_config,
            }
        )

    def as_flat_lookup(self) -> dict[str, Any]:
        """
        Return a shallow flat lookup mapping top-level section names to their
        raw dicts across all loaded config files.

        Useful for debugging and for config-dump tooling.  Key collisions
        (same section name in multiple files) are unlikely given the distinct
        section names used across the five config files, but later keys win
        if they occur.

        Returns:
            Flat dict of ``{section_name: section_dict, ...}``
        """
        flat: dict[str, Any] = {}
        for cfg in (
            self.base_config,
            self.materials_config,
            self.loadcases_config,
            self.sweep_config,
            self.meshing_config,
        ):
            flat.update(cfg)
        return flat


# ---------------------------------------------------------------------------
# Factory / loader functions
# ---------------------------------------------------------------------------

def load_pipeline_config(
    project_root: str | Path | None = None,
) -> PipelineConfig:
    """
    Load, validate, and return the full ``PipelineConfig`` for the pipeline.

    This is the primary entry point for all pipeline modules that need
    configuration access.

    ARCHITECTURAL DECISION — project_root auto-detection:
        If ``project_root`` is None, the function walks upward from this
        source file's location to find the directory containing ``config/``.
        This makes the loader callable from any sub-directory without
        requiring callers to pass an absolute path.

    ARCHITECTURAL DECISION — companion config paths resolved from base_config:
        The ``paths`` section of base_config.yaml contains relative paths to
        each companion config.  These are used as the primary path source.
        If a path key is missing from ``paths``, a convention-based fallback
        (``config/<filename>``) is used so the loader remains non-fragile
        during config development.

    Args:
        project_root: Absolute path to the project root directory.
                      If None, auto-detected from the source file location.

    Returns:
        Fully loaded and validated ``PipelineConfig``.

    Raises:
        ConfigError: if any config file is missing, malformed, or fails
                     structural validation.
    """
    # --- Resolve project root ---
    if project_root is None:
        root = _find_project_root(Path(__file__).parent)
    else:
        root = Path(project_root).resolve()
        if not root.is_dir():
            raise ConfigError(
                f"Provided project_root does not exist or is not a directory: {root}"
            )

    config_dir = root / "config"
    if not config_dir.is_dir():
        raise ConfigError(
            f"Config directory not found: {config_dir}.  "
            f"Ensure the project_root contains a 'config/' sub-directory."
        )

    logger.info("Loading pipeline config from project root: %s", root)

    # --- Load base config first ---
    base_path = config_dir / _CONFIG_FILENAMES["base"]
    base_cfg = _read_yaml_file(base_path)
    _validate_base_config(base_cfg)

    # --- Resolve companion config paths ---
    # Prefer paths declared in base_config.yaml [paths]; fall back to
    # the expected convention-based filenames in the config directory.
    declared_paths = base_cfg.get("paths", {})

    def _companion_path(key: str, fallback_filename: str) -> Path:
        declared = declared_paths.get(key)
        if declared:
            return _resolve_path(root, declared)
        return config_dir / fallback_filename

    materials_path  = _companion_path("materials_config",  _CONFIG_FILENAMES["materials"])
    loadcases_path  = _companion_path("loadcases_config",  _CONFIG_FILENAMES["loadcases"])
    sweep_path      = _companion_path("sweep_config",      _CONFIG_FILENAMES["sweep"])
    meshing_path    = _companion_path("meshing_config",    _CONFIG_FILENAMES["meshing"])

    # --- Load and validate companion configs ---
    materials_cfg = _read_yaml_file(materials_path)
    _validate_materials_config(materials_cfg)

    loadcases_cfg = _read_yaml_file(loadcases_path)
    _validate_loadcases_config(loadcases_cfg)

    sweep_cfg = _read_yaml_file(sweep_path)
    _validate_sweep_config(sweep_cfg)

    meshing_cfg = _read_yaml_file(meshing_path)
    _validate_meshing_config(meshing_cfg)

    logger.info(
        "Pipeline config loaded successfully.  "
        "Designs enabled: %s.  Materials enabled: %s.",
        sweep_cfg.get("global_defaults", {}).get("enabled_designs", []),
        sweep_cfg.get("materials", {}).get("enabled_materials", []),
    )

    return PipelineConfig(
        project_root=root,
        config_dir=config_dir,
        base_config=base_cfg,
        materials_config=materials_cfg,
        loadcases_config=loadcases_cfg,
        sweep_config=sweep_cfg,
        meshing_config=meshing_cfg,
    )


# ARCHITECTURAL DECISION — load_config_bundle as an alias:
#   Some callers may prefer a name that emphasises they are loading the
#   full bundle rather than a single file.  Both names are supported and
#   point to the same implementation.  If behaviour diverges in a future
#   version, the alias can be replaced with an independent implementation.
load_config_bundle = load_pipeline_config
