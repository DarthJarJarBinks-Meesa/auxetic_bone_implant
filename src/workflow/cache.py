"""
src/workflow/cache.py
======================
Workflow caching module for the auxetic plate pipeline.

This module provides deterministic hashing, cache-key generation, canonical
artifact path resolution, and cache-hit checks for version-1 geometry, mesh,
solver-input, and result artifacts.

ARCHITECTURAL DECISION — filesystem-based caching only:
    Version 1 uses file-existence checks as the cache-hit mechanism.  There
    is no database, no lock file, no TTL, and no eviction policy.  A cached
    artifact is valid if the file exists at the expected path.  Invalidation
    is done by deleting the file or the case run directory.  This is simple,
    auditable, and compatible with both local and HPC workflows.

ARCHITECTURAL DECISION — layered signatures for incremental reuse:
    Signatures are computed at three levels of specificity:
        geometry_signature   = f(design_parameters, plate_thickness, lattice)
        mesh_signature       = f(geometry_signature, meshing_preset)
        solver_signature     = f(mesh_signature, material, load_case)
    Each layer depends on the previous.  A cached mesh can be reused across
    different materials and load cases; a cached geometry can be reused
    across different mesh presets.  This avoids recomputing expensive
    upstream steps when only downstream parameters change.

ARCHITECTURAL DECISION — SHA-256 with 16-char prefix as the default hash:
    16 hex characters (64-bit space) provide negligible collision probability
    for the expected sweep sizes (< 10,000 cases) while remaining readable
    in filenames and logs.  The full SHA-256 digest is available via
    ``CacheSignature.full_signature`` for audit purposes.

ARCHITECTURAL DECISION — per-case run directories keyed by case_id:
    Each case gets a deterministic subdirectory under ``runs/`` named by
    its ``case_id``.  This is predictable, human-navigable, and avoids
    the need to scan for matching hashes at runtime.  Global generated
    assets (unit cells, lattices, solids, meshes) use hash-based filenames
    for content-addressable reuse across cases with identical geometry.

UNITS: not applicable — this module manages paths and hashes only.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from workflow.case_schema import CaseDefinition, DesignParameterSet


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class CacheError(Exception):
    """
    Raised when cache operations encounter unrecoverable problems.

    Covers:
      - malformed case inputs that cannot be serialised for hashing
      - invalid cache base directories
      - JSON serialisation failures in strict modes
    """


# ---------------------------------------------------------------------------
# Typed dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CacheSignature:
    """
    Deterministic identity signature for a cacheable pipeline artifact.

    Attributes:
        full_signature:  Full hex digest (SHA-256 by default; 64 chars).
        short_signature: Truncated digest used in filenames (default 16 chars).
        components:      Human-readable dict of the inputs that produced this
                         signature.  Useful for debugging and audit logging.
    """

    full_signature: str
    short_signature: str
    components: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "full_signature":  self.full_signature,
            "short_signature": self.short_signature,
            "components":      self.components,
        }


@dataclass
class CaseArtifactPaths:
    """
    Canonical filesystem paths for all artifacts belonging to one case run.

    All paths are strings for JSON serialisability.  Callers that need Path
    objects should wrap at the call site: ``Path(paths.mesh_directory)``.
    """

    run_directory: str
    geometry_directory: str
    mesh_directory: str
    solver_directory: str
    results_directory: str
    status_file: str
    metadata_file: str
    case_config_file: str

    def to_dict(self) -> dict[str, str]:
        return {
            "run_directory":       self.run_directory,
            "geometry_directory":  self.geometry_directory,
            "mesh_directory":      self.mesh_directory,
            "solver_directory":    self.solver_directory,
            "results_directory":   self.results_directory,
            "status_file":         self.status_file,
            "metadata_file":       self.metadata_file,
            "case_config_file":    self.case_config_file,
        }


@dataclass
class CacheProbeResult:
    """
    Result of a cache-hit check for one cacheable artifact.

    Attributes:
        cache_key:     Short signature used to identify this artifact.
        artifact_type: Human-readable label (e.g. ``"geometry"``, ``"mesh"``).
        exists:        True if the expected file or directory is present.
        path:          Expected artifact path (whether or not it exists).
        metadata:      Optional supporting context.
    """

    cache_key: str
    artifact_type: str
    exists: bool
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_key":    self.cache_key,
            "artifact_type": self.artifact_type,
            "exists":       self.exists,
            "path":         self.path,
            "metadata":     self.metadata,
        }


# ---------------------------------------------------------------------------
# Stable serialisation helpers
# ---------------------------------------------------------------------------

def _normalize_for_hashing(value: Any) -> Any:
    """
    Recursively normalise a value into a JSON-serialisable, deterministic form.

    Handles:
        - dataclasses with ``to_dict()`` → calls ``to_dict()``
        - Enum members                   → ``.value``
        - Path objects                   → ``str()``
        - sets and tuples                → sorted list
        - dicts                          → sorted-key dict (recursive values)
        - None / primitives              → unchanged

    ARCHITECTURAL DECISION — explicit normalisation, not ``vars()``-based:
        Using ``vars(obj)`` on arbitrary objects is fragile when private
        attributes or non-serialisable types are present.  Checking for
        ``to_dict()`` first ensures we use the canonical serialisation
        already defined on each data model.

    Args:
        value: Any Python value.

    Returns:
        JSON-serialisable form.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _normalize_for_hashing(value.to_dict())
    if isinstance(value, dict):
        return {str(k): _normalize_for_hashing(v) for k, v in sorted(value.items())}
    if isinstance(value, (set, frozenset)):
        return sorted(_normalize_for_hashing(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_normalize_for_hashing(v) for v in value]
    # Fallback for unknown types
    return str(value)


def _stable_json_dumps(value: Any) -> str:
    """
    Produce a stable, deterministic JSON string from any value.

    Applies ``_normalize_for_hashing`` first, then serialises with
    sorted keys and no extraneous whitespace.

    Args:
        value: Any value.

    Returns:
        Compact, sorted-key JSON string.

    Raises:
        CacheError: if serialisation fails.
    """
    try:
        normalised = _normalize_for_hashing(value)
        return json.dumps(normalised, sort_keys=True, separators=(",", ":"))
    except Exception as exc:
        raise CacheError(
            f"Failed to produce stable JSON for hashing: {exc}.  "
            f"Ensure all case-defining objects implement to_dict()."
        ) from exc


def stable_hash(
    value: Any,
    algorithm: str = "sha256",
    length: int = 16,
) -> str:
    """
    Compute a stable, deterministic hex hash of any value.

    ARCHITECTURAL DECISION — SHA-256 with 16-char prefix by default:
        See module docstring.  The full digest is always computable from
        ``full_signature = stable_hash(x, length=64)``.

    Args:
        value:     Value to hash (normalised before hashing).
        algorithm: Hash algorithm name (``"sha256"`` recommended).
        length:    Number of hex characters to return (max 64 for SHA-256).

    Returns:
        Hex string of ``length`` characters.

    Raises:
        CacheError: if normalisation or hashing fails.
    """
    canonical = _stable_json_dumps(value)
    try:
        digest = hashlib.new(algorithm, canonical.encode("utf-8")).hexdigest()
    except Exception as exc:
        raise CacheError(f"Hashing failed with algorithm '{algorithm}': {exc}") from exc
    return "".join(c for i, c in enumerate(digest) if i < length)


# ---------------------------------------------------------------------------
# Signature builders
# ---------------------------------------------------------------------------

def design_parameter_signature(
    parameters: DesignParameterSet,
) -> CacheSignature:
    """
    Build a ``CacheSignature`` for a set of design parameters.

    Components: design type + all independent parameter values.

    Args:
        parameters: Typed design parameter dataclass.

    Returns:
        ``CacheSignature``.
    """
    components = parameters.to_dict()
    # Remove derived values — signature must reflect independent params only
    components.pop("fillet_radius_derived", None)
    full = stable_hash(components, length=64)
    short = "".join(c for i, c in enumerate(full) if i < 16)
    return CacheSignature(
        full_signature=full,
        short_signature=short,
        components={"design_parameters": components},
    )


def geometry_signature(
    case_definition: CaseDefinition,
) -> CacheSignature:
    """
    Build a ``CacheSignature`` for the geometry of a case.

    Captures only the geometry-determining inputs:
        - design parameters (independent fields only)
        - plate_thickness
        - lattice_repeats_x, lattice_repeats_y

    ARCHITECTURAL DECISION — material and load case excluded:
        Material and load case do not affect the 3D solid geometry.  The
        same solid can be reused across multiple simulation scenarios.

    Args:
        case_definition: Full case definition.

    Returns:
        ``CacheSignature`` for the geometry.
    """
    params = case_definition.design_parameters.to_dict()
    params.pop("fillet_radius_derived", None)

    components = {
        "design_parameters":  params,
        "plate_thickness":     case_definition.plate_thickness,
        "lattice_repeats_x":   case_definition.lattice_repeats_x,
        "lattice_repeats_y":   case_definition.lattice_repeats_y,
    }
    full = stable_hash(components, length=64)
    short = "".join(c for i, c in enumerate(full) if i < 16)
    return CacheSignature(
        full_signature=full,
        short_signature=short,
        components=components,
    )


def mesh_signature(
    case_definition: CaseDefinition,
    meshing_preset: str | None = None,
) -> CacheSignature:
    """
    Build a ``CacheSignature`` for a mesh.

    Depends on the geometry signature plus the meshing preset name.

    Args:
        case_definition: Full case definition.
        meshing_preset:  Preset name (e.g. ``"default"``).

    Returns:
        ``CacheSignature`` for the mesh.
    """
    geom_sig = geometry_signature(case_definition)
    components = {
        "geometry_signature": geom_sig.full_signature,
        "meshing_preset":     meshing_preset or "default",
    }
    full = stable_hash(components, length=64)
    short = "".join(c for i, c in enumerate(full) if i < 16)
    return CacheSignature(
        full_signature=full,
        short_signature=short,
        components=components,
    )


def solver_signature(
    case_definition: CaseDefinition,
    meshing_preset: str | None = None,
) -> CacheSignature:
    """
    Build a ``CacheSignature`` for a solver input deck.

    Depends on mesh signature plus material and load case.

    Args:
        case_definition: Full case definition.
        meshing_preset:  Preset name (e.g. ``"default"``).

    Returns:
        ``CacheSignature`` for the solver input.
    """
    mesh_sig = mesh_signature(case_definition, meshing_preset)
    components = {
        "mesh_signature": mesh_sig.full_signature,
        "material_name":  case_definition.material.name,
        "load_case_type": case_definition.load_case.load_case_type.value,
        "load_case_name": case_definition.load_case.name,
    }
    full = stable_hash(components, length=64)
    short = "".join(c for i, c in enumerate(full) if i < 16)
    return CacheSignature(
        full_signature=full,
        short_signature=short,
        components=components,
    )


def case_signature(case_definition: CaseDefinition) -> CacheSignature:
    """
    Build the full ``CacheSignature`` for an entire case (all inputs).

    Includes:
        - design parameters
        - plate thickness
        - lattice repeats
        - material name
        - load case type + name

    Args:
        case_definition: Full case definition.

    Returns:
        ``CacheSignature`` for the complete case.
    """
    # Use solver_signature as the most inclusive layer
    sig = solver_signature(case_definition, meshing_preset=None)
    return CacheSignature(
        full_signature=sig.full_signature,
        short_signature=sig.short_signature,
        components={**sig.components, "case_id": case_definition.case_id},
    )


# ---------------------------------------------------------------------------
# Project root resolution helper
# ---------------------------------------------------------------------------

def _resolve_project_root(project_root: str | Path | None) -> Path:
    """
    Resolve the project root directory.

    Uses the supplied value if provided; otherwise attempts auto-detection
    via ``config_loader``, falling back to ``Path.cwd()``.

    Args:
        project_root: Explicit root path or None.

    Returns:
        Resolved ``Path``.
    """
    if project_root is not None:
        return Path(project_root).resolve()
    try:
        from utils.config_loader import _find_project_root
        return _find_project_root(Path(__file__).parent)
    except Exception:
        return Path.cwd()


# ---------------------------------------------------------------------------
# Run-directory and artifact path helpers
# ---------------------------------------------------------------------------

def case_run_directory(
    case_definition: CaseDefinition,
    project_root: str | Path | None = None,
) -> Path:
    """
    Return the canonical run directory for a case.

    Path: ``<project_root>/runs/<case_id>/``

    Args:
        case_definition: Full case definition.
        project_root:    Project root (auto-detected if None).

    Returns:
        ``Path`` to the case run directory (may not yet exist).
    """
    root = _resolve_project_root(project_root)
    return root / "runs" / case_definition.case_id


def build_case_artifact_paths(
    case_definition: CaseDefinition,
    project_root: str | Path | None = None,
) -> CaseArtifactPaths:
    """
    Construct canonical per-case artifact paths.

    Path layout::

        runs/<case_id>/
        ├── case_config.yaml
        ├── metadata.json
        ├── status.txt
        ├── geometry/
        ├── mesh/
        ├── solver/
        └── results/

    ARCHITECTURAL DECISION — paths derived from case_id only:
        Using ``case_id`` as the sole directory discriminator keeps the
        path predictable and stable.  The case_id is set by the case
        generator and never changes for a given case.

    Args:
        case_definition: Full case definition.
        project_root:    Project root (auto-detected if None).

    Returns:
        ``CaseArtifactPaths`` instance.
    """
    run_dir = case_run_directory(case_definition, project_root)
    return CaseArtifactPaths(
        run_directory=str(run_dir),
        geometry_directory=str(run_dir / "geometry"),
        mesh_directory=str(run_dir / "mesh"),
        solver_directory=str(run_dir / "solver"),
        results_directory=str(run_dir / "results"),
        status_file=str(run_dir / "status.txt"),
        metadata_file=str(run_dir / "metadata.json"),
        case_config_file=str(run_dir / "case_config.yaml"),
    )


def ensure_case_directories(
    case_definition: CaseDefinition,
    project_root: str | Path | None = None,
) -> CaseArtifactPaths:
    """
    Create the per-case artifact directory tree if it does not yet exist.

    Args:
        case_definition: Full case definition.
        project_root:    Project root (auto-detected if None).

    Returns:
        ``CaseArtifactPaths`` after ensuring all directories exist.
    """
    paths = build_case_artifact_paths(case_definition, project_root)
    for dir_path_str in (
        paths.geometry_directory,
        paths.mesh_directory,
        paths.solver_directory,
        paths.results_directory,
        str(Path(paths.solver_directory) / "logs"),
    ):
        Path(dir_path_str).mkdir(parents=True, exist_ok=True)
    return paths


# ---------------------------------------------------------------------------
# Generated-cache path helpers
# ---------------------------------------------------------------------------

def _generated_base(project_root: str | Path | None, sub: str) -> Path:
    """Return ``<project_root>/data/generated/<sub>/``."""
    return _resolve_project_root(project_root) / "data" / "generated" / sub


def default_geometry_filename(
    signature: CacheSignature,
    suffix: str = ".step",
) -> str:
    """Canonical filename for a generated geometry file."""
    components = signature.components
    design = components.get("design_parameters", {}).get("design_type", "unknown")
    return f"{design}_{signature.short_signature}{suffix}"


def default_mesh_filename(
    signature: CacheSignature,
    suffix: str = ".msh",
) -> str:
    """Canonical filename for a generated mesh file."""
    return f"mesh_{signature.short_signature}{suffix}"


def generated_unit_cell_path(
    signature: CacheSignature,
    project_root: str | Path | None = None,
    suffix: str = ".step",
) -> Path:
    """
    Path for a cached 2D unit-cell geometry file.

    Location: ``data/generated/unit_cells_2d/<design>_<hash>.step``
    """
    return _generated_base(project_root, "unit_cells_2d") / default_geometry_filename(signature, suffix)


def generated_lattice_path(
    signature: CacheSignature,
    project_root: str | Path | None = None,
    suffix: str = ".step",
) -> Path:
    """
    Path for a cached 2D lattice geometry file.

    Location: ``data/generated/lattices_2d/<design>_<hash>.step``
    """
    return _generated_base(project_root, "lattices_2d") / default_geometry_filename(signature, suffix)


def generated_solid_path(
    signature: CacheSignature,
    project_root: str | Path | None = None,
    suffix: str = ".step",
) -> Path:
    """
    Path for a cached 3D solid geometry file.

    Location: ``data/generated/solids_3d/<design>_<hash>.step``
    """
    return _generated_base(project_root, "solids_3d") / default_geometry_filename(signature, suffix)


def generated_mesh_path(
    signature: CacheSignature,
    project_root: str | Path | None = None,
    suffix: str = ".msh",
) -> Path:
    """
    Path for a cached mesh file.

    Location: ``data/generated/meshes/mesh_<hash>.msh``
    """
    return _generated_base(project_root, "meshes") / default_mesh_filename(signature, suffix)


# ---------------------------------------------------------------------------
# Cache-hit probe helpers
# ---------------------------------------------------------------------------

def probe_geometry_cache(
    case_definition: CaseDefinition,
    project_root: str | Path | None = None,
) -> CacheProbeResult:
    """
    Check whether a cached 3D solid geometry exists for this case.

    Probes the global ``data/generated/solids_3d/`` directory using the
    geometry signature (design params + plate thickness + lattice).

    Args:
        case_definition: Full case definition.
        project_root:    Project root (auto-detected if None).

    Returns:
        ``CacheProbeResult`` indicating whether the geometry is cached.
    """
    sig = geometry_signature(case_definition)
    path = generated_solid_path(sig, project_root)
    return CacheProbeResult(
        cache_key=sig.short_signature,
        artifact_type="geometry",
        exists=path.exists(),
        path=str(path),
        metadata={"signature_components": sig.components},
    )


def probe_mesh_cache(
    case_definition: CaseDefinition,
    meshing_preset: str | None = None,
    project_root: str | Path | None = None,
) -> CacheProbeResult:
    """
    Check whether a cached mesh exists for this case.

    Probes the global ``data/generated/meshes/`` directory using the
    mesh signature (geometry + preset).

    Args:
        case_definition: Full case definition.
        meshing_preset:  Mesh preset name.
        project_root:    Project root (auto-detected if None).

    Returns:
        ``CacheProbeResult``.
    """
    sig = mesh_signature(case_definition, meshing_preset)
    path = generated_mesh_path(sig, project_root)
    return CacheProbeResult(
        cache_key=sig.short_signature,
        artifact_type="mesh",
        exists=path.exists(),
        path=str(path),
        metadata={"signature_components": sig.components},
    )


def probe_solver_input_cache(
    case_definition: CaseDefinition,
    meshing_preset: str | None = None,
    project_root: str | Path | None = None,
) -> CacheProbeResult:
    """
    Check whether a solver input deck exists for this case.

    Probes the case's per-run solver directory for ``input.inp``.

    Args:
        case_definition: Full case definition.
        meshing_preset:  Mesh preset name (used for signature).
        project_root:    Project root (auto-detected if None).

    Returns:
        ``CacheProbeResult``.
    """
    sig = solver_signature(case_definition, meshing_preset)
    paths = build_case_artifact_paths(case_definition, project_root)
    inp_path = Path(paths.solver_directory) / "input.inp"
    return CacheProbeResult(
        cache_key=sig.short_signature,
        artifact_type="solver_input",
        exists=inp_path.exists(),
        path=str(inp_path),
        metadata={"signature_components": sig.components},
    )


def probe_postprocess_cache(
    case_definition: CaseDefinition,
    project_root: str | Path | None = None,
) -> CacheProbeResult:
    """
    Check whether postprocessed result metrics exist for this case.

    Probes the case's per-run results directory for
    ``extracted_metrics.json``.

    Args:
        case_definition: Full case definition.
        project_root:    Project root (auto-detected if None).

    Returns:
        ``CacheProbeResult``.
    """
    sig = case_signature(case_definition)
    paths = build_case_artifact_paths(case_definition, project_root)
    metrics_path = Path(paths.results_directory) / "extracted_metrics.json"
    return CacheProbeResult(
        cache_key=sig.short_signature,
        artifact_type="postprocess_results",
        exists=metrics_path.exists(),
        path=str(metrics_path),
        metadata={"signature_components": sig.components},
    )


# ---------------------------------------------------------------------------
# Reuse-decision helpers
# ---------------------------------------------------------------------------

def should_reuse_geometry(
    case_definition: CaseDefinition,
    project_root: str | Path | None = None,
) -> bool:
    """
    Return True if a cached 3D solid geometry exists and can be reused.

    Args:
        case_definition: Full case definition.
        project_root:    Project root (auto-detected if None).

    Returns:
        True if the geometry cache file exists.
    """
    return probe_geometry_cache(case_definition, project_root).exists


def should_reuse_mesh(
    case_definition: CaseDefinition,
    meshing_preset: str | None = None,
    project_root: str | Path | None = None,
) -> bool:
    """
    Return True if a cached mesh exists and can be reused.

    Args:
        case_definition: Full case definition.
        meshing_preset:  Mesh preset name.
        project_root:    Project root (auto-detected if None).

    Returns:
        True if the mesh cache file exists.
    """
    return probe_mesh_cache(case_definition, meshing_preset, project_root).exists


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def write_cache_metadata(
    path: str | Path,
    payload: Mapping[str, Any],
) -> None:
    """
    Write a JSON metadata sidecar file alongside a cached artifact.

    Args:
        path:    Destination ``.json`` file path.
        payload: JSON-serialisable dict.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(dict(payload), fh, indent=2, default=str)


def read_cache_metadata(path: str | Path) -> dict[str, Any] | None:
    """
    Read a JSON metadata sidecar file.

    Args:
        path: ``.json`` metadata file path.

    Returns:
        Parsed dict, or ``None`` if the file does not exist or is unreadable.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None
