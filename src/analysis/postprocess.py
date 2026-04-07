"""
src/analysis/postprocess.py
=============================
Solver postprocessing module for the auxetic plate pipeline.

This module discovers CalculiX output artifacts, extracts structural result
scalars where possible, and returns structured ``PostprocessResult`` objects
for downstream metrics computation, fatigue-risk proxy, ranking, and reporting.

PIPELINE POSITION:
    solver/  (*.frd, *.dat, logs)  →  [THIS MODULE]  →  metrics.py
                                                      →  fatigue_model.py
                                                      →  ranking / reporting

ARCHITECTURAL DECISION — honest partial results, not fake full parsing:
    Full binary ``.frd`` parsing requires either a dedicated reader library
    or a significant OCC/Abaqus-format parser implementation.  Neither is
    in scope for version 1.  Instead, this module:
      (a) discovers available artifact files;
      (b) reads parseable text-based outputs (``.dat``, stdout logs);
      (c) extracts scalar results with simple, clearly-documented regexes;
      (d) returns ``None`` for values it could not extract, with explicit
          warnings — never inventing results.
    ``.frd`` files are detected and their presence recorded in metadata,
    but binary parsing is a future-version extension.

ARCHITECTURAL DECISION — ``.dat`` file as primary text result source:
    CalculiX writes summary quantities (max/min node results) to the
    ``.dat`` file when ``*NODE FILE`` / ``*EL FILE`` output requests are
    present.  This file is human-readable and the most practical source
    of scalar extraction in version 1.  stdout logs are a secondary fallback.

ARCHITECTURAL DECISION — regex parsers are heuristic and version-1 scoped:
    The ``.dat`` format is not fully standardised across CalculiX versions.
    The regexes below target the most common summary output patterns.  If a
    pattern does not match, the value is left as ``None`` with a warning.
    Future versions should add format-version-aware parsers or use a proper
    CalculiX result reader.

UNITS: mm, N, MPa throughout (consistent with the project-wide convention
and the solver input deck unit system).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from workflow.case_schema import CaseResult, CaseStatus, MetricSet

logger = logging.getLogger(__name__)

# Recognized CalculiX result file suffixes
_CCX_SUFFIXES: tuple[str, ...] = (".frd", ".dat", ".sta", ".cvg", ".rout")
_LOG_PATTERNS: tuple[str, ...] = ("*_stdout.log", "*_stderr.log", "*.log")


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class PostprocessingError(Exception):
    """
    Raised only for hard, unrecoverable postprocessing failures.

    Most postprocessing shortcomings (missing files, unparseable content)
    are recorded in ``PostprocessResult.warnings`` rather than raised.
    """


# ---------------------------------------------------------------------------
# Typed dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PostprocessArtifacts:
    """
    Registry of discovered solver output artifact file paths.

    Paths are stored as strings (None if not found) for JSON serialisability.

    Attributes:
        input_deck_path:  The ``.inp`` file that was run.
        stdout_log_path:  Captured solver stdout log.
        stderr_log_path:  Captured solver stderr log.
        dat_path:         CalculiX ``.dat`` text results file.
        frd_path:         CalculiX ``.frd`` binary/text results file.
        sta_path:         CalculiX ``.sta`` convergence statistics.
        cvg_path:         CalculiX ``.cvg`` convergence log.
        other_files:      Any other files found in the directory.
    """

    input_deck_path: str | None = None
    stdout_log_path: str | None = None
    stderr_log_path: str | None = None
    dat_path: str | None = None
    frd_path: str | None = None
    sta_path: str | None = None
    cvg_path: str | None = None
    other_files: list[str] = field(default_factory=list)

    def any_parseable(self) -> bool:
        """Return True if at least one text-parseable artifact is present."""
        return any((
            self.dat_path,
            self.stdout_log_path,
            self.stderr_log_path,
        ))

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_deck_path":  self.input_deck_path,
            "stdout_log_path":  self.stdout_log_path,
            "stderr_log_path":  self.stderr_log_path,
            "dat_path":         self.dat_path,
            "frd_path":         self.frd_path,
            "sta_path":         self.sta_path,
            "cvg_path":         self.cvg_path,
            "other_files":      self.other_files,
        }


@dataclass
class PostprocessResult:
    """
    Structured postprocessing result for one pipeline case.

    ``metrics`` is a plain dict keyed by metric name (e.g.
    ``"max_von_mises_stress_mpa"``).  Use ``to_metric_set()`` to convert
    to the typed ``MetricSet`` schema for downstream modules.

    Attributes:
        success:              True if at least partial results were extracted.
        metrics:              Extracted scalar result values.
        stress_strain_points: (strain, stress_mpa) pairs if parseable.
        artifacts:            Discovered artifact registry.
        warnings:             Non-fatal notes about missing or partial data.
        metadata:             Additional context for logging/reporting.
        error_message:        Error description if success is False.
    """

    success: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    stress_strain_points: list[tuple[float, float]] = field(default_factory=list)
    artifacts: PostprocessArtifacts | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        artifacts = self.artifacts
        artifacts_dict = None
        if artifacts is not None:
            artifacts_dict = artifacts.to_dict()
        return {
            "success":              self.success,
            "metrics":              self.metrics,
            "stress_strain_points": self.stress_strain_points,
            "artifacts":            artifacts_dict,
            "warnings":             self.warnings,
            "metadata":             self.metadata,
            "error_message":        self.error_message,
        }

    def to_metric_set(self) -> MetricSet:
        """
        Convert extracted metrics to the typed ``MetricSet`` schema.

        Fields that were not extracted are left as ``None``.

        Returns:
            ``MetricSet`` for use by metrics.py, fatigue_model.py, etc.
        """
        m = self.metrics
        return MetricSet(
            max_von_mises_stress_mpa=m.get("max_von_mises_stress_mpa"),
            max_displacement_mm=m.get("max_displacement_mm"),
            hotspot_stress_mpa=m.get("hotspot_stress_mpa"),
            effective_stiffness_n_per_mm=m.get("effective_stiffness_n_per_mm"),
            effective_modulus_mpa=m.get("effective_modulus_mpa"),
            fatigue_risk_score=m.get("fatigue_risk_score"),
            stress_strain_points=self.stress_strain_points,
        )


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def _read_text_file(path: Path) -> str:
    """
    Read a text file and return its contents.

    Args:
        path: File path.

    Returns:
        File contents as a string.

    Raises:
        PostprocessingError: if the file cannot be read.
    """
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        raise PostprocessingError(
            f"Could not read file '{path}': {exc}"
        ) from exc


def _safe_read_text(path: Path | str | None) -> str | None:
    """
    Read a text file safely, returning ``None`` on any failure.

    Args:
        path: File path or None.

    Returns:
        File contents or ``None``.
    """
    if path is None:
        return None
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def _find_first_with_suffix(
    directory: Path,
    suffixes: tuple[str, ...],
) -> Path | None:
    """
    Return the first file in ``directory`` that has one of the given suffixes.

    Files are sorted by name for deterministic selection.

    Args:
        directory: Directory to search (non-recursive).
        suffixes:  Tuple of lowercase suffix strings (e.g. ``(".dat", ".frd")``).

    Returns:
        First matching ``Path`` or ``None``.
    """
    try:
        candidates = sorted(
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in suffixes
        )
        return candidates[0] if candidates else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Artifact discovery
# ---------------------------------------------------------------------------

def discover_solver_artifacts(directory: str | Path) -> PostprocessArtifacts:
    """
    Scan a solver/results directory and register all known artifact files.

    Args:
        directory: Directory to scan (typically ``runs/<case>/solver/``
                   or ``runs/<case>/results/``).

    Returns:
        ``PostprocessArtifacts`` with paths for each recognised file type.
        Returns an empty ``PostprocessArtifacts`` if the directory does
        not exist (with a debug log rather than raising).
    """
    directory = Path(directory)
    artifacts = PostprocessArtifacts()

    if not directory.exists():
        logger.debug(
            "Artifact directory does not exist: %s", directory
        )
        return artifacts

    recognised_paths: set[str] = set()

    # --- Input deck ---
    inp = _find_first_with_suffix(directory, (".inp",))
    if inp:
        artifacts.input_deck_path = str(inp)
        recognised_paths.add(str(inp))

    # --- CalculiX result files ---
    for attr, suffix in (
        ("dat_path", ".dat"),
        ("frd_path", ".frd"),
        ("sta_path", ".sta"),
        ("cvg_path", ".cvg"),
    ):
        found = _find_first_with_suffix(directory, (suffix,))
        if found:
            setattr(artifacts, attr, str(found))
            recognised_paths.add(str(found))

    # --- Log files ---
    # Search for stdout/stderr logs by naming convention
    for p in sorted(directory.rglob("*.log")):
        name = p.name.lower()
        if "stdout" in name and artifacts.stdout_log_path is None:
            artifacts.stdout_log_path = str(p)
            recognised_paths.add(str(p))
        elif "stderr" in name and artifacts.stderr_log_path is None:
            artifacts.stderr_log_path = str(p)
            recognised_paths.add(str(p))

    # Also check parent logs/ sub-directory
    logs_dir = directory / "logs"
    if logs_dir.exists():
        for p in sorted(logs_dir.iterdir()):
            if p.is_file():
                name = p.name.lower()
                if "stdout" in name and artifacts.stdout_log_path is None:
                    artifacts.stdout_log_path = str(p)
                    recognised_paths.add(str(p))
                elif "stderr" in name and artifacts.stderr_log_path is None:
                    artifacts.stderr_log_path = str(p)
                    recognised_paths.add(str(p))

    # --- Anything else ---
    try:
        for p in sorted(directory.iterdir()):
            if p.is_file() and str(p) not in recognised_paths:
                artifacts.other_files.append(str(p))
    except Exception:
        pass

    return artifacts


# ---------------------------------------------------------------------------
# Text-based scalar parsers
# ---------------------------------------------------------------------------
# ARCHITECTURAL DECISION — heuristic regex parsers:
#   These target common CalculiX .dat output patterns.  They are NOT
#   format-version-aware and will return None if the pattern is not matched.
#   All values returned are floats in MPa or mm consistent with project units.
#   Do NOT invent values; return None when no match is found.

def parse_max_von_mises_stress(text: str) -> float | None:
    """
    Attempt to extract the maximum von Mises stress from CalculiX text output.

    Targets common patterns in ``.dat`` files such as::

        MAXIMUM VON MISES STRESS =  4.321E+02
        max mises =  432.1

    Also tries to match summary lines from stdout logs.

    Args:
        text: Text content to search.

    Returns:
        Max von Mises stress in MPa, or ``None`` if not found.
    """
    patterns = [
        # CalculiX .dat style
        r"MAXIMUM\s+VON\s+MISES\s+STRESS[^=\n]*=\s*([+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)",
        r"MAX(?:IMUM)?\s+MISES[^=\n]*=\s*([+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)",
        # Common export-format label
        r"max_von_mises_stress_mpa[:\s=]+([+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)",
        # Generic "S, Mises" header followed by max line
        r"S\s+Mises[^\n]*\n[^\n]*\n?\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                value = float(m.group(1))
                if value >= 0.0:
                    return value
            except ValueError:
                continue
    return None


def parse_max_displacement(text: str) -> float | None:
    """
    Attempt to extract the maximum nodal displacement from CalculiX text output.

    Targets patterns such as::

        MAXIMUM DISPLACEMENT =  1.234E-01
        max displacement = 0.1234

    Args:
        text: Text content to search.

    Returns:
        Max displacement in mm, or ``None`` if not found.
    """
    patterns = [
        r"MAXIMUM\s+DISPLACEMENT[^=\n]*=\s*([+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)",
        r"MAX(?:IMUM)?\s+U[^=\n]*=\s*([+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)",
        r"max_displacement_mm[:\s=]+([+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                value = float(m.group(1))
                if value >= 0.0:
                    return value
            except ValueError:
                continue
    return None


def parse_hotspot_stress(text: str) -> float | None:
    """
    Attempt to extract a hotspot or local peak stress from text output.

    Hotspot stress is not a standard CalculiX output label.  This parser
    looks for explicitly labelled hotspot lines (e.g. written by a custom
    post-processing script or tagged output) or peak element stress summaries.

    VERSION 1 NOTE:
        Hotspot identification from raw solver output is not reliably
        automated in version 1.  If a hotspot value is not explicitly
        labelled in the text, this returns ``None``.  The fatigue proxy
        will fall back to max von Mises stress.

    Args:
        text: Text content to search.

    Returns:
        Hotspot stress in MPa, or ``None`` if not found.
    """
    patterns = [
        r"HOTSPOT\s+STRESS[^=\n]*=\s*([+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)",
        r"hotspot_stress_mpa[:\s=]+([+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)",
        r"LOCAL\s+PEAK\s+STRESS[^=\n]*=\s*([+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                value = float(m.group(1))
                if value >= 0.0:
                    return value
            except ValueError:
                continue
    return None


def parse_stress_strain_points(
    text: str,
) -> list[tuple[float, float]]:
    """
    Attempt to extract (strain, stress_mpa) point pairs from text output.

    Looks for two-column numeric data in sections labelled as stress-strain
    or strain-stress tables.  Accepts whitespace-separated pairs after a
    recognisable header.

    VERSION 1 NOTE:
        Full stress-strain curve extraction requires element-level result
        scanning.  Version 1 returns at most a few summary points if found;
        a complete curve requires dedicated FRD parsing (future version).

    Args:
        text: Text content to search.

    Returns:
        List of (strain, stress_mpa) tuples, or empty list if none found.
    """
    points: list[tuple[float, float]] = []

    # Look for a labelled section header followed by two-column data
    header_pattern = re.compile(
        r"(?:STRESS[-_]STRAIN|STRAIN[-_]STRESS|S[-,_]LE)[^\n]*\n"
        r"((?:\s*[+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
        r"\s+[+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\s*\n){1,})",
        re.IGNORECASE,
    )
    for block_match in header_pattern.finditer(text):
        block = str(block_match.group(1))
        for line in block.strip().splitlines():
            nums = re.findall(r"[+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", line)
            if len(nums) >= 2:
                try:
                    strain = float(nums[0])
                    stress = float(nums[1])
                    points.append((strain, stress))
                except ValueError:
                    continue

    return points


# ---------------------------------------------------------------------------
# Combined artifact parsing
# ---------------------------------------------------------------------------

def extract_scalar_results_from_artifacts(
    artifacts: PostprocessArtifacts,
) -> PostprocessResult:
    """
    Read available artifact text files and extract scalar result quantities.

    Preference order for text sources:
      1. ``.dat`` file (primary CalculiX text result summary)
      2. stdout log (secondary)
      3. stderr log (last resort)

    Args:
        artifacts: Discovered ``PostprocessArtifacts`` for one case.

    Returns:
        ``PostprocessResult`` with extracted metrics and traceability warnings.
    """
    result = PostprocessResult(success=False, artifacts=artifacts)
    metrics: dict[str, Any] = {}
    warnings: list[str] = []
    sources_used: list[str] = []

    # --- Check if FRD is present (log it but don't attempt binary parse) ---
    if artifacts.frd_path:
        result.metadata["frd_path"] = artifacts.frd_path
        result.metadata["frd_binary_parsing"] = (
            "FRD file detected but not parsed in version 1.  "
            "Full binary FRD parsing is a future-version extension."
        )
        warnings.append(
            f"FRD file found at '{artifacts.frd_path}' but binary FRD "
            f"parsing is not implemented in version 1.  "
            f"Scalar results will be extracted from .dat / log files only."
        )

    # --- Assemble text sources ---
    text_sources: list[tuple[str, str | None]] = [
        ("dat",    artifacts.dat_path),
        ("stdout", artifacts.stdout_log_path),
        ("stderr", artifacts.stderr_log_path),
    ]

    combined_texts: list[str] = []
    for label, path in text_sources:
        content = _safe_read_text(path)
        if content:
            combined_texts.append(f"\n{content}")
            sources_used.append(label)

    combined_text = "".join(combined_texts)
    if not combined_text.strip():
        warnings.append(
            "No parseable text content found in .dat, stdout, or stderr artifacts.  "
            "All scalar metrics are unavailable.  "
            "Run the solver and check that output requests were included in the input deck."
        )
        result.warnings = warnings
        result.metadata["sources_used"] = sources_used
        result.success = True   # partial success; no error
        result.metrics = metrics
        return result

    result.metadata["sources_used"] = sources_used

    # --- Extract scalars ---
    max_mises = parse_max_von_mises_stress(combined_text)
    if max_mises is not None:
        metrics["max_von_mises_stress_mpa"] = max_mises
    else:
        warnings.append(
            "max_von_mises_stress_mpa could not be extracted from available "
            "text artifacts.  Check that *EL FILE output includes S, MISES."
        )

    max_disp = parse_max_displacement(combined_text)
    if max_disp is not None:
        metrics["max_displacement_mm"] = max_disp
    else:
        warnings.append(
            "max_displacement_mm could not be extracted.  "
            "Check that *NODE FILE output includes U."
        )

    hotspot = parse_hotspot_stress(combined_text)
    if hotspot is not None:
        metrics["hotspot_stress_mpa"] = hotspot
    else:
        warnings.append(
            "hotspot_stress_mpa not found in text output.  "
            "The fatigue proxy will fall back to max_von_mises_stress_mpa."
        )

    ss_points = parse_stress_strain_points(combined_text)
    if ss_points:
        result.stress_strain_points = ss_points
    else:
        warnings.append(
            "No stress-strain data points found in text output.  "
            "Stress-strain curve will be empty."
        )

    result.metrics = metrics
    result.warnings = warnings
    result.success = True
    return result


# ---------------------------------------------------------------------------
# Main postprocessing entry point
# ---------------------------------------------------------------------------

def postprocess_solver_outputs(
    artifacts_directory: str | Path,
) -> PostprocessResult:
    """
    Discover artifacts and extract scalar results for a pipeline case.

    This is the primary postprocessing entry point.

    Args:
        artifacts_directory: Directory containing solver output files.
                             Typically ``runs/<case>/solver/`` or
                             ``runs/<case>/results/``.

    Returns:
        ``PostprocessResult`` with extracted metrics and warnings.
    """
    directory = Path(artifacts_directory)
    logger.info("Postprocessing artifacts in: %s", directory)

    artifacts = discover_solver_artifacts(directory)
    result = extract_scalar_results_from_artifacts(artifacts)

    logger.info(
        "Postprocessing complete: %d metrics extracted, %d warnings.",
        len(result.metrics), len(result.warnings),
    )
    return result


# ---------------------------------------------------------------------------
# Case convenience helper
# ---------------------------------------------------------------------------

def postprocess_case_outputs(
    case_definition: Any,
    artifacts_directory: str | Path | None = None,
) -> PostprocessResult:
    """
    Postprocess outputs for a specific pipeline case.

    ARCHITECTURAL DECISION — duck-typed on case_definition:
        ``CaseDefinition`` is not imported explicitly to avoid cross-layer
        coupling.  The function uses ``getattr`` to read ``.case_id`` for
        logging and attempts to infer the artifacts directory from
        ``case_definition.paths`` if not explicitly provided.

    Args:
        case_definition:      Object with ``.case_id`` and optionally
                              ``.paths.results_directory`` attributes.
        artifacts_directory:  Override path to artifacts (optional).

    Returns:
        ``PostprocessResult``.
    """
    case_id = getattr(case_definition, "case_id", "unknown_case")

    if artifacts_directory is None:
        # Try to infer from case paths
        paths_obj = getattr(case_definition, "paths", None)
        if paths_obj is not None:
            inferred = getattr(paths_obj, "results_directory", None)
            if inferred is not None:
                artifacts_directory = inferred
            else:
                inferred = getattr(paths_obj, "solver_directory", None)
                if inferred is not None:
                    artifacts_directory = inferred

    if artifacts_directory is None:
        result = PostprocessResult(success=False)
        result.error_message = (
            f"Case '{case_id}': no artifacts_directory provided and could not "
            f"be inferred from case_definition.  Pass artifacts_directory explicitly."
        )
        return result

    logger.info("Postprocessing case '%s' from '%s'.", case_id, artifacts_directory)
    result = postprocess_solver_outputs(artifacts_directory)
    result.metadata["case_id"] = case_id
    return result


# ---------------------------------------------------------------------------
# CaseResult compatibility helper
# ---------------------------------------------------------------------------

def build_case_result_from_postprocess(
    case_id: str,
    postprocess_result: PostprocessResult,
    runtime_seconds: float | None = None,
    solver_return_code: int | None = None,
) -> CaseResult:
    """
    Build a ``CaseResult`` from a ``PostprocessResult``.

    Maps extracted metrics, artifacts, and warnings into the canonical
    schema-layer result object used by the workflow orchestrator.

    Args:
        case_id:              Case identifier string.
        postprocess_result:   Completed postprocess result.
        runtime_seconds:      Solver runtime (from runner.py) if available.
        solver_return_code:   CalculiX return code if available.

    Returns:
        ``CaseResult`` ready for workflow use.
    """
    metric_set = postprocess_result.to_metric_set()

    artifact_dict: dict[str, str] = {}
    if postprocess_result.artifacts:
        for attr, label in (
            ("input_deck_path",  "input_deck"),
            ("dat_path",         "dat_file"),
            ("frd_path",         "frd_file"),
            ("stdout_log_path",  "stdout_log"),
            ("stderr_log_path",  "stderr_log"),
        ):
            val = getattr(postprocess_result.artifacts, attr)
            if val:
                artifact_dict[label] = val

    status = (
        CaseStatus.COMPLETED
        if postprocess_result.success
        else CaseStatus.FAILED
    )

    case_result = CaseResult(
        case_id=case_id,
        status=status,
        success=postprocess_result.success,
        metrics=metric_set,
        error_message=postprocess_result.error_message,
        runtime_seconds=runtime_seconds,
        solver_return_code=solver_return_code,
        artifacts=artifact_dict,
        metadata={
            **postprocess_result.metadata,
            "postprocess_warnings": postprocess_result.warnings,
        },
    )

    return case_result
