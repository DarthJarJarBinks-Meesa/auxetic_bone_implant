"""
src/workflow/status_tracker.py
================================
Workflow status-tracking module for the auxetic plate pipeline.

This module manages per-case execution status, status-file reads/writes,
lifecycle transitions, and lightweight progress metadata for per-case run
folders.

PIPELINE POSITION:
    orchestrator.py  →  [THIS MODULE]  →  status.txt / metadata.json
                                       →  case_runner.py (query helpers)

ARCHITECTURAL DECISION — filesystem-backed status, not a database:
    Status is stored in two files per case run directory:
      - ``status.txt``    : single-line human-readable status value
                            (e.g. "running", "completed")
      - ``metadata.json`` : richer structured payload with timestamps,
                            stage name, messages, and optional metadata
    This is intentionally simple: every file manager, shell script, and
    Python subprocess can read and write these without special tooling.
    The trade-off (no atomic transactions, no locking) is acceptable for
    version-1 single-process sweeps.

ARCHITECTURAL DECISION — no state-machine framework:
    Transition logic is implemented as a lightweight helper function
    (``is_valid_transition``) and convenience wrappers (``mark_case_*``).
    There is no base class, metaclass, or external library involved.  This
    keeps the module dependency-free and easy to audit.

ARCHITECTURAL DECISION — UTC timestamps as ISO 8601 strings:
    All timestamps are stored as UTC ISO 8601 strings (e.g.
    "2026-03-22T17:55:00Z").  Using strings rather than datetimes avoids
    serialisation edge cases and is directly human-readable in the files.

UNITS: not applicable — this module manages file paths and metadata only.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from workflow.case_schema import CaseDefinition, CaseStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class StatusTrackerError(Exception):
    """
    Raised for unrecoverable status-tracking failures.

    Covers:
      - invalid status transitions in strict helpers
      - malformed or unreadable status files
      - path resolution failures for the case run directory
      - file write failures in hard-fail situations

    Soft failures (e.g. missing files when reading optional status) are
    represented by returning ``None`` rather than raising.
    """


# ---------------------------------------------------------------------------
# Typed dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CaseStatusRecord:
    """
    Rich per-case status record combining status.txt and metadata.json data.

    Attributes:
        case_id:             Case identifier string.
        status:              Current lifecycle status.
        updated_at_utc:      ISO 8601 UTC timestamp of the last status write.
        started_at_utc:      ISO 8601 UTC timestamp when the case started running.
        completed_at_utc:    ISO 8601 UTC timestamp when the case completed or failed.
        message:             Optional human-readable status message or error note.
        stage:               Optional pipeline stage name (e.g. ``"meshing"``).
        runtime_seconds:     Elapsed runtime if known (completed/failed cases).
        metadata:            Open dict for additional context (warnings, paths, etc.).
    """

    case_id: str
    status: CaseStatus
    updated_at_utc: str
    started_at_utc: str | None = None
    completed_at_utc: str | None = None
    message: str | None = None
    stage: str | None = None
    runtime_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary of all record fields."""
        return {
            "case_id":           self.case_id,
            "status":            self.status.value,
            "updated_at_utc":   self.updated_at_utc,
            "started_at_utc":   self.started_at_utc,
            "completed_at_utc": self.completed_at_utc,
            "message":           self.message,
            "stage":             self.stage,
            "runtime_seconds":  self.runtime_seconds,
            "metadata":          self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CaseStatusRecord":
        """
        Deserialise a ``CaseStatusRecord`` from a plain dictionary.

        Unknown keys in ``data`` are silently ignored so that older metadata
        files remain readable after schema additions.

        Args:
            data: Dict loaded from ``metadata.json`` or equivalent.

        Returns:
            ``CaseStatusRecord`` instance.

        Raises:
            StatusTrackerError: if required fields (case_id, status) are missing
                or the status value is not a valid ``CaseStatus``.
        """
        try:
            case_id = str(data["case_id"])
            raw_status = data["status"]
            status = CaseStatus(raw_status)
        except KeyError as exc:
            raise StatusTrackerError(
                f"CaseStatusRecord.from_dict: missing required field {exc}."
            ) from exc
        except ValueError:
            raise StatusTrackerError(
                f"CaseStatusRecord.from_dict: unrecognised status value "
                f"'{data.get('status')}'.  Valid values: "
                f"{[s.value for s in CaseStatus]}."
            )

        return cls(
            case_id=case_id,
            status=status,
            updated_at_utc=str(data.get("updated_at_utc", utc_now_iso())),
            started_at_utc=data.get("started_at_utc"),
            completed_at_utc=data.get("completed_at_utc"),
            message=data.get("message"),
            stage=data.get("stage"),
            runtime_seconds=data.get("runtime_seconds"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class StatusWriteResult:
    """
    Result of a status write operation.

    Attributes:
        success:        True if both status.txt and metadata.json were written.
        status_file:    Path to the written status.txt file (if successful).
        metadata_file:  Path to the written metadata.json file (if successful).
        warnings:       Non-fatal notes (e.g. partial write, directory created).
        error_message:  Error description if ``success`` is False.
    """

    success: bool
    status_file: str | None = None
    metadata_file: str | None = None
    warnings: list[str] = field(default_factory=list)
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary of this result."""
        return {
            "success":       self.success,
            "status_file":   self.status_file,
            "metadata_file": self.metadata_file,
            "warnings":      self.warnings,
            "error_message": self.error_message,
        }


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    """
    Return the current UTC time as an ISO 8601 string ending in ``Z``.

    Example output: ``"2026-03-22T17:55:00Z"``

    Returns:
        UTC timestamp string.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _elapsed_seconds(
    start_iso: str | None,
    end_iso: str | None,
) -> float | None:
    """
    Compute elapsed seconds between two ISO 8601 UTC timestamp strings.

    Args:
        start_iso: ISO 8601 start timestamp (``"YYYY-MM-DDTHH:MM:SSZ"``).
        end_iso:   ISO 8601 end timestamp.

    Returns:
        Elapsed seconds as a float, or ``None`` if either timestamp is
        missing or unparseable.
    """
    if start_iso is None or end_iso is None:
        return None
    try:
        fmt = "%Y-%m-%dT%H:%M:%SZ"
        t_start = datetime.strptime(str(start_iso), fmt).replace(tzinfo=timezone.utc)
        t_end   = datetime.strptime(str(end_iso),   fmt).replace(tzinfo=timezone.utc)
        delta = (t_end - t_start).total_seconds()
        return float(f"{delta:.3f}")
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Project-root resolution helper
# ---------------------------------------------------------------------------

def _resolve_project_root(project_root: str | Path | None) -> Path:
    """
    Resolve the project root directory.

    Uses the supplied value if present; otherwise tries auto-detection via
    ``utils.config_loader``, falling back to ``Path.cwd()``.

    Args:
        project_root: Explicit root path or None.

    Returns:
        Resolved ``Path``.
    """
    if project_root is not None:
        return Path(project_root).resolve()
    try:
        from utils.config_loader import _find_project_root  # type: ignore[import]
        return _find_project_root(Path(__file__).parent)
    except Exception:
        return Path.cwd()


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------

def _case_id_from(case_definition_or_id: CaseDefinition | str) -> str:
    """
    Extract the case_id string from either a ``CaseDefinition`` or a raw string.

    Args:
        case_definition_or_id: Full ``CaseDefinition`` or case ID string.

    Returns:
        Case ID string.
    """
    if isinstance(case_definition_or_id, str):
        return case_definition_or_id
    return case_definition_or_id.case_id


def resolve_case_run_dir(
    case_definition_or_id: CaseDefinition | str,
    project_root: str | Path | None = None,
) -> Path:
    """
    Return the canonical run directory path for a case.

    Path: ``<project_root>/runs/<case_id>/``

    Directories are **not** created by this function — creation happens only
    in write helpers.

    Args:
        case_definition_or_id: ``CaseDefinition`` or case ID string.
        project_root:          Project root (auto-detected if None).

    Returns:
        ``Path`` to the case run directory (may not yet exist).
    """
    case_id = _case_id_from(case_definition_or_id)
    root = _resolve_project_root(project_root)
    return root / "runs" / case_id


def resolve_case_status_path(
    case_definition_or_id: CaseDefinition | str,
    project_root: str | Path | None = None,
) -> Path:
    """
    Return the canonical path to ``status.txt`` for a case.

    Path: ``<project_root>/runs/<case_id>/status.txt``

    Args:
        case_definition_or_id: ``CaseDefinition`` or case ID string.
        project_root:          Project root (auto-detected if None).

    Returns:
        ``Path`` to the status file (may not yet exist).
    """
    return resolve_case_run_dir(case_definition_or_id, project_root) / "status.txt"


def resolve_case_metadata_path(
    case_definition_or_id: CaseDefinition | str,
    project_root: str | Path | None = None,
) -> Path:
    """
    Return the canonical path to ``metadata.json`` for a case.

    Path: ``<project_root>/runs/<case_id>/metadata.json``

    Args:
        case_definition_or_id: ``CaseDefinition`` or case ID string.
        project_root:          Project root (auto-detected if None).

    Returns:
        ``Path`` to the metadata file (may not yet exist).
    """
    return resolve_case_run_dir(case_definition_or_id, project_root) / "metadata.json"


# ---------------------------------------------------------------------------
# Low-level read/write helpers
# ---------------------------------------------------------------------------

def _write_text(path: Path, text: str) -> None:
    """
    Write a text string to a file, creating parent directories as needed.

    Args:
        path: Destination file path.
        text: Content to write (UTF-8).

    Raises:
        StatusTrackerError: if the write fails.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except Exception as exc:
        raise StatusTrackerError(
            f"Failed to write file '{path}': {exc}"
        ) from exc


def _read_text(path: Path) -> str:
    """
    Read a file and return its contents as a string.

    Args:
        path: File path to read.

    Returns:
        File contents.

    Raises:
        StatusTrackerError: if the file cannot be read.
    """
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        raise StatusTrackerError(
            f"Failed to read file '{path}': {exc}"
        ) from exc


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """
    Write a JSON payload to a file, creating parent directories as needed.

    Args:
        path:    Destination file path.
        payload: JSON-serialisable mapping.

    Raises:
        StatusTrackerError: if serialisation or the write fails.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(dict(payload), fh, indent=2, default=str)
    except Exception as exc:
        raise StatusTrackerError(
            f"Failed to write JSON file '{path}': {exc}"
        ) from exc


def _read_json(path: Path) -> dict[str, Any]:
    """
    Read and parse a JSON file.

    Args:
        path: File path to read.

    Returns:
        Parsed dict.

    Raises:
        StatusTrackerError: if the file cannot be read or parsed.
    """
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise StatusTrackerError(
                f"JSON file '{path}' does not contain a top-level object."
            )
        return data
    except StatusTrackerError:
        raise
    except Exception as exc:
        raise StatusTrackerError(
            f"Failed to read JSON file '{path}': {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Transition validation
# ---------------------------------------------------------------------------

# Allowed transitions: old_status (or None) → set of valid new statuses
_VALID_TRANSITIONS: dict[CaseStatus | None, set[CaseStatus]] = {
    None: {
        CaseStatus.PENDING,
        CaseStatus.RUNNING,
        CaseStatus.COMPLETED,
        CaseStatus.FAILED,
        CaseStatus.SKIPPED,
    },
    CaseStatus.PENDING: {
        CaseStatus.RUNNING,
        CaseStatus.SKIPPED,
        CaseStatus.FAILED,
    },
    CaseStatus.RUNNING: {
        CaseStatus.COMPLETED,
        CaseStatus.FAILED,
        CaseStatus.SKIPPED,
    },
    CaseStatus.COMPLETED: {
        # Allow re-runs; callers decide whether to enforce stricter policies.
        CaseStatus.PENDING,
        CaseStatus.RUNNING,
    },
    CaseStatus.FAILED: {
        CaseStatus.PENDING,
        CaseStatus.RUNNING,
        CaseStatus.SKIPPED,
    },
    CaseStatus.SKIPPED: {
        CaseStatus.PENDING,
        CaseStatus.RUNNING,
    },
}


def is_valid_transition(
    old_status: CaseStatus | None,
    new_status: CaseStatus,
) -> bool:
    """
    Return True if transitioning from ``old_status`` to ``new_status`` is valid.

    A ``None`` old status means the case has no prior status record (first write).

    ARCHITECTURAL DECISION — permissive in version 1:
        The transition table is intentionally broad.  Orchestrators that need
        stricter enforcement should use ``require_case_not_completed`` or
        check before writing.  The helper here exists to make valid paths
        inspectable and testable without being a blocking gate.

    Args:
        old_status: Current ``CaseStatus`` or ``None`` (no prior record).
        new_status: Status being written.

    Returns:
        True if the transition is in the allowed set.
    """
    allowed = _VALID_TRANSITIONS.get(old_status, set())
    return new_status in allowed


# ---------------------------------------------------------------------------
# Main read function
# ---------------------------------------------------------------------------

def read_case_status(
    case_definition_or_id: CaseDefinition | str,
    project_root: str | Path | None = None,
) -> CaseStatusRecord | None:
    """
    Read the current status of a pipeline case from its run directory.

    Combines data from ``status.txt`` (human-readable single-line status) and
    ``metadata.json`` (structured payload) into a ``CaseStatusRecord``.

    Preference rules:
      - If ``metadata.json`` exists and is valid, it is the authoritative source.
      - If only ``status.txt`` exists, a minimal record is synthesised from it.
      - If neither exists, ``None`` is returned (the case has no recorded status).

    Args:
        case_definition_or_id: ``CaseDefinition`` or case ID string.
        project_root:          Project root (auto-detected if None).

    Returns:
        ``CaseStatusRecord`` or ``None`` if no status files exist yet.
    """
    case_id = _case_id_from(case_definition_or_id)
    status_path   = resolve_case_status_path(case_id, project_root)
    metadata_path = resolve_case_metadata_path(case_id, project_root)

    # --- Try metadata.json first ---
    if metadata_path.exists():
        try:
            data = _read_json(metadata_path)
            record = CaseStatusRecord.from_dict(data)
            logger.debug(
                "Read status for '%s' from metadata.json: %s",
                case_id, record.status.value,
            )
            return record
        except StatusTrackerError as exc:
            logger.warning(
                "Could not parse metadata.json for '%s': %s.  "
                "Falling back to status.txt.",
                case_id, exc,
            )

    # --- Fall back to status.txt only ---
    if status_path.exists():
        try:
            raw = _read_text(status_path).strip()
            status = CaseStatus(raw)
            logger.debug(
                "Read status for '%s' from status.txt: %s", case_id, raw
            )
            return CaseStatusRecord(
                case_id=case_id,
                status=status,
                updated_at_utc=utc_now_iso(),
            )
        except (StatusTrackerError, ValueError) as exc:
            logger.warning(
                "Could not parse status.txt for '%s': %s.", case_id, exc
            )

    # --- Neither file exists ---
    return None


# ---------------------------------------------------------------------------
# Main write function
# ---------------------------------------------------------------------------

def write_case_status(
    case_definition_or_id: CaseDefinition | str,
    status: CaseStatus,
    project_root: str | Path | None = None,
    message: str | None = None,
    stage: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    started_at_utc: str | None = None,
    completed_at_utc: str | None = None,
    runtime_seconds: float | None = None,
) -> StatusWriteResult:
    """
    Write the status of a pipeline case to its run directory.

    Writes:
      - ``runs/<case_id>/status.txt``    — single-line human-readable status
      - ``runs/<case_id>/metadata.json`` — full structured payload

    If a ``metadata.json`` already exists, it is merged with the new data
    (new values take precedence over old ones).

    Args:
        case_definition_or_id: ``CaseDefinition`` or case ID string.
        status:                New lifecycle status to record.
        project_root:          Project root (auto-detected if None).
        message:               Optional human-readable message or error note.
        stage:                 Optional pipeline stage name (e.g. ``"meshing"``).
        metadata:              Optional additional context dict (merged into payload).
        started_at_utc:        ISO 8601 UTC start timestamp (used for RUNNING status).
        completed_at_utc:      ISO 8601 UTC completion timestamp.
        runtime_seconds:       Elapsed runtime in seconds.

    Returns:
        ``StatusWriteResult`` indicating what was written (or what failed).
    """
    case_id = _case_id_from(case_definition_or_id)
    now = utc_now_iso()

    status_path   = resolve_case_status_path(case_id, project_root)
    metadata_path = resolve_case_metadata_path(case_id, project_root)

    result = StatusWriteResult(success=False)
    warnings: list[str] = []

    # --- Load existing metadata for merge (best-effort) ---
    existing: dict[str, Any] = {}
    if metadata_path.exists():
        try:
            existing = _read_json(metadata_path)
        except StatusTrackerError as exc:
            warnings.append(
                f"Could not read existing metadata.json; will overwrite: {exc}"
            )

    # --- Build record ---
    # Preserve timestamps from existing record when not supplied
    resolved_started = (
        started_at_utc
        or existing.get("started_at_utc")
    )
    resolved_completed = (
        completed_at_utc
        or existing.get("completed_at_utc")
    )
    resolved_runtime = (
        runtime_seconds
        if runtime_seconds is not None
        else existing.get("runtime_seconds")
    )

    merged_meta: dict[str, Any] = dict(existing.get("metadata") or {})
    if metadata:
        merged_meta.update(metadata)

    record = CaseStatusRecord(
        case_id=case_id,
        status=status,
        updated_at_utc=now,
        started_at_utc=resolved_started,
        completed_at_utc=resolved_completed,
        message=message,
        stage=stage,
        runtime_seconds=resolved_runtime,
        metadata=merged_meta,
    )

    # --- Write status.txt ---
    try:
        _write_text(status_path, status.value + "\n")
        result.status_file = str(status_path)
    except StatusTrackerError as exc:
        result.error_message = f"Failed to write status.txt: {exc}"
        result.warnings = warnings
        return result

    # --- Write metadata.json ---
    try:
        _write_json(metadata_path, record.to_dict())
        result.metadata_file = str(metadata_path)
    except StatusTrackerError as exc:
        # status.txt was written; note the partial write
        warnings.append(
            f"status.txt written but metadata.json failed: {exc}.  "
            "Status is visible but structured metadata is unavailable."
        )
        result.warnings = warnings
        result.error_message = f"Partial write — metadata.json not written: {exc}"
        return result

    result.success = True
    result.warnings = warnings
    logger.debug(
        "Status written for '%s': %s (stage=%s).",
        case_id, status.value, stage or "—",
    )
    return result


# ---------------------------------------------------------------------------
# Transition convenience helpers
# ---------------------------------------------------------------------------

def mark_case_pending(
    case_definition_or_id: CaseDefinition | str,
    project_root: str | Path | None = None,
    message: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> StatusWriteResult:
    """
    Mark a case as ``PENDING``.

    Clears ``started_at_utc``, ``completed_at_utc``, and ``runtime_seconds``
    so the record is clean for a fresh run.

    Args:
        case_definition_or_id: ``CaseDefinition`` or case ID string.
        project_root:          Project root (auto-detected if None).
        message:               Optional status message.
        metadata:              Optional additional metadata.

    Returns:
        ``StatusWriteResult``.
    """
    return write_case_status(
        case_definition_or_id=case_definition_or_id,
        status=CaseStatus.PENDING,
        project_root=project_root,
        message=message or "Case queued and waiting to run.",
        stage="pending",
        metadata=metadata,
        started_at_utc=None,
        completed_at_utc=None,
        runtime_seconds=None,
    )


def mark_case_running(
    case_definition_or_id: CaseDefinition | str,
    project_root: str | Path | None = None,
    stage: str | None = None,
    message: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> StatusWriteResult:
    """
    Mark a case as ``RUNNING`` and record the start timestamp.

    If a ``started_at_utc`` is not already recorded, it is set to now.

    Args:
        case_definition_or_id: ``CaseDefinition`` or case ID string.
        project_root:          Project root (auto-detected if None).
        stage:                 Current pipeline stage (e.g. ``"geometry"``).
        message:               Optional status message.
        metadata:              Optional additional metadata.

    Returns:
        ``StatusWriteResult``.
    """
    # Preserve an existing started_at_utc if it was set in a prior RUNNING write
    existing_record = read_case_status(case_definition_or_id, project_root)
    started_at = (
        existing_record.started_at_utc
        if (existing_record and existing_record.started_at_utc)
        else utc_now_iso()
    )

    return write_case_status(
        case_definition_or_id=case_definition_or_id,
        status=CaseStatus.RUNNING,
        project_root=project_root,
        message=message or f"Case running: stage={stage or 'unknown'}.",
        stage=stage or "running",
        metadata=metadata,
        started_at_utc=started_at,
    )


def mark_case_completed(
    case_definition_or_id: CaseDefinition | str,
    project_root: str | Path | None = None,
    message: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    runtime_seconds: float | None = None,
) -> StatusWriteResult:
    """
    Mark a case as ``COMPLETED`` and record the completion timestamp.

    Runtime is computed automatically if the case has a ``started_at_utc``
    in its existing record and ``runtime_seconds`` is not explicitly supplied.

    Args:
        case_definition_or_id: ``CaseDefinition`` or case ID string.
        project_root:          Project root (auto-detected if None).
        message:               Optional status message.
        metadata:              Optional additional metadata.
        runtime_seconds:       Explicit runtime override.

    Returns:
        ``StatusWriteResult``.
    """
    now = utc_now_iso()
    existing_record = read_case_status(case_definition_or_id, project_root)

    # Auto-compute runtime if possible
    if runtime_seconds is None and existing_record:
        runtime_seconds = _elapsed_seconds(
            existing_record.started_at_utc, now
        )

    return write_case_status(
        case_definition_or_id=case_definition_or_id,
        status=CaseStatus.COMPLETED,
        project_root=project_root,
        message=message or "Case completed successfully.",
        stage="completed",
        metadata=metadata,
        completed_at_utc=now,
        runtime_seconds=runtime_seconds,
    )


def mark_case_failed(
    case_definition_or_id: CaseDefinition | str,
    project_root: str | Path | None = None,
    error_message: str | None = None,
    stage: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    runtime_seconds: float | None = None,
) -> StatusWriteResult:
    """
    Mark a case as ``FAILED`` and record the failure timestamp and reason.

    Args:
        case_definition_or_id: ``CaseDefinition`` or case ID string.
        project_root:          Project root (auto-detected if None).
        error_message:         Description of the failure.
        stage:                 Pipeline stage where the failure occurred.
        metadata:              Optional additional metadata (e.g. traceback excerpt).
        runtime_seconds:       Elapsed runtime if known.

    Returns:
        ``StatusWriteResult``.
    """
    now = utc_now_iso()
    existing_record = read_case_status(case_definition_or_id, project_root)

    if runtime_seconds is None and existing_record:
        runtime_seconds = _elapsed_seconds(
            existing_record.started_at_utc, now
        )

    return write_case_status(
        case_definition_or_id=case_definition_or_id,
        status=CaseStatus.FAILED,
        project_root=project_root,
        message=error_message or "Case failed — see metadata for details.",
        stage=stage or "failed",
        metadata=metadata,
        completed_at_utc=now,
        runtime_seconds=runtime_seconds,
    )


def mark_case_skipped(
    case_definition_or_id: CaseDefinition | str,
    project_root: str | Path | None = None,
    reason: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> StatusWriteResult:
    """
    Mark a case as ``SKIPPED`` (e.g. because a cached result already exists).

    Args:
        case_definition_or_id: ``CaseDefinition`` or case ID string.
        project_root:          Project root (auto-detected if None).
        reason:                Optional human-readable reason for skipping.
        metadata:              Optional additional metadata.

    Returns:
        ``StatusWriteResult``.
    """
    return write_case_status(
        case_definition_or_id=case_definition_or_id,
        status=CaseStatus.SKIPPED,
        project_root=project_root,
        message=reason or "Case skipped (result already cached).",
        stage="skipped",
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Status query helpers
# ---------------------------------------------------------------------------

def case_is_completed(
    case_definition_or_id: CaseDefinition | str,
    project_root: str | Path | None = None,
) -> bool:
    """
    Return True if the case status is ``COMPLETED``.

    Args:
        case_definition_or_id: ``CaseDefinition`` or case ID string.
        project_root:          Project root (auto-detected if None).

    Returns:
        True if the recorded status is ``CaseStatus.COMPLETED``.
    """
    record = read_case_status(case_definition_or_id, project_root)
    return record is not None and record.status == CaseStatus.COMPLETED


def case_is_running(
    case_definition_or_id: CaseDefinition | str,
    project_root: str | Path | None = None,
) -> bool:
    """
    Return True if the case status is ``RUNNING``.

    Args:
        case_definition_or_id: ``CaseDefinition`` or case ID string.
        project_root:          Project root (auto-detected if None).

    Returns:
        True if the recorded status is ``CaseStatus.RUNNING``.
    """
    record = read_case_status(case_definition_or_id, project_root)
    return record is not None and record.status == CaseStatus.RUNNING


def case_has_failed(
    case_definition_or_id: CaseDefinition | str,
    project_root: str | Path | None = None,
) -> bool:
    """
    Return True if the case status is ``FAILED``.

    Args:
        case_definition_or_id: ``CaseDefinition`` or case ID string.
        project_root:          Project root (auto-detected if None).

    Returns:
        True if the recorded status is ``CaseStatus.FAILED``.
    """
    record = read_case_status(case_definition_or_id, project_root)
    return record is not None and record.status == CaseStatus.FAILED


def case_should_be_skipped(
    case_definition_or_id: CaseDefinition | str,
    skip_completed_cases: bool = True,
    project_root: str | Path | None = None,
) -> bool:
    """
    Return True if a case should be skipped in the current run.

    Conditions for skipping:
      - ``skip_completed_cases=True`` and the case is already ``COMPLETED``.
      - The case is already ``SKIPPED``.

    ``RUNNING`` cases are NOT automatically skipped here — the caller should
    decide whether a zombie running case should be re-attempted.

    Args:
        case_definition_or_id: ``CaseDefinition`` or case ID string.
        skip_completed_cases:  If True, already-completed cases are skipped.
        project_root:          Project root (auto-detected if None).

    Returns:
        True if the case should be skipped.
    """
    record = read_case_status(case_definition_or_id, project_root)
    if record is None:
        return False
    if record.status == CaseStatus.SKIPPED:
        return True
    if skip_completed_cases and record.status == CaseStatus.COMPLETED:
        return True
    return False


# ---------------------------------------------------------------------------
# Strict wrapper
# ---------------------------------------------------------------------------

def require_case_not_completed(
    case_definition_or_id: CaseDefinition | str,
    project_root: str | Path | None = None,
) -> None:
    """
    Raise ``StatusTrackerError`` if the case is already ``COMPLETED``.

    Use this in orchestrators or case runners that must not overwrite a
    completed result unless the caller has explicitly cleared the status.

    Args:
        case_definition_or_id: ``CaseDefinition`` or case ID string.
        project_root:          Project root (auto-detected if None).

    Raises:
        StatusTrackerError: if the case status is ``COMPLETED``.
    """
    if case_is_completed(case_definition_or_id, project_root):
        case_id = _case_id_from(case_definition_or_id)
        raise StatusTrackerError(
            f"Case '{case_id}' is already COMPLETED.  "
            "Clear its status file or use a skip/rerun policy to proceed."
        )


# ---------------------------------------------------------------------------
# Status summary helper
# ---------------------------------------------------------------------------

def summarize_case_statuses(
    case_ids: Sequence[str],
    project_root: str | Path | None = None,
) -> dict[str, int]:
    """
    Count cases by status across a collection of case IDs.

    Useful for the orchestrator to report sweep progress at a glance.

    Args:
        case_ids:     Sequence of case ID strings to query.
        project_root: Project root (auto-detected if None).

    Returns:
        Dict mapping status name → count.  Includes an ``"unknown"`` bucket
        for cases with no status file.  All six status values are always
        present in the output (zero if no cases have that status).

    Example::

        {
            "pending":   3,
            "running":   1,
            "completed": 8,
            "failed":    2,
            "skipped":   1,
            "unknown":   0,
        }
    """
    counts: dict[str, int] = {
        "pending":   0,
        "running":   0,
        "completed": 0,
        "failed":    0,
        "skipped":   0,
        "unknown":   0,
    }

    for case_id in case_ids:
        record = read_case_status(case_id, project_root)
        if record is None:
            counts["unknown"] += 1
        else:
            key = record.status.value
            counts[key] = counts.get(key, 0) + 1

    return counts
