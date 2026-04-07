"""
src/simulation/runner.py
=========================
CalculiX solver-runner module for the auxetic plate pipeline.

This module executes CalculiX jobs from generated ``.inp`` input decks,
captures logs and return codes, and returns structured ``SolverRunResult``
objects for downstream postprocessing.

PIPELINE POSITION:
    solver/input.inp  →  [THIS MODULE]  →  solver/*.frd, *.dat  →  postprocess.py

ARCHITECTURAL DECISION — export-only mode is the version-1 default:
    ``base_config.yaml`` sets ``solver.run_solver_by_default: false`` and
    ``solver.export_solver_input_only: true``.  The runner respects these
    flags through ``default_run_options()``.  When ``run_solver=False`` the
    runner returns a valid ``SolverRunResult`` with ``executed=False`` and
    ``success=True`` so the downstream workflow does not abort — it simply
    knows that no FE results are available yet.  This allows the full
    pipeline to be exercised (geometry, mesh, export) on any machine
    regardless of whether CalculiX is installed.

ARCHITECTURAL DECISION — no fake solver execution:
    If ``run_solver=True`` but CalculiX is not installed, the runner returns
    a structured ``SolverRunResult`` with ``success=False`` and a clear
    ``error_message``.  It does NOT pretend the run succeeded.

ARCHITECTURAL DECISION — subprocess.run with timeout:
    CalculiX is invoked via ``subprocess.run`` with an optional timeout.
    The timeout prevents a pathological mesh from hanging the entire sweep.
    Stdout and stderr are captured to log files so the pipeline does not
    buffer arbitrarily large output in memory.

ARCHITECTURAL DECISION — CalculiX job invocation convention:
    CalculiX expects to be called as::

        ccx <job_basename>

    where ``<job_basename>`` is the ``.inp`` filename without extension,
    and the working directory is the directory containing the ``.inp`` file.
    The solver writes result files (``.frd``, ``.dat``, ``.cvg``) to the
    working directory under the same basename.  This is the standard
    CalculiX invocation pattern.

ARCHITECTURAL DECISION — result parsing is out of scope here:
    This module does not parse ``.frd`` or ``.dat`` result files.  It only
    manages process execution and log capture.  All result parsing is in
    ``postprocess.py``.

UNITS: not applicable — this module manages processes and files only.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# CalculiX standard result file extensions (used for artifact listing)
_CCX_RESULT_EXTENSIONS: tuple[str, ...] = (".frd", ".dat", ".cvg", ".sta", ".rout")

# Log file name suffixes
_STDOUT_LOG_SUFFIX: str = "_stdout.log"
_STDERR_LOG_SUFFIX: str = "_stderr.log"

# Maximum characters returned in log excerpts (avoid huge strings in memory)
_EXCERPT_MAX_CHARS: int = 2000


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class SolverRunError(Exception):
    """
    Raised when solver execution fails in an unrecoverable way.

    Soft failures (e.g. CalculiX not installed, non-zero return code) are
    represented by ``SolverRunResult.success = False`` and do not raise.
    Only callers using ``require_successful_solver_run()`` will raise.
    """


# ---------------------------------------------------------------------------
# Typed dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SolverRunOptions:
    """
    Configuration for a solver run.

    Attributes:
        run_solver:          If ``False`` (default), return export-only mode
                             result without executing CalculiX.
        solver_executable:   Name or full path of the CalculiX executable
                             (``"ccx"`` or ``"ccx_2.21"`` etc.).
        timeout_seconds:     Maximum wall-clock seconds to allow the solver
                             to run.  ``None`` means no timeout.
        capture_stdout:      Capture solver stdout (required for log writing).
        capture_stderr:      Capture solver stderr (required for log writing).
        write_stdout_log:    Write captured stdout to a ``.log`` file.
        write_stderr_log:    Write captured stderr to a ``.log`` file.
        overwrite_logs:      Overwrite existing log files if present.
        working_directory:   Override the working directory for the solver
                             process.  Defaults to the ``.inp`` parent dir.
        environment:         Optional environment variable overrides for the
                             solver subprocess.
    """

    run_solver: bool = False           # default: export-only
    solver_executable: str = "ccx"
    timeout_seconds: int | None = None
    capture_stdout: bool = True
    capture_stderr: bool = True
    write_stdout_log: bool = True
    write_stderr_log: bool = True
    overwrite_logs: bool = True
    working_directory: str | None = None
    environment: dict[str, str] | None = None


@dataclass
class SolverRunResult:
    """
    Structured result of a solver run attempt.

    Attributes:
        success:          True if the run was either skipped (export-only)
                          or executed successfully (return_code == 0).
        executed:         True only if CalculiX was actually invoked.
        solver_backend:   Always ``"calculix"`` in version 1.
        input_deck_path:  Path to the ``.inp`` file that was (or would be) run.
        working_directory: Directory where the solver was invoked.
        return_code:      CalculiX process return code (None if not executed).
        runtime_seconds:  Wall-clock time of solver execution (None if not run).
        stdout_log_path:  Path to the written stdout log file.
        stderr_log_path:  Path to the written stderr log file.
        stdout_excerpt:   First/last N chars of stdout (for quick diagnostics).
        stderr_excerpt:   First/last N chars of stderr.
        warnings:         Non-fatal warning strings.
        metadata:         Additional metadata for logging and reporting.
        error_message:    Error description if ``success`` is False.
    """

    success: bool
    executed: bool
    solver_backend: str = "calculix"
    input_deck_path: str = ""
    working_directory: str | None = None
    return_code: int | None = None
    runtime_seconds: float | None = None
    stdout_log_path: str | None = None
    stderr_log_path: str | None = None
    stdout_excerpt: str | None = None
    stderr_excerpt: str | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success":           self.success,
            "executed":          self.executed,
            "solver_backend":    self.solver_backend,
            "input_deck_path":   self.input_deck_path,
            "working_directory": self.working_directory,
            "return_code":       self.return_code,
            "runtime_seconds":   self.runtime_seconds,
            "stdout_log_path":   self.stdout_log_path,
            "stderr_log_path":   self.stderr_log_path,
            "stdout_excerpt":    self.stdout_excerpt,
            "stderr_excerpt":    self.stderr_excerpt,
            "warnings":          self.warnings,
            "metadata":          self.metadata,
            "error_message":     self.error_message,
        }


# ---------------------------------------------------------------------------
# Executable detection helpers
# ---------------------------------------------------------------------------

def find_solver_executable(executable_name: str = "ccx") -> str | None:
    """
    Locate the CalculiX executable using ``shutil.which``.

    Args:
        executable_name: Executable name or absolute path.  Defaults to
                         ``"ccx"`` (standard CalculiX binary name).

    Returns:
        Absolute path to the executable as a string, or ``None`` if not found.
    """
    return shutil.which(executable_name)


def solver_available(executable_name: str = "ccx") -> bool:
    """
    Return ``True`` if the CalculiX executable is findable on PATH.

    Args:
        executable_name: Name or path of the executable.

    Returns:
        ``True`` if found, ``False`` otherwise.
    """
    return find_solver_executable(executable_name) is not None


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_input_deck(input_deck_path: Path) -> None:
    """
    Validate that the input deck file is present, has a ``.inp`` extension,
    and is non-empty.

    Args:
        input_deck_path: Path to the ``.inp`` file.

    Raises:
        SolverRunError: with a descriptive message on any failure.
    """
    if not input_deck_path.exists():
        raise SolverRunError(
            f"Input deck not found: {input_deck_path}.  "
            f"Run the solver exporter before calling the runner."
        )
    if not input_deck_path.is_file():
        raise SolverRunError(
            f"Input deck path is not a file: {input_deck_path}."
        )
    if input_deck_path.suffix.lower() not in (".inp", ""):
        raise SolverRunError(
            f"Input deck has unexpected extension '{input_deck_path.suffix}'.  "
            f"Expected a CalculiX .inp file."
        )
    if input_deck_path.stat().st_size == 0:
        raise SolverRunError(
            f"Input deck is empty: {input_deck_path}.  "
            f"The solver exporter may have failed to write content."
        )


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _resolve_working_directory(
    input_deck_path: Path,
    options: SolverRunOptions,
) -> Path:
    """
    Determine the working directory for the solver process.

    If ``options.working_directory`` is set, that path is used.
    Otherwise the parent directory of the input deck is used, which is the
    standard CalculiX convention (solver writes result files alongside the
    input deck).

    Args:
        input_deck_path: Path to the ``.inp`` file.
        options:         Run options.

    Returns:
        Resolved ``Path`` for the working directory.
    """
    working_dir_str = options.working_directory
    if working_dir_str is not None and working_dir_str:
        wd = Path(working_dir_str).resolve()
    else:
        wd = input_deck_path.parent.resolve()

    wd.mkdir(parents=True, exist_ok=True)
    return wd


def _solver_job_name(input_deck_path: Path) -> str:
    """
    Return the CalculiX job name (``inp`` stem without extension).

    CalculiX is invoked as ``ccx <job_name>`` in the working directory.

    Args:
        input_deck_path: Path to the ``.inp`` file.

    Returns:
        Job name string (filename stem).
    """
    return input_deck_path.stem


def _stdout_log_path(
    input_deck_path: Path,
    working_directory: Path,
) -> Path:
    """Path for the stdout log file."""
    return working_directory / "logs" / (
        input_deck_path.stem + _STDOUT_LOG_SUFFIX
    )


def _stderr_log_path(
    input_deck_path: Path,
    working_directory: Path,
) -> Path:
    """Path for the stderr log file."""
    return working_directory / "logs" / (
        input_deck_path.stem + _STDERR_LOG_SUFFIX
    )


# ---------------------------------------------------------------------------
# Log helpers
# ---------------------------------------------------------------------------

def _write_text_log(path: Path, text: str, overwrite: bool = True) -> None:
    """
    Write a text string to a log file.

    Args:
        path:      Destination file path.
        text:      Text content to write.
        overwrite: If True, overwrite an existing file.  If False and the
                   file exists, skip writing and log a debug message.
    """
    if not overwrite and path.exists():
        logger.debug("Log file exists and overwrite=False; skipping: %s", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _excerpt(text: str, max_chars: int = _EXCERPT_MAX_CHARS) -> str:
    """
    Return a truncated excerpt of a text string for embedding in results.

    If the text is shorter than ``max_chars``, it is returned unchanged.
    Otherwise the first half and last half of the budget are returned
    with an ellipsis in the middle so both the start and end are visible.

    Args:
        text:      Source text.
        max_chars: Maximum total characters in the excerpt.

    Returns:
        Excerpt string.
    """
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    start_excerpt = "".join([text[i] for i in range(half)])
    l = len(text)
    end_excerpt = "".join([text[i] for i in range(l - half, l)])
    return start_excerpt + f"\n... [{l - max_chars} chars omitted] ...\n" + end_excerpt


# ---------------------------------------------------------------------------
# Config-driven default options
# ---------------------------------------------------------------------------

def default_run_options(
    project_root: str | Path | None = None,
) -> SolverRunOptions:
    """
    Build ``SolverRunOptions`` from ``base_config.yaml`` solver settings.

    Reads:
        - ``solver.run_solver_by_default``  → ``run_solver``
        - ``solver.solver_backend``         → used to confirm "calculix"
        - Fallback: ``run_solver=False``, ``solver_executable="ccx"``

    ARCHITECTURAL DECISION — default is export-only:
        ``run_solver_by_default: false`` is set in base_config.yaml.  This
        loader faithfully respects that setting rather than substituting a
        different default.  Users who want to execute the solver must either
        change the config or pass ``run_solver=True`` explicitly.

    Args:
        project_root: Project root for config resolution (optional).

    Returns:
        ``SolverRunOptions`` populated from config.
    """
    run_solver = False
    solver_executable = "ccx"

    try:
        from utils.config_loader import load_pipeline_config
        cfg = load_pipeline_config(project_root)
        solver_cfg = cfg.solver
        run_solver = bool(
            solver_cfg.get("run_solver_by_default", False)
        )
        # Confirm the backend is calculix; warn if not.
        backend = solver_cfg.get("solver_backend", "calculix").lower()
        if backend != "calculix":
            logger.warning(
                "base_config.yaml solver.solver_backend = '%s'; "
                "runner.py only supports CalculiX in version 1.",
                backend,
            )
    except Exception as exc:
        logger.warning(
            "Could not load solver config from base_config.yaml: %s.  "
            "Using default run options (run_solver=False).",
            exc,
        )

    return SolverRunOptions(
        run_solver=run_solver,
        solver_executable=solver_executable,
    )


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run_calculix_input_deck(
    input_deck_path: str | Path,
    options: SolverRunOptions | None = None,
) -> SolverRunResult:
    """
    Execute (or skip) a CalculiX job from an ``.inp`` input deck.

    Modes:
        ``options.run_solver = False`` (default):
            Return a successful ``SolverRunResult`` with ``executed=False``
            and a warning noting this is export-only mode.  No CalculiX
            process is launched.

        ``options.run_solver = True``:
            Attempt to find and run CalculiX.  If the executable is not
            found, return ``success=False`` with a clear error message.
            On successful execution (return_code == 0), return
            ``success=True``.  On non-zero return_code, return
            ``success=False`` with the return code and log excerpts.

    CalculiX invocation convention::

        cd <working_directory>
        ccx <job_basename>

    Args:
        input_deck_path: Path to the ``.inp`` file (written by solver_exporter).
        options:         Run options.  Defaults to ``SolverRunOptions()``
                         (export-only mode).

    Returns:
        ``SolverRunResult`` — always returned, never raises for soft failures.
    """
    input_deck_path = Path(input_deck_path)
    options = options or SolverRunOptions()

    result = SolverRunResult(
        success=False,
        executed=False,
        input_deck_path=str(input_deck_path),
    )

    # --- Validate input deck ---
    try:
        _validate_input_deck(input_deck_path)
    except SolverRunError as exc:
        result.error_message = str(exc)
        return result

    # =====================================================================
    # EXPORT-ONLY MODE
    # =====================================================================
    if not options.run_solver:
        result.success = True
        result.executed = False
        result.warnings.append(
            "Export-only mode: CalculiX was NOT executed.  "
            "Set SolverRunOptions.run_solver=True (or base_config.yaml "
            "solver.run_solver_by_default: true) to enable solver execution."
        )
        result.metadata["mode"] = "export_only"
        logger.info(
            "Solver runner: export-only mode for '%s'.", input_deck_path.name
        )
        return result

    # =====================================================================
    # EXECUTION MODE
    # =====================================================================

    # --- Detect executable ---
    solver_exe = find_solver_executable(options.solver_executable)
    if solver_exe is None:
        result.error_message = (
            f"CalculiX executable '{options.solver_executable}' not found on PATH.  "
            f"Install CalculiX or add it to PATH, or set "
            f"SolverRunOptions.run_solver=False to use export-only mode."
        )
        result.metadata["mode"] = "execution_failed_no_executable"
        return result

    logger.info(
        "Solver runner: found CalculiX at '%s'.", solver_exe
    )

    # --- Resolve working directory ---
    working_dir = _resolve_working_directory(input_deck_path, options)
    result.working_directory = str(working_dir)

    job_name = _solver_job_name(input_deck_path)
    stdout_log = _stdout_log_path(input_deck_path, working_dir)
    stderr_log = _stderr_log_path(input_deck_path, working_dir)

    # CalculiX command: ccx <job_basename>
    # (no .inp extension; CalculiX appends it automatically)
    cmd = [solver_exe, job_name]

    # Build subprocess environment
    env = dict(os.environ)
    env_overrides = options.environment
    if env_overrides is not None:
        env.update(env_overrides)

    logger.info(
        "Launching CalculiX: cmd=%s, cwd='%s', timeout=%s s.",
        cmd, working_dir, options.timeout_seconds,
    )

    # --- Execute ---
    t_start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(working_dir),
            capture_output=True,  # captures stdout and stderr
            text=True,
            timeout=options.timeout_seconds,
            env=env,
        )
        runtime = time.monotonic() - t_start
        result.executed = True
        result.return_code = proc.returncode
        result.runtime_seconds = float(f"{runtime:.3f}")

        stdout_text = proc.stdout or ""
        stderr_text = proc.stderr or ""

        # Write log files
        if options.write_stdout_log and stdout_text:
            try:
                _write_text_log(stdout_log, stdout_text, overwrite=options.overwrite_logs)
                result.stdout_log_path = str(stdout_log)
            except Exception as exc:
                result.warnings.append(
                    f"Failed to write stdout log '{stdout_log}': {exc}"
                )

        if options.write_stderr_log and stderr_text:
            try:
                _write_text_log(stderr_log, stderr_text, overwrite=options.overwrite_logs)
                result.stderr_log_path = str(stderr_log)
            except Exception as exc:
                result.warnings.append(
                    f"Failed to write stderr log '{stderr_log}': {exc}"
                )

        result.stdout_excerpt = _excerpt(stdout_text)
        result.stderr_excerpt = _excerpt(stderr_text)

        result.metadata["mode"] = "execution"
        result.metadata["solver_exe"] = solver_exe
        result.metadata["cmd"] = cmd

        if proc.returncode == 0:
            result.success = True
            logger.info(
                "CalculiX completed successfully: job='%s', "
                "runtime=%.2f s, return_code=0.",
                job_name, runtime,
            )
        else:
            result.success = False
            result.error_message = (
                f"CalculiX returned non-zero exit code {proc.returncode} "
                f"for job '{job_name}'.  Check stderr log for details."
            )
            logger.error(
                "CalculiX failed: job='%s', return_code=%d, runtime=%.2f s.",
                job_name, proc.returncode, runtime,
            )
            if stderr_text:
                logger.error(
                    "CalculiX stderr excerpt:\n%s",
                    _excerpt(stderr_text, 500),
                )

    except subprocess.TimeoutExpired:
        runtime = time.monotonic() - t_start
        result.executed = True
        result.runtime_seconds = float(f"{runtime:.3f}")
        result.return_code = None
        result.error_message = (
            f"CalculiX job '{job_name}' timed out after "
            f"{options.timeout_seconds} seconds."
        )
        result.metadata["mode"] = "timeout"
        logger.error(
            "CalculiX timed out: job='%s', timeout=%s s.",
            job_name, options.timeout_seconds,
        )

    except FileNotFoundError:
        result.error_message = (
            f"CalculiX executable not found when launching: '{solver_exe}'.  "
            f"This may indicate the PATH changed after initial detection."
        )
        result.metadata["mode"] = "execution_failed_file_not_found"

    except Exception as exc:
        runtime = time.monotonic() - t_start
        result.executed = True
        result.runtime_seconds = float(f"{runtime:.3f}")
        result.error_message = (
            f"Unexpected error running CalculiX for job '{job_name}': {exc}"
        )
        result.metadata["mode"] = "execution_failed_unexpected"
        logger.exception(
            "Unexpected error running CalculiX job '%s'.", job_name
        )

    return result


# ---------------------------------------------------------------------------
# Case convenience helper
# ---------------------------------------------------------------------------

def run_solver_for_case(
    case_definition: Any,
    input_deck_path: str | Path,
    options: SolverRunOptions | None = None,
) -> SolverRunResult:
    """
    Run the CalculiX solver for a specific pipeline case.

    ARCHITECTURAL DECISION — duck-typed on case_definition:
        Accesses ``.case_id`` for logging only.  ``CaseDefinition`` is not
        imported here to avoid cross-layer coupling.

    Args:
        case_definition:  Object with ``.case_id`` (e.g. ``CaseDefinition``).
        input_deck_path:  Path to the ``.inp`` file.
        options:          Run options (optional; uses ``SolverRunOptions()``
                          defaults if not provided).

    Returns:
        ``SolverRunResult``.
    """
    case_id = getattr(case_definition, "case_id", "unknown_case")
    logger.info(
        "Runner: dispatching case '%s', input='%s'.",
        case_id, input_deck_path,
    )

    result = run_calculix_input_deck(input_deck_path, options)
    result.metadata["case_id"] = case_id
    return result


# ---------------------------------------------------------------------------
# Hard-fail wrapper
# ---------------------------------------------------------------------------

def require_successful_solver_run(
    input_deck_path: str | Path,
    options: SolverRunOptions | None = None,
) -> SolverRunResult:
    """
    Run CalculiX and raise ``SolverRunError`` if execution fails.

    NOTE: In export-only mode (``options.run_solver = False``) this function
    does NOT raise — a skipped run is considered a non-failure.  Use
    ``options.run_solver = True`` if you need to assert the solver actually ran.

    ARCHITECTURAL DECISION — export-only is not a failure:
        A deliberate skip (run_solver=False) is a valid workflow state in
        version 1.  Raising on a skip would break the standard pipeline flow
        where most runs are export-only.  Callers that need to assert
        execution occurred should check ``result.executed`` after calling
        ``run_calculix_input_deck`` directly.

    Args:
        input_deck_path: Path to the ``.inp`` file.
        options:         Run options.

    Returns:
        ``SolverRunResult`` with ``success=True``.

    Raises:
        SolverRunError: if ``run_solver=True`` and execution failed.
    """
    result = run_calculix_input_deck(input_deck_path, options)

    if not result.success and result.executed:
        raise SolverRunError(
            f"CalculiX execution failed for '{input_deck_path}': "
            f"{result.error_message}"
        )

    return result
