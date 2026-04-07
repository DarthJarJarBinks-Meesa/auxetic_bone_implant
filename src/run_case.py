"""
src/run_case.py
================
Single-case CLI entrypoint for the auxetic plate pipeline.

Allows a user to run exactly one pipeline case from the command line by
specifying either a case ID (looked up from the generated case list) or a
case-definition JSON file.  All heavy lifting is delegated to
``src/workflow/case_runner.py``.

USAGE EXAMPLES::

    # Run by case ID (looks up from generated case list)
    python src/run_case.py --case-id reentrant_pt2p5_ti64_axialcompression_a1b2c3d4

    # Run by JSON file
    python src/run_case.py --case-file /path/to/case_config.json

    # Run with solver execution enabled and a custom meshing preset
    python src/run_case.py --case-id <id> --run-solver --meshing-preset fine

    # Print pure JSON output (machine-readable)
    python src/run_case.py --case-id <id> --json

    # Specify an explicit project root
    python src/run_case.py --case-id <id> --project-root /path/to/project

EXIT CODES:
    0 — case completed successfully
    1 — case failed or a CLI/setup error occurred

SCRIPT POSITION:
    User / CLI  →  [THIS SCRIPT]  →  case_runner.run_case(...)
                                   →  stdout summary
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class RunCaseCliError(Exception):
    """
    Raised for unrecoverable CLI-level errors before case execution begins.

    Examples:
      - both ``--case-id`` and ``--case-file`` were provided
      - neither selection argument was provided
      - the requested case ID was not found in the generated case list
      - the supplied ``--case-file`` is missing or malformed
    """


# ---------------------------------------------------------------------------
# Case-loading helpers
# ---------------------------------------------------------------------------

def load_case_from_json(path: str | Path) -> Any:
    """
    Load a ``CaseDefinition`` from a JSON file written by the pipeline.

    The JSON file is expected to be the output of
    ``CaseDefinition.to_dict()`` (e.g. the ``case_config.json`` written by
    ``case_runner._write_case_config_snapshot``).

    Args:
        path: Path to the JSON file.

    Returns:
        Reconstructed ``CaseDefinition`` object.

    Raises:
        RunCaseCliError: if the file does not exist, cannot be parsed, or
                         cannot be converted to a ``CaseDefinition``.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise RunCaseCliError(
            f"Case JSON file not found: '{file_path}'.  "
            "Check the path and try again."
        )

    try:
        with file_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        raise RunCaseCliError(
            f"Could not parse case JSON file '{file_path}': {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise RunCaseCliError(
            f"Case JSON file '{file_path}' does not contain a top-level object."
        )

    # Attempt reconstruction via schema helpers
    try:
        from workflow.case_schema import CaseDefinition
        cd = CaseDefinition.from_dict(data)
        return cd
    except Exception as exc:
        raise RunCaseCliError(
            f"Could not reconstruct CaseDefinition from '{file_path}': {exc}.  "
            "Ensure the file was produced by CaseDefinition.to_dict()."
        ) from exc


def find_generated_case_by_id(
    case_id: str,
    project_root: str | Path | None = None,
) -> Any:
    """
    Find a case definition by case ID from the generated case list.

    Calls ``generate_cases`` with the default baseline-plus-one-factor mode
    (the most complete non-factorial set) and searches for a matching
    ``case_id``.  If not found there, falls back to full-factorial generation
    with a modest ``max_case_count`` guard.

    Args:
        case_id:      Exact case ID string to look for.
        project_root: Project root for config resolution.

    Returns:
        Matching ``CaseDefinition``.

    Raises:
        RunCaseCliError: if the case ID is not found or generation fails.
    """
    from generate_cases import (
        generate_cases,
        CaseGenerationOptions,
        CaseGenerationError,
    )

    # First pass: default mode (covers the most common case IDs)
    for mode, first_pass in [
        ("baseline_plus_one_factor_variation", False),
        ("baseline_plus_one_factor_variation", True),
        ("full_factorial", True),
    ]:
        try:
            opts = CaseGenerationOptions(
                mode=mode,
                use_first_pass_values=first_pass,
                max_case_count=2000,
                sort_cases=False,
            )
            result = generate_cases(options=opts, project_root=project_root)
        except CaseGenerationError as exc:
            raise RunCaseCliError(
                f"Case generation failed while searching for '{case_id}': {exc}"
            ) from exc

        for cd in result.cases:
            if cd.case_id == case_id:
                return cd

    raise RunCaseCliError(
        f"Case ID '{case_id}' was not found in any generated case list.  "
        "Possible causes:\n"
        "  • The case ID is misspelled or has been regenerated with different params.\n"
        "  • The design/material/load-case referenced in the case ID is disabled.\n"
        "  • Use '--case-file' to supply a case definition JSON directly.\n"
        "  • Run 'python src/main.py --list-cases' to see available case IDs."
    )


def resolve_case_definition(args: argparse.Namespace) -> Any:
    """
    Resolve the target ``CaseDefinition`` from CLI arguments.

    Exactly one of ``--case-id`` or ``--case-file`` must be provided.

    Args:
        args: Parsed ``argparse.Namespace``.

    Returns:
        ``CaseDefinition`` ready for execution.

    Raises:
        RunCaseCliError: if argument constraints are violated or resolution fails.
    """
    has_id   = bool(getattr(args, "case_id",   None))
    has_file = bool(getattr(args, "case_file", None))

    if has_id and has_file:
        raise RunCaseCliError(
            "Provide either '--case-id' or '--case-file', not both."
        )
    if not has_id and not has_file:
        raise RunCaseCliError(
            "No case selected.  Use '--case-id <id>' or '--case-file <path>'."
        )

    if has_file:
        return load_case_from_json(args.case_file)

    project_root = getattr(args, "project_root", None)
    return find_generated_case_by_id(args.case_id, project_root=project_root)


# ---------------------------------------------------------------------------
# Option-building helper
# ---------------------------------------------------------------------------

def build_case_run_options_from_args(args: argparse.Namespace) -> Any:
    """
    Build a ``CaseRunOptions`` object from parsed CLI arguments.

    Maps:
        ``--run-solver``                  → ``run_solver``
        ``--meshing-preset``              → ``meshing_preset``
        ``--no-skip-completed``           → ``skip_completed_cases = False``
        ``--uncentered-extrusion``        → ``centered_extrusion = False``
        ``--strict-validation``           → ``strict_validation``
        ``--minimum-feature-size-mm``     → ``minimum_feature_size_mm``
        ``--solver-timeout-seconds``      → ``solver_timeout_seconds``
        ``--no-write-intermediate-geometry`` → ``write_intermediate_geometry = False``
        ``--no-write-mesh-files``         → ``write_mesh_files = False``
        ``--no-write-solver-input-files`` → ``write_solver_input_files = False``

    Falls back to ``default_case_run_options`` for values not supplied on the
    command line, so ``base_config.yaml`` remains the authoritative defaults.

    Args:
        args: Parsed ``argparse.Namespace``.

    Returns:
        ``CaseRunOptions`` instance.
    """
    from workflow.case_runner import default_case_run_options

    project_root = getattr(args, "project_root", None)
    opts = default_case_run_options(project_root)

    if getattr(args, "run_solver", False):
        opts.run_solver = True

    preset = getattr(args, "meshing_preset", None)
    if preset:
        opts.meshing_preset = preset

    if getattr(args, "no_skip_completed", False):
        opts.skip_completed_cases = False

    if getattr(args, "uncentered_extrusion", False):
        opts.centered_extrusion = False

    if getattr(args, "strict_validation", False):
        opts.strict_validation = True

    feat_size = getattr(args, "minimum_feature_size_mm", None)
    if feat_size is not None:
        opts.minimum_feature_size_mm = float(feat_size)

    timeout = getattr(args, "solver_timeout_seconds", None)
    if timeout is not None:
        opts.solver_timeout_seconds = int(timeout)

    if getattr(args, "no_write_intermediate_geometry", False):
        opts.write_intermediate_geometry = False

    if getattr(args, "no_write_mesh_files", False):
        opts.write_mesh_files = False

    if getattr(args, "no_write_solver_input_files", False):
        opts.write_solver_input_files = False

    return opts


# ---------------------------------------------------------------------------
# CLI parser builder
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build and return the CLI argument parser for ``run_case.py``.

    Returns:
        Configured ``argparse.ArgumentParser``.
    """
    parser = argparse.ArgumentParser(
        prog="run_case",
        description=(
            "Run a single auxetic plate pipeline case.\n\n"
            "Select a case with --case-id (looked up from generated cases)\n"
            "or --case-file (a case_config.json produced by a prior run).\n\n"
            "Example:\n"
            "  python src/run_case.py --case-id reentrant_pt2p5_ti64_axialcomp_a1b2c3d4\n"
            "  python src/run_case.py --case-file runs/my_case/case_config.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Case selection (mutually exclusive) ---
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--case-id",
        metavar="CASE_ID",
        help=(
            "Case ID to run.  Must match a case from the generated case list.  "
            "Run 'python src/main.py --list-cases' to see available IDs."
        ),
    )
    selection.add_argument(
        "--case-file",
        metavar="PATH",
        help=(
            "Path to a case_config.json file (produced by a prior case run or "
            "by generate_cases).  Bypasses case-list lookup."
        ),
    )

    # --- Project root ---
    parser.add_argument(
        "--project-root",
        metavar="DIR",
        default=None,
        help=(
            "Path to the project root directory.  Defaults to auto-detection "
            "based on the location of this script."
        ),
    )

    # --- Solver options ---
    parser.add_argument(
        "--run-solver",
        action="store_true",
        default=False,
        help=(
            "Execute CalculiX after exporting the .inp input deck.  "
            "Default: export only (no solver execution)."
        ),
    )
    parser.add_argument(
        "--solver-timeout-seconds",
        type=int,
        metavar="N",
        default=None,
        help="Wall-clock timeout for CalculiX in seconds (default: no limit).",
    )

    # --- Meshing options ---
    parser.add_argument(
        "--meshing-preset",
        metavar="PRESET",
        default=None,
        help=(
            "gmsh meshing preset to use (e.g. 'default', 'coarse', 'fine').  "
            "Default: taken from meshing.yaml."
        ),
    )

    # --- Geometry options ---
    parser.add_argument(
        "--uncentered-extrusion",
        action="store_true",
        default=False,
        help=(
            "Extrude the solid starting from Z=0 rather than centred about Z=0.  "
            "Default: centred extrusion."
        ),
    )
    parser.add_argument(
        "--no-write-intermediate-geometry",
        action="store_true",
        default=False,
        help=(
            "Skip optional STEP export of 2D unit-cell and lattice geometry.  "
            "The final 3D solid STEP is still exported."
        ),
    )

    # --- Validation options ---
    parser.add_argument(
        "--strict-validation",
        action="store_true",
        default=False,
        help=(
            "Treat validation warnings as hard failures.  "
            "Default: warnings are logged but do not abort execution."
        ),
    )
    parser.add_argument(
        "--minimum-feature-size-mm",
        type=float,
        metavar="MM",
        default=None,
        help=(
            "Minimum feature size threshold in mm for geometry validation.  "
            "Default: taken from base_config.yaml."
        ),
    )

    # --- Skip / rerun control ---
    parser.add_argument(
        "--no-skip-completed",
        action="store_true",
        default=False,
        help=(
            "Force re-execution even if the case is already marked COMPLETED.  "
            "Default: skip completed cases."
        ),
    )

    # --- Artifact control ---
    parser.add_argument(
        "--no-write-mesh-files",
        action="store_true",
        default=False,
        help="Skip writing mesh output files.",
    )
    parser.add_argument(
        "--no-write-solver-input-files",
        action="store_true",
        default=False,
        help="Skip writing the CalculiX .inp input deck.",
    )

    # --- Output format ---
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        dest="json_output",
        help="Print machine-readable JSON output instead of a human summary.",
    )

    # --- Verbosity ---
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )

    return parser


# ---------------------------------------------------------------------------
# Summary formatting helpers
# ---------------------------------------------------------------------------

def case_run_summary_to_console_dict(summary: Any) -> dict[str, Any]:
    """
    Extract a compact, human-useful dict from a ``CaseRunSummary``.

    Args:
        summary: ``CaseRunSummary`` object.

    Returns:
        Dict with the most useful fields for console display.
    """
    cr = summary.case_result

    # Extract key metrics (best-effort — metric_set fields may be None)
    metrics: dict[str, Any] = {}
    ms = getattr(cr, "metrics", None)
    if ms is not None:
        for attr in (
            "effective_stiffness_n_per_mm",
            "effective_modulus_proxy_mpa",
            "max_von_mises_stress_mpa",
            "min_safety_factor",
            "fatigue_risk_score",
            "displacement_at_load_mm",
        ):
            val = getattr(ms, attr, None)
            if val is not None:
                v = float(val)
                metrics[attr] = float(f"{v:.4f}")

    # Build artifact summary (only paths that exist)
    artifacts: dict[str, str] = {}
    art = summary.artifacts
    for label, attr in (
        ("solid_step",   "solid_geometry_file"),
        ("mesh",         "mesh_file"),
        ("solver_inp",   "solver_input_file"),
        ("run_dir",      "run_directory"),
    ):
        val = getattr(art, attr, None)
        if val:
            artifacts[label] = val

    result: dict[str, Any] = {
        "case_id":         getattr(cr, "case_id", "unknown"),
        "success":         summary.success,
        "status":          getattr(cr, "status", "?"),
        "runtime_seconds": getattr(cr, "runtime_seconds", None),
        "metrics":         metrics or None,
        "artifacts":       artifacts or None,
    }

    if summary.error_message:
        result["error"] = summary.error_message

    if summary.warnings:
        result["warning_count"] = len(summary.warnings)
        # Show first 5 warnings to avoid overwhelming output
        result["warnings_preview"] = summary.warnings[:5]
        if len(summary.warnings) > 5:
            result["warnings_preview"].append(
                f"… and {len(summary.warnings) - 5} more."
            )

    return result


def print_human_summary(summary: Any) -> None:
    """
    Print a compact, human-readable case run summary to stdout.

    Args:
        summary: ``CaseRunSummary`` object.
    """
    d = case_run_summary_to_console_dict(summary)

    status_icon = "✓" if d["success"] else "✗"
    rt = d.get("runtime_seconds")
    rt_str = f" ({rt:.1f} s)" if rt is not None else ""

    print(f"\n{'═' * 60}")
    print(f"  run_case — {status_icon} {d['status'].upper()}{rt_str}")
    print(f"{'═' * 60}")
    print(f"  case_id : {d['case_id']}")
    print(f"  success : {d['success']}")

    if d.get("error"):
        print(f"  error   : {d['error']}")

    if d.get("metrics"):
        print("  metrics :")
        for k, v in d["metrics"].items():
            print(f"    {k}: {v}")

    if d.get("artifacts"):
        print("  artifacts :")
        for k, v in d["artifacts"].items():
            print(f"    {k}: {v}")

    if d.get("warning_count"):
        print(f"  warnings: {d['warning_count']} (showing first {min(5, d['warning_count'])})")
        for w in d.get("warnings_preview", []):
            print(f"    ⚠  {w}")

    print(f"{'─' * 60}\n")


def print_json_summary(summary: Any) -> None:
    """
    Print a machine-readable JSON case run summary to stdout.

    Args:
        summary: ``CaseRunSummary`` object.
    """
    d = case_run_summary_to_console_dict(summary)
    # Also include status as a raw string for machine parsing
    status_val = d.get("status")
    status_value = getattr(status_val, "value", None)
    if status_value is not None:
        d["status"] = status_value

    print(json.dumps(d, indent=2, default=str))


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """
    Main entrypoint for the single-case CLI.

    Args:
        argv: Argument list (uses ``sys.argv[1:]`` if None).

    Returns:
        Exit code — ``0`` on success, ``1`` on failure.
    """
    parser = build_arg_parser()
    args   = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Configure logging
    # ------------------------------------------------------------------
    log_level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s | %(name)s | %(message)s",
        stream=sys.stderr,
    )

    # ------------------------------------------------------------------
    # Validate project root
    # ------------------------------------------------------------------
    project_root: Path | None = None
    if getattr(args, "project_root", None):
        project_root = Path(args.project_root).resolve()
        if not project_root.is_dir():
            print(
                f"ERROR: --project-root '{project_root}' is not a directory.",
                file=sys.stderr,
            )
            return 1

    # ------------------------------------------------------------------
    # Resolve case definition
    # ------------------------------------------------------------------
    try:
        case_definition = resolve_case_definition(args)
    except RunCaseCliError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(
            f"ERROR: unexpected error while resolving case: {exc}",
            file=sys.stderr,
        )
        logger.exception("Unexpected error in resolve_case_definition.")
        return 1

    logger.info("Resolved case: '%s'.", case_definition.case_id)

    # ------------------------------------------------------------------
    # Build run options
    # ------------------------------------------------------------------
    try:
        options = build_case_run_options_from_args(args)
    except Exception as exc:
        print(
            f"ERROR: could not build run options: {exc}",
            file=sys.stderr,
        )
        return 1

    # ------------------------------------------------------------------
    # Execute case
    # ------------------------------------------------------------------
    summary = None
    try:
        from workflow.case_runner import run_case
        summary = run_case(
            case_definition=case_definition,
            options=options,
            project_root=project_root,
        )
    except Exception as exc:
        print(
            f"ERROR: unhandled exception during case execution: {exc}",
            file=sys.stderr,
        )
        logger.exception("Unhandled exception in run_case.")
        return 1

    # ------------------------------------------------------------------
    # Print output
    # ------------------------------------------------------------------
    if getattr(args, "json_output", False):
        print_json_summary(summary)
    else:
        print_human_summary(summary)

    # ------------------------------------------------------------------
    # Exit code
    # ------------------------------------------------------------------
    return 0 if summary.success else 1


# ---------------------------------------------------------------------------
# Script guard
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(main())
