"""
src/main.py
===========
Top-level entrypoint for the auxetic plate pipeline.

Provides a unified CLI for:
  - generating cases via ``src/generate_cases.py``
  - filtering by stage or sweeping mode
  - orchestrating the sweep via ``src/workflow/orchestrator.py``
  - generating plans without running execution (``--plan-only``)
  - viewing available cases (``--list-cases``)

USAGE EXAMPLES::

    # Generate and execute the baseline + one-factor sweep with defaults
    python src/main.py

    # Show the case generation plan without running anything
    python src/main.py --plan-only

    # Run the stage-1 structural screen
    python src/main.py --stage stage_1

    # Run a full factorial sweep (may be large!)
    python src/main.py --mode full_factorial

    # Run without skipping already-completed cases
    python src/main.py --no-skip-completed

    # Print pure JSON output (machine-readable summary)
    python src/main.py --json

EXIT CODES:
    0 — sweep completed successfully
    1 — sweep failed or a CLI/setup error occurred

SCRIPT POSITION:
    User / CLI  →  [THIS SCRIPT]  →  generate_cases(...)
                                  →  orchestrator.run_cases(...)
                                  →  stdout summary
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, List, Optional, Union

from generate_cases import (
    CaseGenerationError,
    CaseGenerationOptions,
    GeneratedCaseSet,
    generate_cases,
)
from workflow.case_runner import CaseRunOptions
from workflow.orchestrator import (
    OrchestratorError,
    OrchestratorOptions,
    OrchestratorSummary,
    default_orchestrator_options,
    run_cases,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class MainCliError(Exception):
    """
    Raised for unrecoverable errors during top-level CLI setup or generation.

    Examples:
      - malformed CLI arguments
      - empty case list generated
      - invalid stage or mode names
    """


# ---------------------------------------------------------------------------
# CLI parser builder
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build and return the main CLI argument parser.

    Returns:
        Configured ``argparse.ArgumentParser``.
    """
    parser = argparse.ArgumentParser(
        prog="main",
        description=(
            "Run a multi-case parameter sweep for the auxetic plate pipeline.\n\n"
            "By default, reads sweep_config.yaml, generates the baseline + "
            "one-factor variation case set, and runs them sequentially.\n"
            "Use '--plan-only' to see what cases will be generated without "
            "running the solver pipeline."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Mode & Filtering ---
    parser.add_argument(
        "--mode",
        choices=[
            "baseline_only",
            "baseline_plus_one_factor_variation",
            "full_factorial",
        ],
        default=None,
        help=(
            "Case generation mode (default: from sweep_config.yaml, usually "
            "'baseline_plus_one_factor_variation')."
        ),
    )
    parser.add_argument(
        "--stage",
        metavar="STAGE_NAME",
        default=None,
        help=(
            "Run a specific stage defined in sweep_config.yaml "
            "(e.g. 'stage_1', 'geometry_and_coarse_screen')."
        ),
    )
    parser.add_argument(
        "--first-pass",
        action="store_true",
        default=False,
        help="Force use of 'first_pass_parameters' for geometric sweeps.",
    )
    parser.add_argument(
        "--max-case-count",
        type=int,
        metavar="N",
        default=None,
        help="Hard limit on the number of cases to generate.",
    )

    # --- Execution control ---
    parser.add_argument(
        "--plan-only",
        action="store_true",
        default=False,
        help="Generate cases and print the summary, but do not run them.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        default=False,
        help="Generate cases and print a raw list of case IDs, then exit.",
    )
    parser.add_argument(
        "--project-root",
        metavar="DIR",
        default=None,
        help="Path to project root (default: auto-detected).",
    )

    # --- Orchestrator overrides ---
    parser.add_argument(
        "--no-skip-completed",
        action="store_true",
        default=False,
        help="Force re-execution of cases even if they are marked COMPLETED.",
    )
    parser.add_argument(
        "--stop-on-first-failure",
        action="store_true",
        default=False,
        help="Abort the sweep immediately if any case fails.",
    )
    parser.add_argument(
        "--run-solver",
        action="store_true",
        default=False,
        help="Execute CalculiX for each case (default: export-only).",
    )
    parser.add_argument(
        "--meshing-preset",
        metavar="PRESET",
        default=None,
        help="gmsh meshing preset to use (e.g. 'default', 'coarse', 'fine').",
    )
    parser.add_argument(
        "--strict-validation",
        action="store_true",
        default=False,
        help="Treat geometric validation warnings as hard failures.",
    )
    parser.add_argument(
        "--minimum-feature-size-mm",
        type=float,
        metavar="MM",
        default=None,
        help="Override minimum feature size constraint for validation.",
    )
    parser.add_argument(
        "--solver-timeout-seconds",
        type=int,
        metavar="N",
        default=None,
        help="Wall-clock timeout for each solver run.",
    )

    # --- Reporting / Ranking overrides ---
    parser.add_argument(
        "--no-ranking",
        action="store_true",
        default=False,
        help="Disable ranking analysis at the end of the sweep.",
    )
    parser.add_argument(
        "--no-reporting",
        action="store_true",
        default=False,
        help="Disable CSV and plot report generation at the end of the sweep.",
    )
    parser.add_argument(
        "--reports-dir",
        metavar="DIR",
        default=None,
        help="Directory to write reports into (default: <project_root>/reports).",
    )
    parser.add_argument(
        "--top-n-filtered",
        type=int,
        metavar="N",
        default=None,
        help="Number of top candidates to include in the filtered report.",
    )

    # --- Output format ---
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        dest="json_output",
        help="Print machine-readable JSON output instead of human summaries.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )

    return parser


# ---------------------------------------------------------------------------
# Option-building helpers
# ---------------------------------------------------------------------------

def build_case_generation_options_from_args(
    args: argparse.Namespace,
) -> CaseGenerationOptions:
    """
    Map CLI arguments into ``CaseGenerationOptions``.

    Args:
        args: Parsed CLI namespace.

    Returns:
        Configured generation options.
    """
    opts = CaseGenerationOptions()

    if getattr(args, "mode", None):
        opts.mode = args.mode

    if getattr(args, "stage", None):
        opts.stage_name = args.stage

    if getattr(args, "first_pass", False):
        opts.use_first_pass_values = True

    if getattr(args, "max_case_count", None) is not None:
        opts.max_case_count = args.max_case_count

    return opts


def build_orchestrator_options_from_args(
    args: argparse.Namespace,
    project_root: Optional[Path],
) -> OrchestratorOptions:
    """
    Map CLI arguments into ``OrchestratorOptions``.

    Loads base defaults from config, then applies CLI overrides.

    Args:
        args:         Parsed CLI namespace.
        project_root: Resolved project root.

    Returns:
        Configured orchestration options.
    """
    opts = default_orchestrator_options(project_root)

    if getattr(args, "no_skip_completed", False):
        opts.skip_completed_cases = False

    if getattr(args, "stop_on_first_failure", False):
        opts.stop_on_first_failure = True

    if getattr(args, "no_ranking", False):
        opts.run_ranking = False

    if getattr(args, "no_reporting", False):
        opts.run_reporting = False

    if getattr(args, "reports_dir", None):
        opts.reports_dir = args.reports_dir

    if getattr(args, "top_n_filtered", None) is not None:
        opts.top_n_filtered = args.top_n_filtered

    if getattr(args, "stage", None):
        opts.stage_name = args.stage

    # --- Nested CaseRunOptions overrides ---
    cro = opts.case_run_options or CaseRunOptions()

    cro.skip_completed_cases = opts.skip_completed_cases

    if getattr(args, "run_solver", False):
        cro.run_solver = True

    preset = getattr(args, "meshing_preset", None)
    if preset:
        cro.meshing_preset = preset

    if getattr(args, "strict_validation", False):
        cro.strict_validation = True

    feat_size = getattr(args, "minimum_feature_size_mm", None)
    if feat_size is not None:
        cro.minimum_feature_size_mm = float(feat_size)

    timeout = getattr(args, "solver_timeout_seconds", None)
    if timeout is not None:
        cro.solver_timeout_seconds = int(timeout)

    opts.case_run_options = cro
    return opts


# ---------------------------------------------------------------------------
# Plan-summary formatting helpers
# ---------------------------------------------------------------------------

def generated_case_set_to_plan_dict(
    generated_case_set: GeneratedCaseSet,
) -> dict[str, Any]:
    """
    Convert a ``GeneratedCaseSet`` into a summary dict for console output.

    Args:
        generated_case_set: Output from ``generate_cases``.

    Returns:
        Summary dict suitable for JSON serialization or human printing.
    """
    cases = generated_case_set.cases
    meta  = generated_case_set.metadata

    preview = [c.case_id for c in cases[:10]]
    if len(cases) > 10:
        preview.append(f"... and {len(cases) - 10} more.")

    return {
        "status":        "plan-only",
        "total_cases":   len(cases),
        "mode":          meta.get("mode"),
        "stage":         meta.get("stage_name"),
        "counts": {
            "by_design":   meta.get("by_design", {}),
            "by_material": meta.get("by_material", {}),
            "by_loadcase": meta.get("by_load_case", {}),
        },
        "case_preview":  preview,
        "warnings":      generated_case_set.warnings,
    }


def print_plan_human(generated_case_set: GeneratedCaseSet) -> None:
    """Print the case generation plan in a human-readable format."""
    plan = generated_case_set_to_plan_dict(generated_case_set)
    print(f"\n{'═' * 60}")
    print("  CASE GENERATION PLAN (no execution)")
    print(f"{'═' * 60}")
    print(f"  total cases : {plan['total_cases']}")
    print(f"  mode        : {plan['mode']}")
    if plan["stage"]:
        print(f"  stage       : {plan['stage']}")

    print("\n  Counts by Design:")
    for k, v in plan["counts"]["by_design"].items():
        print(f"    {k}: {v}")

    print("\n  Counts by Material:")
    for k, v in plan["counts"]["by_material"].items():
        print(f"    {k}: {v}")

    print("\n  Counts by Load Case:")
    for k, v in plan["counts"]["by_loadcase"].items():
        print(f"    {k}: {v}")

    print("\n  Case IDs (preview):")
    for cid in plan["case_preview"]:
        line = f"    {cid}" if not cid.startswith("...") else f"    {cid}"
        print(line)

    if plan["warnings"]:
        print(f"\n  Warnings ({len(plan['warnings'])}):")
        for w in plan["warnings"]:
            print(f"    ⚠  {w}")

    print(f"{'─' * 60}\n")


def print_plan_json(generated_case_set: GeneratedCaseSet) -> None:
    """Print the case generation plan as machine-readable JSON."""
    plan = generated_case_set_to_plan_dict(generated_case_set)
    print(json.dumps(plan, indent=2))


# ---------------------------------------------------------------------------
# Orchestration-summary formatting helpers
# ---------------------------------------------------------------------------

def orchestrator_summary_to_console_dict(
    summary: OrchestratorSummary,
) -> dict[str, Any]:
    """
    Extract a compact, human-useful dict from an ``OrchestratorSummary``.

    Args:
        summary: Result of ``run_cases``.

    Returns:
        Summary dict suitable for JSON serialization or human printing.
    """
    result = {
        "success":               summary.success,
        "total_cases":           summary.total_cases,
        "completed_cases":       summary.completed_cases,
        "failed_cases":          summary.failed_cases,
        "skipped_cases":         summary.skipped_cases,
        "total_sweep_seconds":   summary.metadata.get("total_sweep_seconds"),
        "ran_ranking":           summary.ranking_result is not None,
        "ran_reporting":         summary.reporting_result is not None,
        "error_message":         summary.error_message,
        "warnings":              summary.warnings,
    }

    if summary.ranking_result:
        try:
            ranks = summary.ranking_result.ranked_cases
            if ranks:
                top = ranks[0]
                result["top_ranked_case_id"] = top.case_id
                result["top_ranked_score"]   = round(top.normalized_score, 4)
        except Exception:
            pass

    return result


def print_human_summary(summary: OrchestratorSummary) -> None:
    """Print the orchestration run summary in a human-readable format."""
    d = orchestrator_summary_to_console_dict(summary)
    status_icon = "✓" if d["success"] else "✗"
    rt = d.get("total_sweep_seconds")
    rt_str = f" ({rt:.1f} s)" if rt is not None else ""

    print(f"\n{'═' * 60}")
    print(f"  SWEEP COMPLETE — {status_icon} {'SUCCESS' if d['success'] else 'FAILED'}{rt_str}")
    print(f"{'═' * 60}")
    print(f"  total cases       : {d['total_cases']}")
    print(f"  completed         : {d['completed_cases']}")
    print(f"  failed            : {d['failed_cases']}")
    print(f"  skipped           : {d['skipped_cases']}")
    print(f"  ranking           : {d['ran_ranking']}")
    print(f"  reporting         : {d['ran_reporting']}")

    if d.get("top_ranked_case_id"):
        print(f"  top ranked case   : {d['top_ranked_case_id']} "
              f"(score: {d.get('top_ranked_score', '?')})")

    if d.get("error_message"):
        print(f"  ERROR             : {d['error_message']}")

    if d["warnings"]:
        print(f"  warnings          : {len(d['warnings'])}")
        # Show up to 10 warnings
        for w in d["warnings"][:10]:
            print(f"    ⚠  {w}")
        if len(d["warnings"]) > 10:
            print(f"    ⚠  ... and {len(d['warnings']) - 10} more.")

    print(f"{'─' * 60}\n")


def print_json_summary(summary: OrchestratorSummary) -> None:
    """Print the orchestration run summary as machine-readable JSON."""
    # Ensure full nested output is JSON serializable
    print(json.dumps(summary.to_dict(), indent=2))


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    """
    Main entrypoint for the top-level pipeline CLI.

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

    use_json = getattr(args, "json_output", False)

    # ------------------------------------------------------------------
    # Test config load
    # ------------------------------------------------------------------
    try:
        from utils.config_loader import load_pipeline_config
        _ = load_pipeline_config(project_root)
    except Exception as exc:
        msg = f"ERROR: Could not load pipeline configuration: {exc}"
        if use_json:
            print(json.dumps({"success": False, "error": msg}))
        else:
            print(msg, file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Step 1: Generate cases
    # ------------------------------------------------------------------
    try:
        gen_opts = build_case_generation_options_from_args(args)
        case_set = generate_cases(options=gen_opts, project_root=project_root)
    except CaseGenerationError as exc:
        msg = f"ERROR: Case generation failed: {exc}"
        if use_json:
            print(json.dumps({"success": False, "error": msg}))
        else:
            print(msg, file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error during case generation.")
        msg = f"ERROR: Unexpected error during case generation: {exc}"
        if use_json:
            print(json.dumps({"success": False, "error": msg}))
        else:
            print(msg, file=sys.stderr)
        return 1

    if not case_set.cases:
        msg = "ERROR: No cases were generated based on current config/filters."
        if use_json:
            print(json.dumps({"success": False, "error": msg, "warnings": case_set.warnings}))
        else:
            print(msg, file=sys.stderr)
            if case_set.warnings:
                print("Warnings generated during run:", file=sys.stderr)
                for w in case_set.warnings:
                    print(f"  - {w}", file=sys.stderr)
            logger.warning(
                "Zero cases generated.  Check sweep_config.yaml enabled options "
                "or your --stage filter."
            )
        return 1

    # ------------------------------------------------------------------
    # Step 2: Handle --list-cases or --plan-only
    # ------------------------------------------------------------------
    if getattr(args, "list_cases", False):
        if use_json:
            print(json.dumps([c.case_id for c in case_set.cases]))
        else:
            for c in case_set.cases:
                print(c.case_id)
        return 0

    if getattr(args, "plan_only", False):
        if use_json:
            print_plan_json(case_set)
        else:
            print_plan_human(case_set)
        return 0

    # ------------------------------------------------------------------
    # Step 3: Run orchestrator
    # ------------------------------------------------------------------
    try:
        orch_opts = build_orchestrator_options_from_args(args, project_root)
        summary   = run_cases(
            case_definitions=case_set.cases,
            options=orch_opts,
            project_root=project_root,
        )
    except OrchestratorError as exc:
        msg = f"ERROR: Orchestration failed: {exc}"
        if use_json:
            print(json.dumps({"success": False, "error": msg}))
        else:
            print(msg, file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error during orchestration.")
        msg = f"ERROR: Unexpected error during orchestration: {exc}"
        if use_json:
            print(json.dumps({"success": False, "error": msg}))
        else:
            print(msg, file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Step 4: Output and exit
    # ------------------------------------------------------------------
    if use_json:
        print_json_summary(summary)
    else:
        print_human_summary(summary)

    return 0 if summary.success else 1


# ---------------------------------------------------------------------------
# Script guard
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(main())
