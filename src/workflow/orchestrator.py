"""
src/workflow/orchestrator.py
=============================
Multi-case orchestration module for the auxetic plate pipeline.

This module coordinates:
  - loading and validating collections of ``CaseDefinition`` objects
  - status-aware sequential execution of all cases via ``case_runner.run_case``
  - aggregating per-case outputs (results, warnings, artifacts)
  - optional staged filtering via ``sweep_config.yaml`` semantics
  - optional ranking and reporting of successful results
  - returning a single structured ``OrchestratorSummary``

PIPELINE POSITION:
    main.py / run_sweep.py  →  [THIS MODULE]  →  OrchestratorSummary
                                              →  ranking results
                                              →  report files

ARCHITECTURAL DECISION — sequential execution in version 1:
    Version 1 runs cases one by one in a simple ``for`` loop.  This makes
    failure handling, logging, and debugging straightforward.  Parallel
    execution (multiprocessing, concurrent.futures) can be added in a
    future version behind an ``OrchestratorOptions.parallel`` flag without
    changing the public interface: the output is always ``OrchestratorSummary``.

ARCHITECTURAL DECISION — continue on case failure by default:
    The default ``OrchestratorOptions.continue_on_case_failure = True`` means
    a single failing case (geometry error, gmsh crash, CalculiX timeout) does
    not abort the remaining sweep.  Successful results are preserved and passed
    to ranking/reporting regardless of how many cases failed.  Use
    ``stop_on_first_failure = True`` in CI or when early-exit is preferred.

ARCHITECTURAL DECISION — ranking and reporting are optional post-run layers:
    If no cases succeed, ranking and reporting are skipped with a warning
    rather than crashing.  This keeps the orchestrator usable for diagnostic
    partial sweeps.

UNITS: consistent with project-wide convention (mm, N, MPa).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union, Optional

from workflow.case_schema import CaseDefinition, CaseResult, CaseStatus
from workflow.case_runner import (
    CaseRunOptions,
    CaseRunSummary,
    default_case_run_options,
    run_case,
)
from workflow.status_tracker import (
    case_should_be_skipped,
    mark_case_skipped,
    summarize_case_statuses,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class OrchestratorError(Exception):
    """
    Raised for unrecoverable orchestration-level failures.

    Examples:
      - empty or entirely-invalid case list supplied
      - ranking/reporting setup failure when ``strict`` mode is active
      - malformed stage selection with ``use_stage_case_filtering=True``

    Individual case failures are collected into ``OrchestratorSummary`` and
    do NOT raise this exception unless ``stop_on_first_failure=True`` is set.
    """


# ---------------------------------------------------------------------------
# Typed dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorOptions:
    """
    Configuration controlling how the orchestrator runs a case sweep.

    Attributes:
        skip_completed_cases:       Skip cases already marked COMPLETED.
        stop_on_first_failure:      Abort the sweep on the first case failure.
        run_ranking:                Call the ranking module after all cases run.
        run_reporting:              Call the reporting module after ranking.
        reports_dir:                Directory for report outputs (auto-resolved if None).
        top_n_filtered:             How many top-ranked results to include in filtered reports.
        stage_name:                 Named execution stage from ``sweep_config.yaml`` (optional).
        use_stage_case_filtering:   Enable filtering the case list by stage membership.
        continue_on_case_failure:   Continue to the next case even if one fails.
        case_run_options:           Per-case execution options (populated from config if None).
    """

    skip_completed_cases: bool = True
    stop_on_first_failure: bool = False
    run_ranking: bool = True
    run_reporting: bool = True
    reports_dir: Optional[str] = None
    top_n_filtered: int = 10
    stage_name: Optional[str] = None
    use_stage_case_filtering: bool = False
    continue_on_case_failure: bool = True
    case_run_options: Optional[CaseRunOptions] = None


@dataclass
class OrchestratorSummary:
    """
    Aggregated result of a multi-case orchestration run.

    Attributes:
        success:           True if all required stages completed without hard failure.
        total_cases:       Number of cases presented for execution.
        completed_cases:   Number of cases that completed successfully.
        failed_cases:      Number of cases that failed.
        skipped_cases:     Number of cases skipped (already completed or filtered).
        successful_results: ``{case_id: CaseResult}`` for completed cases only.
        all_case_results:   ``{case_id: CaseResult}`` for all cases attempted.
        ranking_result:    Output from ``analysis.ranking`` (or None if not run).
        reporting_result:  Output from ``analysis.reporting`` (or None if not run).
        warnings:          Non-fatal notes from any case or post-processing stage.
        metadata:          Supporting context (timings, plan summary, status counts).
        error_message:     Primary failure description if ``success`` is False.
    """

    success: bool
    total_cases: int
    completed_cases: int
    failed_cases: int
    skipped_cases: int
    successful_results: dict[str, CaseResult] = field(default_factory=dict)
    all_case_results: dict[str, CaseResult] = field(default_factory=dict)
    ranking_result: Optional[Any] = None
    reporting_result: Optional[Any] = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the summary."""
        rr = self.ranking_result
        repr_res = self.reporting_result
        return {
            "success":           self.success,
            "total_cases":       self.total_cases,
            "completed_cases":   self.completed_cases,
            "failed_cases":      self.failed_cases,
            "skipped_cases":     self.skipped_cases,
            "successful_results": {
                cid: (r.to_dict() if hasattr(r, "to_dict") else str(r))
                for cid, r in self.successful_results.items()
            },
            "all_case_results":  {
                cid: (r.to_dict() if hasattr(r, "to_dict") else str(r))
                for cid, r in self.all_case_results.items()
            },
            "ranking_result":    rr.to_dict() if rr is not None else None,
            "reporting_result":  repr_res.to_dict() if repr_res is not None else None,
            "warnings":          self.warnings,
            "metadata":          self.metadata,
            "error_message":     self.error_message,
        }


# ---------------------------------------------------------------------------
# Default options loader
# ---------------------------------------------------------------------------

def default_orchestrator_options(
    project_root: Union[str, Path, None] = None,
) -> OrchestratorOptions:
    """
    Build ``OrchestratorOptions`` from ``base_config.yaml`` workflow settings.

    Reads:
        - ``workflow.skip_completed_cases``     → ``skip_completed_cases``
        - ``workflow.stop_on_first_failure``    → ``stop_on_first_failure``
        - ``workflow.continue_on_case_failure`` → ``continue_on_case_failure``
        - ``workflow.run_ranking``              → ``run_ranking``
        - ``workflow.run_reporting``            → ``run_reporting``
        - ``workflow.top_n_filtered``           → ``top_n_filtered``

    Falls back to field defaults for any absent key.

    Args:
        project_root: Project root for config resolution.

    Returns:
        ``OrchestratorOptions`` populated from config where possible.
    """
    opts = OrchestratorOptions()
    try:
        from utils.config_loader import load_pipeline_config
        cfg = load_pipeline_config(project_root)
        wf = getattr(cfg, "workflow", {}) or {}

        opts.skip_completed_cases     = bool(wf.get("skip_completed_cases", True))
        opts.stop_on_first_failure    = bool(wf.get("stop_on_first_failure", False))
        opts.continue_on_case_failure = bool(wf.get("continue_on_case_failure", True))
        opts.run_ranking              = bool(wf.get("run_ranking", True))
        opts.run_reporting            = bool(wf.get("run_reporting", True))
        top_n = wf.get("top_n_filtered")
        if top_n is not None:
            opts.top_n_filtered = int(top_n)

        # Propagate solver/meshing defaults into nested CaseRunOptions
        opts.case_run_options = default_case_run_options(project_root)

    except Exception as exc:
        logger.warning(
            "Could not load orchestrator options from config: %s.  "
            "Using defaults.",
            exc,
        )
        opts.case_run_options = CaseRunOptions()

    return opts


# ---------------------------------------------------------------------------
# Stage filtering helper
# ---------------------------------------------------------------------------

def filter_cases_for_stage(
    case_definitions: Sequence[CaseDefinition],
    stage_name: Optional[str],
    project_root: Union[str, Path, None] = None,
) -> list[CaseDefinition]:
    """
    Filter a case list to only those belonging to a named execution stage.

    Version-1 stage filtering reads ``sweep_config.yaml`` to discover which
    load-case types and materials belong to a given stage.  Cases whose
    ``load_case.load_case_type`` and ``material.name`` both appear in the
    stage definition are included.

    If ``stage_name`` is None or no matching stage entry is found, all cases
    are returned with a warning rather than an empty list.

    Args:
        case_definitions: Full list of case definitions.
        stage_name:       Named stage from ``sweep_config.yaml`` (e.g.
                          ``"structural_screening"``).
        project_root:     Project root for config resolution.

    Returns:
        Filtered list of ``CaseDefinition`` objects.  Never empty if the
        original list was non-empty and stage resolution fails — falls back
        to all cases in that event.
    """
    if stage_name is None:
        return list(case_definitions)

    stage_load_cases: set[str] = set()
    stage_materials:  set[str] = set()

    try:
        from utils.config_loader import load_pipeline_config
        cfg = load_pipeline_config(project_root)
        raw_sweep = getattr(cfg, "sweep", {}) or {}
        stages = raw_sweep.get("stages", {})
        stage_cfg = stages.get(stage_name)
        if stage_cfg is None:
            logger.warning(
                "Stage '%s' not found in sweep_config.yaml.  "
                "No filtering applied — running all cases.",
                stage_name,
            )
            return list(case_definitions)

        lc_filter = stage_cfg.get("load_cases") or []
        mat_filter = stage_cfg.get("materials") or []
        stage_load_cases = {str(lc).lower() for lc in lc_filter}
        stage_materials  = {str(m).lower()  for m in mat_filter}

    except Exception as exc:
        logger.warning(
            "Could not load stage filter config for stage '%s': %s.  "
            "Running all cases.",
            stage_name, exc,
        )
        return list(case_definitions)

    filtered: list[CaseDefinition] = []
    for cd in case_definitions:
        lc_match  = (
            not stage_load_cases
            or cd.load_case.load_case_type.value.lower() in stage_load_cases
        )
        mat_match = (
            not stage_materials
            or cd.material.name.lower() in stage_materials
        )
        if lc_match and mat_match:
            filtered.append(cd)

    logger.info(
        "Stage filter '%s': %d/%d cases selected.",
        stage_name, len(filtered), len(case_definitions),
    )
    return filtered


# ---------------------------------------------------------------------------
# Case collection helpers
# ---------------------------------------------------------------------------

def case_definitions_by_id(
    case_definitions: Sequence[CaseDefinition],
) -> dict[str, CaseDefinition]:
    """
    Build a ``{case_id: CaseDefinition}`` mapping from a sequence.

    Duplicate case IDs are logged as warnings; the last definition wins.

    Args:
        case_definitions: Sequence of case definitions.

    Returns:
        Dict keyed by ``case_id``.
    """
    result: dict[str, CaseDefinition] = {}
    for cd in case_definitions:
        if cd.case_id in result:
            logger.warning(
                "Duplicate case_id '%s' in input — last definition takes precedence.",
                cd.case_id,
            )
        result[cd.case_id] = cd
    return result


def summarize_case_plan(
    case_definitions: Sequence[CaseDefinition],
) -> dict[str, Any]:
    """
    Produce a human-readable summary dict describing the planned case sweep.

    Includes:
      - total case count
      - breakdown by design type
      - breakdown by material name
      - breakdown by load-case type

    Args:
        case_definitions: Sequence of case definitions to summarise.

    Returns:
        Dict suitable for logging and metadata.
    """
    by_design:    dict[str, int] = {}
    by_material:  dict[str, int] = {}
    by_load_case: dict[str, int] = {}

    for cd in case_definitions:
        dt  = cd.design_type.value
        mat = cd.material.name
        lc  = cd.load_case.load_case_type.value
        by_design[dt]       = by_design.get(dt, 0) + 1
        by_material[mat]    = by_material.get(mat, 0) + 1
        by_load_case[lc]    = by_load_case.get(lc, 0) + 1

    return {
        "total_cases":   len(case_definitions),
        "by_design":     by_design,
        "by_material":   by_material,
        "by_load_case":  by_load_case,
    }


# ---------------------------------------------------------------------------
# Skip decision helper
# ---------------------------------------------------------------------------

def _should_skip_case(
    case_definition: CaseDefinition,
    options: OrchestratorOptions,
    project_root: Path,
) -> bool:
    """
    Return True if the orchestrator should skip this case.

    A case is skipped when ``skip_completed_cases=True`` and the on-disk
    status for the case is COMPLETED or SKIPPED.

    Args:
        case_definition: Case to check.
        options:         Orchestrator options.
        project_root:    Project root for status file resolution.

    Returns:
        True if the case should be skipped.
    """
    if not options.skip_completed_cases:
        return False
    return case_should_be_skipped(
        case_definition.case_id,
        skip_completed_cases=True,
        project_root=project_root,
    )


# ---------------------------------------------------------------------------
# Ranking integration helper
# ---------------------------------------------------------------------------

def _run_ranking_if_enabled(
    options: OrchestratorOptions,
    successful_results: dict[str, CaseResult],
    warnings: list[str],
    meta: dict[str, Any],
) -> Optional[Any]:
    """
    Run the ranking module on successful case results if enabled.

    Args:
        options:            Orchestrator options.
        successful_results: ``{case_id: CaseResult}`` for completed cases.
        warnings:           Mutable warning list.
        meta:               Mutable metadata dict for timing.

    Returns:
        Ranking result object or None.
    """
    if not options.run_ranking:
        return None

    if not successful_results:
        warnings.append(
            "Ranking skipped: no successful case results to rank."
        )
        return None

    try:
        from analysis.ranking import rank_cases

        t0 = time.monotonic()
        ranking_result = rank_cases(
            case_metrics={case_id: result.metrics for case_id, result in successful_results.items()},
        )
        meta["ranking_seconds"] = float(f"{time.monotonic() - t0:.3f}")

        if ranking_result.warnings:
            warnings.extend(
                f"[ranking] {w}" for w in ranking_result.warnings
            )

        logger.info(
            "Ranking complete: %d cases ranked in %.3f s.",
            len(successful_results),
            meta["ranking_seconds"],
        )
        return ranking_result

    except Exception as exc:
        warnings.append(
            f"Ranking failed (non-fatal): {exc}.  "
            "Case results are still available in the summary."
        )
        return None


# ---------------------------------------------------------------------------
# Reporting integration helper
# ---------------------------------------------------------------------------

def _run_reporting_if_enabled(
    options: OrchestratorOptions,
    all_case_results: dict[str, CaseResult],
    ranking_result: Optional[Any],
    project_root: Path,
    warnings: list[str],
    meta: dict[str, Any],
    case_definitions: Mapping[str, CaseDefinition] | None = None,
) -> Optional[Any]:
    """
    Run the reporting module after ranking if enabled.

    Uses all case results (including failures) for the full CSV summary,
    and ranking_result for the filtered/ranked outputs.

    Args:
        options:            Orchestrator options.
        all_case_results:   ``{case_id: CaseResult}`` for all attempted cases.
        ranking_result:     Ranking output (or None).
        project_root:       Project root for report output resolution.
        warnings:           Mutable warning list.
        meta:               Mutable metadata dict for timing.
        case_definitions:   ``{case_id: CaseDefinition}`` for design-grouped
                            plots and CSV (recommended).

    Returns:
        Reporting result object or None.
    """
    if not options.run_reporting:
        return None

    if not all_case_results:
        warnings.append(
            "Reporting skipped: no case results to report on."
        )
        return None

    if options.reports_dir is not None:
        rep_str = f"{options.reports_dir}"
        reports_dir: Path = Path(rep_str)
    else:
        reports_dir: Path = project_root / "reports"

    try:
        from analysis.reporting import generate_reports

        t0 = time.monotonic()
        reporting_result = generate_reports(
            case_results=all_case_results,
            ranking_result=ranking_result,
            case_definitions=case_definitions,
            project_root=project_root,
            reports_dir=reports_dir,
            top_n_filtered=options.top_n_filtered,
        )
        meta["reporting_seconds"] = float(f"{time.monotonic() - t0:.3f}")

        if reporting_result.warnings:
            warnings.extend(
                f"[reporting] {w}" for w in reporting_result.warnings
            )

        logger.info(
            "Reporting complete: output directory '%s' (%.3f s).",
            reports_dir,
            meta["reporting_seconds"],
        )
        return reporting_result

    except Exception as exc:
        warnings.append(
            f"Reporting failed (non-fatal): {exc}.  "
            "Case results and ranking are preserved in the summary."
        )
        return None


# ---------------------------------------------------------------------------
# Main orchestration function
# ---------------------------------------------------------------------------

def run_cases(
    case_definitions: Sequence[CaseDefinition],
    options: Optional[OrchestratorOptions] = None,
    project_root: Union[str, Path, None] = None,
) -> OrchestratorSummary:
    """
    Execute a collection of pipeline cases sequentially and return a summary.

    Orchestration flow:
        1.  Resolve options and project root.
        2.  Validate the incoming case list (must be non-empty).
        3.  Optionally filter by stage using ``sweep_config.yaml`` semantics.
        4.  Log and record the run plan.
        5.  Iterate through cases sequentially:
              a. Check skip condition.
              b. Call ``case_runner.run_case``.
              c. Collect result, update counters, accumulate warnings.
              d. If configured, stop on first failure.
        6.  Run ranking on successful results (if enabled).
        7.  Run reporting on all results (if enabled).
        8.  Assemble and return ``OrchestratorSummary``.

    Args:
        case_definitions: Sequence of ``CaseDefinition`` objects to execute.
        options:          Orchestration options (defaults loaded from config if None).
        project_root:     Project root (auto-detected if None).

    Returns:
        ``OrchestratorSummary`` with full sweep outcome, results, and metadata.

    Raises:
        OrchestratorError: only if ``stop_on_first_failure=True`` and a case
                           fails, or if the supplied case list is invalid.
    """
    # ------------------------------------------------------------------
    # 1. Resolve options and project root
    # ------------------------------------------------------------------
    if options is None:
        options = default_orchestrator_options(project_root)

    resolved_root: Path = (
        Path(project_root).resolve()
        if project_root is not None
        else Path.cwd()
    )

    # Ensure case_run_options is populated
    case_run_opts: CaseRunOptions = options.case_run_options or default_case_run_options(
        resolved_root
    )
    # Propagate skip_completed into case-level options for uniformity
    case_run_opts.skip_completed_cases = options.skip_completed_cases

    t_sweep_start = time.monotonic()
    warnings: list[str] = []
    meta: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 2. Validate case list
    # ------------------------------------------------------------------
    if not case_definitions:
        msg = "No case definitions provided to orchestrator."
        logger.error(msg)
        raise OrchestratorError(msg)

    # ------------------------------------------------------------------
    # 3. Stage filtering
    # ------------------------------------------------------------------
    if options.use_stage_case_filtering and options.stage_name:
        runnable_cases = filter_cases_for_stage(
            case_definitions,
            options.stage_name,
            resolved_root,
        )
        if not runnable_cases:
            warnings.append(
                f"Stage filter '{options.stage_name}' produced zero cases.  "
                "Check stage definitions in sweep_config.yaml."
            )
            runnable_cases = list(case_definitions)
    else:
        runnable_cases = list(case_definitions)

    # ------------------------------------------------------------------
    # 4. Log run plan
    # ------------------------------------------------------------------
    plan = summarize_case_plan(runnable_cases)
    meta["run_plan"]     = plan
    meta["stage_name"]   = options.stage_name
    meta["run_solver"]   = case_run_opts.run_solver

    logger.info(
        "Orchestrator: running %d cases — designs: %s | materials: %s | load cases: %s",
        plan["total_cases"],
        plan["by_design"],
        plan["by_material"],
        plan["by_load_case"],
    )
    if options.stage_name:
        logger.info("Stage filter: '%s'.", options.stage_name)

    # ------------------------------------------------------------------
    # 5. Sequential case execution
    # ------------------------------------------------------------------
    all_case_results: dict[str, CaseResult] = {}
    successful_results: dict[str, CaseResult] = {}
    completed_cases_tracker: list[int] = []
    failed_cases_tracker: list[int] = []
    skipped_cases_tracker: list[int] = []
    stage_timings:   dict[str, float] = {}

    for idx, cd in enumerate(runnable_cases, start=1):
        case_id = cd.case_id
        logger.info(
            "Orchestrator [%d/%d]: case '%s'.",
            idx, len(runnable_cases), case_id,
        )

        # --- Skip check ---
        if _should_skip_case(cd, options, resolved_root):
            logger.info(
                "Case '%s': skipped (already completed).", case_id
            )
            mark_case_skipped(
                case_id,
                project_root=resolved_root,
                reason="Orchestrator: skipped, result already completed.",
            )
            skipped_cases_tracker.append(1)
            # Preserve a minimal result entry for reporting
            all_case_results[case_id] = CaseResult(
                case_id=case_id,
                status=CaseStatus.SKIPPED,
                success=True,
                metadata={"skipped_by_orchestrator": True},
            )
            continue

        # --- Execute case ---
        t0 = time.monotonic()
        try:
            summary: CaseRunSummary = run_case(
                case_definition=cd,
                options=case_run_opts,
                project_root=resolved_root,
            )
        except Exception as exc:
            # Unexpected runner-level exception (should be rare — run_case is
            # designed to return a failed CaseRunSummary rather than raise).
            err_msg = f"Case '{case_id}': unhandled exception in run_case: {exc}"
            logger.exception(err_msg)
            warnings.append(err_msg)
            failed_cases_tracker.append(1)

            placeholder = CaseResult(
                case_id=case_id,
                status=CaseStatus.FAILED,
                success=False,
                error_message=str(exc),
            )
            all_case_results[case_id] = placeholder

            if options.stop_on_first_failure:
                raise OrchestratorError(
                    f"Stopping on first failure: {err_msg}"
                ) from exc
            continue

        case_elapsed = float(f"{time.monotonic() - t0:.3f}")
        stage_timings[case_id] = case_elapsed

        # Accumulate warnings
        if summary.warnings:
            warnings.extend(
                f"[{case_id}] {w}" for w in summary.warnings
            )

        all_case_results[case_id] = summary.case_result

        if summary.success:
            completed_cases_tracker.append(1)
            successful_results[case_id] = summary.case_result
            logger.info(
                "Case '%s': completed in %.3f s.", case_id, case_elapsed
            )
        else:
            failed_cases_tracker.append(1)
            f_count = len(failed_cases_tracker)
            logger.warning(
                "Case '%s': failed in %.3f s — %s",
                case_id, case_elapsed, summary.error_message,
            )
            if f_count <= 5:
                # ARCHITECTURAL DECISION — print first few failures to stdout:
                #   In notebook environments (Colab/Jupyter), deep logs are often
                #   hidden. Printing the first few detailed errors ensures the
                #   user sees the root cause (e.g. missing binary) immediately.
                print(f"\n[ORCHESTRATOR ERROR] Failure {f_count}/5:")
                print(f"  Case: {case_id}")
                print(f"  Error: {summary.error_message}")
                if summary.warnings:
                    print(f"  Warnings: {summary.warnings[:3]}")
                print("-" * 40)

            if options.stop_on_first_failure:
                raise OrchestratorError(
                    f"Stopping on first failure: case '{case_id}' — "
                    f"{summary.error_message}"
                )

            if not options.continue_on_case_failure:
                warnings.append(
                    f"Case '{case_id}' failed and continue_on_case_failure=False.  "
                    "Halting sweep."
                )
                break

    meta["stage_timings"]   = stage_timings
    meta["status_snapshot"] = summarize_case_statuses(
        [cd.case_id for cd in runnable_cases],
        project_root=resolved_root,
    )

    total_sweep_seconds = float(f"{time.monotonic() - t_sweep_start:.3f}")
    meta["total_sweep_seconds"] = total_sweep_seconds

    # ------------------------------------------------------------------
    # 6. Ranking
    # ------------------------------------------------------------------
    ranking_result = _run_ranking_if_enabled(
        options, successful_results, warnings, meta
    )

    # ------------------------------------------------------------------
    # 7. Reporting
    # ------------------------------------------------------------------
    case_def_map = case_definitions_by_id(runnable_cases)
    reporting_result = _run_reporting_if_enabled(
        options,
        all_case_results,
        ranking_result,
        resolved_root,
        warnings,
        meta,
        case_definitions=case_def_map,
    )

    # ------------------------------------------------------------------
    # 8. Assemble summary
    # ------------------------------------------------------------------
    sum_completed = len(completed_cases_tracker)
    sum_failed = len(failed_cases_tracker)
    sum_skipped = len(skipped_cases_tracker)
    
    if sum_completed > 0:
        logger.info(
            "%d case(s) successfully simulated/generated.",
            sum_completed,
        )
    if sum_skipped > 0:
        logger.info("%d case(s) skipped.", sum_skipped)
    if sum_failed > 0:
        logger.info("%d case(s) failed.", sum_failed)

    n_runnable = len(runnable_cases)
    all_skipped = n_runnable > 0 and sum_skipped == n_runnable
    success_overall = sum_failed == 0 and (
        len(successful_results) > 0 or all_skipped
    )
    err_msg: str | None = None
    if sum_failed > 0 and not success_overall:
        err_msg = f"{sum_failed} case(s) failed during the sweep."

    return OrchestratorSummary(
        success=success_overall,
        total_cases=len(runnable_cases),
        completed_cases=sum_completed,
        failed_cases=sum_failed,
        skipped_cases=sum_skipped,
        successful_results=successful_results,
        all_case_results=all_case_results,
        ranking_result=ranking_result,
        reporting_result=reporting_result,
        warnings=warnings,
        metadata=meta,
        error_message=err_msg,
    )


# ---------------------------------------------------------------------------
# Convenience wrapper: mapping input
# ---------------------------------------------------------------------------

def run_case_map(
    case_definitions: Mapping[str, CaseDefinition],
    options: OrchestratorOptions | None = None,
    project_root: str | Path | None = None,
) -> OrchestratorSummary:
    """
    Run a mapping of ``{case_id: CaseDefinition}`` through the orchestrator.

    This is a thin delegating wrapper around ``run_cases`` for callers that
    already have cases in a dictionary (e.g. from ``cache.build_case_map``).

    Args:
        case_definitions: Dict of case definitions keyed by case ID.
        options:          Orchestrator options.
        project_root:     Project root.

    Returns:
        ``OrchestratorSummary`` identical to calling ``run_cases``.
    """
    return run_cases(
        case_definitions=list(case_definitions.values()),
        options=options,
        project_root=project_root,
    )


# ---------------------------------------------------------------------------
# Convenience wrapper: named stage
# ---------------------------------------------------------------------------

def run_stage(
    case_definitions: Sequence[CaseDefinition],
    stage_name: str,
    options: OrchestratorOptions | None = None,
    project_root: str | Path | None = None,
) -> OrchestratorSummary:
    """
    Run a named execution stage from the sweep configuration.

    Delegates to ``run_cases`` with ``use_stage_case_filtering=True`` and
    ``stage_name`` set.

    Args:
        case_definitions: Full set of case definitions (pre-filtering).
        stage_name:       Named stage from ``sweep_config.yaml``.
        options:          Orchestrator options.
        project_root:     Project root.

    Returns:
        ``OrchestratorSummary`` for the filtered stage cases.
    """
    if options is None:
        options = default_orchestrator_options(project_root)

    options.stage_name               = stage_name
    options.use_stage_case_filtering = True

    return run_cases(
        case_definitions=case_definitions,
        options=options,
        project_root=project_root,
    )


# ---------------------------------------------------------------------------
# Strict wrapper
# ---------------------------------------------------------------------------

def require_successful_orchestration(
    case_definitions: Sequence[CaseDefinition],
    options: OrchestratorOptions | None = None,
    project_root: str | Path | None = None,
) -> OrchestratorSummary:
    """
    Run the orchestrator and raise ``OrchestratorError`` if it fails.

    A failure is defined as ``OrchestratorSummary.success == False``.
    This can occur when:
      - ``stop_on_first_failure=True`` and a case fails.
      - No cases complete and no successful results are produced.

    Successful partial sweeps (some failures, some completions, with
    ``continue_on_case_failure=True``) are not raised as errors.

    Args:
        case_definitions: Sequence of case definitions.
        options:          Orchestrator options.
        project_root:     Project root.

    Returns:
        ``OrchestratorSummary`` with ``success=True``.

    Raises:
        OrchestratorError: if the summary reports failure.
    """
    summary = run_cases(case_definitions, options, project_root)
    if not summary.success:
        raise OrchestratorError(
            f"Orchestration failed: {summary.error_message}  "
            f"({summary.failed_cases}/{summary.total_cases} cases failed)"
        )
    return summary
