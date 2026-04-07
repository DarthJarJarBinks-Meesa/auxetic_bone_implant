"""
src/analysis/reporting.py
===========================
Reporting module for the auxetic plate pipeline.

This module converts structured pipeline results, metrics, and ranking
outputs into human-readable and machine-readable summary artifacts.

PIPELINE POSITION:
    ranking.py  +  case_results  →  [THIS MODULE]  →  reports/

OUTPUTS:
    reports/
    ├── summary.csv
    ├── ranked_results.csv
    ├── design_comparison.csv
    ├── filtered_candidates.csv
    └── plots/
        ├── modulus_by_design.png
        ├── max_stress_by_design.png
        ├── fatigue_metric_by_design.png
        ├── displacement_by_design.png
        └── stress_strain_curves/
            └── <case_id>_stress_strain.png ...

ARCHITECTURAL DECISION — standard library csv, no pandas:
    CSV writing uses Python's built-in ``csv.DictWriter``.  All rows are
    plain dicts.  This keeps the dependency footprint minimal and makes
    the output format transparent and auditable.

ARCHITECTURAL DECISION — one figure per chart, no subplots:
    Each metric gets its own matplotlib figure.  This keeps the plot code
    simple, avoids subplot layout issues, and produces individually
    importable images for reports and presentations.

ARCHITECTURAL DECISION — graceful degradation on missing data:
    Any chart or report section that lacks sufficient data is skipped with
    a warning rather than aborting the whole reporting run.  The
    ``ReportingResult.written_files`` list tells callers exactly what was
    produced.

ARCHITECTURAL DECISION — config toggles are honoured but have sane defaults:
    If the config cannot be loaded (e.g. config files absent), all reports
    and plots default to enabled.  Explicit ``False`` in config disables
    individual outputs.

UNITS: consistent with MetricSet (MPa, mm, N/mm).
"""

from __future__ import annotations

import csv
import json
import logging
import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; safe for headless environments
import matplotlib.pyplot as plt

from analysis.ranking import RankedCase, RankingResult
from workflow.case_schema import CaseDefinition, CaseResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class ReportingError(Exception):
    """
    Raised for unrecoverable reporting failures.

    Missing-data situations (empty metric columns, no ranked cases) are
    handled with warnings in ``ReportingResult``.
    """


# ---------------------------------------------------------------------------
# Typed dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ReportPaths:
    """
    Canonical file paths for all version-1 report outputs.

    Attributes:
        reports_dir:              Root reports directory.
        summary_csv:              Flat table of all cases and metrics.
        ranked_results_csv:       Cases sorted by ranking score.
        design_comparison_csv:    Per-design aggregated metrics.
        filtered_candidates_csv:  Top-N ranked candidates.
        plots_dir:                Directory for metric bar charts.
        stress_strain_curves_dir: Directory for per-case S-S curve plots.
    """

    reports_dir: str
    summary_csv: str
    ranked_results_csv: str
    design_comparison_csv: str
    filtered_candidates_csv: str
    plots_dir: str
    stress_strain_curves_dir: str

    def to_dict(self) -> dict[str, str]:
        return {
            "reports_dir":              self.reports_dir,
            "summary_csv":              self.summary_csv,
            "ranked_results_csv":       self.ranked_results_csv,
            "design_comparison_csv":    self.design_comparison_csv,
            "filtered_candidates_csv":  self.filtered_candidates_csv,
            "plots_dir":                self.plots_dir,
            "stress_strain_curves_dir": self.stress_strain_curves_dir,
        }


@dataclass
class ReportingResult:
    """
    Summary of what the reporting run produced.

    Attributes:
        success:       True if at least the primary CSV was written.
        written_files: Absolute paths to all files successfully written.
        warnings:      Non-fatal notes about skipped reports or plots.
        metadata:      Supporting context for logging and orchestration.
        error_message: Error description if ``success`` is False.
    """

    success: bool
    written_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success":       self.success,
            "written_files": self.written_files,
            "warnings":      self.warnings,
            "metadata":      self.metadata,
            "error_message": self.error_message,
        }


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def _ensure_directory(path: Path) -> None:
    """Create ``path`` and any missing parents, silently if it already exists."""
    path.mkdir(parents=True, exist_ok=True)


def resolve_report_paths(
    project_root: str | Path | None = None,
    reports_dir: str | Path | None = None,
) -> ReportPaths:
    """
    Resolve canonical report output paths and create required directories.

    Resolution order:
      1. ``reports_dir`` argument (explicit override).
      2. ``base_config.yaml [paths.reports_dir]`` via config loader.
      3. ``<project_root>/reports`` by convention.

    Args:
        project_root: Project root (auto-detected if None).
        reports_dir:  Override reports root directory (optional).

    Returns:
        ``ReportPaths`` with all subdirectories created.
    """
    if reports_dir is not None:
        root = Path(reports_dir).resolve()
    else:
        root = Path("reports")   # fallback convention
        try:
            from utils.config_loader import load_pipeline_config
            cfg = load_pipeline_config(project_root)
            root = cfg.get_reports_directory()
        except Exception:
            pass

    plots = root / "plots"
    ss_curves = plots / "stress_strain_curves"

    _ensure_directory(root)
    _ensure_directory(plots)
    _ensure_directory(ss_curves)

    return ReportPaths(
        reports_dir=str(root),
        summary_csv=str(root / "summary.csv"),
        ranked_results_csv=str(root / "ranked_results.csv"),
        design_comparison_csv=str(root / "design_comparison.csv"),
        filtered_candidates_csv=str(root / "filtered_candidates.csv"),
        plots_dir=str(plots),
        stress_strain_curves_dir=str(ss_curves),
    )


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _group_case_ids_by_design(
    case_definitions: Mapping[str, CaseDefinition],
) -> dict[str, list[str]]:
    """Group case IDs by design type string."""
    groups: dict[str, list[str]] = {}
    for case_id, cd in case_definitions.items():
        design = cd.design_type.value if cd.design_type else "unknown"
        groups.setdefault(design, []).append(case_id)
    return groups


def _mean_of_available(values: list[float | None]) -> float | None:
    """Return the mean of non-None finite values, or None."""
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.mean(finite) if finite else None


def _metric_from_case_result(
    case_result: CaseResult,
    metric_name: str,
) -> float | None:
    """Safely read a metric field from a CaseResult's MetricSet."""
    if case_result.metrics is None:
        return None
    return getattr(case_result.metrics, metric_name, None)


# ---------------------------------------------------------------------------
# CSV row flattening helpers
# ---------------------------------------------------------------------------

def _stringify_value(value: Any) -> Any:
    """
    Convert a value to a CSV-safe type.

    Lists and dicts are JSON-serialised.  None stays None.  Floats are
    rounded to 6 significant figures for readability.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return float(f"{value:.6f}")
    if isinstance(value, (list, dict)):
        try:
            return json.dumps(value)
        except Exception:
            return str(value)
    return value


def flatten_case_result_row(
    case_id: str,
    case_result: CaseResult,
    case_definition: CaseDefinition | None = None,
    rank: int | None = None,
    total_score: float | None = None,
) -> dict[str, Any]:
    """
    Flatten a ``CaseResult`` (and optional ``CaseDefinition``) into a
    flat row dict suitable for CSV writing.

    Args:
        case_id:         Case identifier.
        case_result:     Result record.
        case_definition: Full case definition (optional).
        rank:            Rank from ``RankingResult`` (optional).
        total_score:     Weighted normalised score (optional).

    Returns:
        Flat dict with string keys and CSV-safe values.
    """
    row: dict[str, Any] = {"case_id": case_id}

    # --- Case definition fields ---
    if case_definition is not None:
        row["design_type"] = case_definition.design_type.value
        row["cell_size_mm"] = case_definition.design_parameters.cell_size
        row["plate_thickness_mm"] = case_definition.plate_thickness
        row["material"] = case_definition.material.name
        row["load_case"] = case_definition.load_case.load_case_type.value
        row["lattice_x"] = case_definition.lattice_repeats_x
        row["lattice_y"] = case_definition.lattice_repeats_y
    else:
        for col in ("design_type", "cell_size_mm", "plate_thickness_mm",
                    "material", "load_case", "lattice_x", "lattice_y"):
            row[col] = None

    # --- Case result fields ---
    row["status"]  = case_result.status.value
    row["success"] = case_result.success
    row["runtime_seconds"] = case_result.runtime_seconds
    row["solver_return_code"] = case_result.solver_return_code
    row["error_message"] = case_result.error_message or ""

    # --- Metrics ---
    ms = case_result.metrics
    if ms is not None:
        row["max_von_mises_stress_mpa"]    = ms.max_von_mises_stress_mpa
        row["max_displacement_mm"]         = ms.max_displacement_mm
        row["effective_stiffness_n_per_mm"]= ms.effective_stiffness_n_per_mm
        row["effective_modulus_mpa"]       = ms.effective_modulus_mpa
        row["fatigue_risk_score"]          = ms.fatigue_risk_score
        row["hotspot_stress_mpa"]          = ms.hotspot_stress_mpa
        row["stress_strain_point_count"]   = len(ms.stress_strain_points)
    else:
        for col in ("max_von_mises_stress_mpa", "max_displacement_mm",
                    "effective_stiffness_n_per_mm", "effective_modulus_mpa",
                    "fatigue_risk_score", "hotspot_stress_mpa",
                    "stress_strain_point_count"):
            row[col] = None

    row["rank"]        = rank
    row["total_score"] = total_score

    return {k: _stringify_value(v) for k, v in row.items()}


def flatten_ranked_case_row(
    ranked_case: RankedCase,
    case_definition: CaseDefinition | None = None,
    case_result: CaseResult | None = None,
) -> dict[str, Any]:
    """
    Flatten a ``RankedCase`` into a flat row dict for the ranked-results CSV.

    Args:
        ranked_case:     Ranked case from ``RankingResult``.
        case_definition: Full case definition (optional).
        case_result:     Case result (optional; used for status/metrics).

    Returns:
        Flat dict with string keys and CSV-safe values.
    """
    row: dict[str, Any] = {
        "case_id":        ranked_case.case_id,
        "rank":           ranked_case.rank,
        "total_score":    ranked_case.total_score,
        "missing_metrics": "|".join(ranked_case.missing_metrics),
    }

    # Design info
    if case_definition is not None:
        row["design_type"]       = case_definition.design_type.value
        row["plate_thickness_mm"]= case_definition.plate_thickness
        row["material"]          = case_definition.material.name
        row["load_case"]         = case_definition.load_case.load_case_type.value
    else:
        for col in ("design_type", "plate_thickness_mm", "material", "load_case"):
            row[col] = None

    # Status
    if case_result is not None:
        row["status"]  = case_result.status.value
        row["success"] = case_result.success

    # Raw metric values
    for k, v in ranked_case.raw_metric_values.items():
        row[f"raw_{k}"] = v

    # Normalised component scores
    for k, v in ranked_case.normalized_component_scores.items():
        row[f"norm_{k}"] = v

    return {k: _stringify_value(v) for k, v in row.items()}


# ---------------------------------------------------------------------------
# CSV writing helpers
# ---------------------------------------------------------------------------

def write_csv_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """
    Write a list of row dicts to a CSV file using the standard library.

    If ``rows`` is empty, writes a single-row CSV with no data.
    Column order follows the key union of all rows.

    Args:
        path: Destination file path.
        rows: List of flat row dicts.
    """
    path = Path(path)
    _ensure_directory(path.parent)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    # Union of all keys, preserving first-seen order
    header: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                header.append(k)
                seen.add(k)

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in header})


# ---------------------------------------------------------------------------
# Summary report writers
# ---------------------------------------------------------------------------

def write_summary_csv(
    case_results: Mapping[str, CaseResult],
    output_path: str | Path,
    case_definitions: Mapping[str, CaseDefinition] | None = None,
) -> str:
    """
    Write ``summary.csv`` — a flat table of all cases and their metrics.

    Args:
        case_results:     All case results.
        output_path:      Destination CSV path.
        case_definitions: Optional case definitions for design metadata.

    Returns:
        Written file path as string.
    """
    rows = []
    for case_id, cr in case_results.items():
        defs = case_definitions
        if defs is not None:
            cd = defs.get(case_id)
        rows.append(flatten_case_result_row(case_id, cr, cd))

    write_csv_rows(output_path, rows)
    logger.info("summary.csv written: %d rows → %s", len(rows), output_path)
    return str(output_path)


def write_ranked_results_csv(
    ranking_result: RankingResult,
    output_path: str | Path,
    case_results: Mapping[str, CaseResult] | None = None,
    case_definitions: Mapping[str, CaseDefinition] | None = None,
) -> str:
    """
    Write ``ranked_results.csv`` — cases sorted by ranking score.

    Args:
        ranking_result:   Completed ``RankingResult``.
        output_path:      Destination CSV path.
        case_results:     Optional case results for status/metrics columns.
        case_definitions: Optional case definitions for design metadata.

    Returns:
        Written file path as string.
    """
    rows = []
    for rc in ranking_result.ranked_cases:
        defs = case_definitions
        if defs is not None:
            cd = defs.get(rc.case_id)
        cr = None
        res = case_results
        if res is not None:
            cr = res.get(rc.case_id)
        rows.append(flatten_ranked_case_row(rc, cd, cr))

    write_csv_rows(output_path, rows)
    logger.info("ranked_results.csv written: %d rows → %s", len(rows), output_path)
    return str(output_path)


def write_design_comparison_csv(
    case_results: Mapping[str, CaseResult],
    output_path: str | Path,
    case_definitions: Mapping[str, CaseDefinition] | None = None,
) -> str:
    """
    Write ``design_comparison.csv`` — per-design aggregated metric summaries.

    Groups successful cases by design type and reports count and mean/max/min
    of each key metric across the group.

    Args:
        case_results:     All case results.
        output_path:      Destination CSV path.
        case_definitions: Required to group by design type.

    Returns:
        Written file path as string.
    """
    if not case_definitions:
        write_csv_rows(output_path, [])
        logger.warning(
            "design_comparison.csv: no case_definitions provided; writing empty file."
        )
        return str(output_path)

    groups = _group_case_ids_by_design(case_definitions)
    metric_fields = [
        ("max_von_mises_stress_mpa",    "max_von_mises_stress_mpa"),
        ("max_displacement_mm",         "max_displacement_mm"),
        ("effective_stiffness_n_per_mm","effective_stiffness_n_per_mm"),
        ("effective_modulus_mpa",       "effective_modulus_mpa"),
        ("fatigue_risk_score",          "fatigue_risk_score"),
    ]

    rows = []
    for design, case_ids in sorted(groups.items()):
        group_results = [
            case_results[cid] for cid in case_ids if cid in case_results
        ]
        successful = [cr for cr in group_results if cr.success]

        row: dict[str, Any] = {
            "design_type":          design,
            "case_count":           len(group_results),
            "successful_case_count": len(successful),
        }

        for col_key, metric_attr in metric_fields:
            values = [
                _metric_from_case_result(cr, metric_attr)
                for cr in successful
            ]
            finite = [v for v in values if v is not None and math.isfinite(v)]
            row[f"mean_{col_key}"] = float(f"{statistics.mean(finite):.4f}") if finite else None
            row[f"min_{col_key}"]  = float(f"{min(finite):.4f}") if finite else None
            row[f"max_{col_key}"]  = float(f"{max(finite):.4f}") if finite else None

        rows.append({k: _stringify_value(v) for k, v in row.items()})

    write_csv_rows(output_path, rows)
    logger.info("design_comparison.csv written: %d rows → %s", len(rows), output_path)
    return str(output_path)


def write_filtered_candidates_csv(
    ranking_result: RankingResult,
    output_path: str | Path,
    top_n: int = 10,
) -> str:
    """
    Write ``filtered_candidates.csv`` — the top-N ranked cases.

    Args:
        ranking_result: Completed ``RankingResult``.
        output_path:    Destination CSV path.
        top_n:          Maximum number of candidates to include.

    Returns:
        Written file path as string.
    """
    sorted_cases = [
        rc for rc in ranking_result.ranked_cases
        if rc.total_score is not None
    ]
    top_cases = [rc for i, rc in enumerate(sorted_cases) if i < top_n]

    rows = [flatten_ranked_case_row(rc) for rc in top_cases]
    write_csv_rows(output_path, rows)
    logger.info(
        "filtered_candidates.csv written: %d rows (top %d) → %s",
        len(rows), top_n, output_path,
    )
    return str(output_path)


# ---------------------------------------------------------------------------
# JSON summary helper
# ---------------------------------------------------------------------------

def write_json_summary(
    path: str | Path,
    payload: dict[str, Any],
) -> str:
    """
    Write a JSON summary file.

    Args:
        path:    Destination file path.
        payload: JSON-serialisable dict.

    Returns:
        Written file path as string.
    """
    path = Path(path)
    _ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return str(path)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_metric_by_design(
    case_results: Mapping[str, CaseResult],
    case_definitions: Mapping[str, CaseDefinition] | None,
    metric_name: str,
    y_label: str,
    output_path: str | Path,
) -> str | None:
    """
    Create a bar chart of a metric grouped by design type.

    ARCHITECTURAL DECISION — mean per design:
        For version-1 comparative screening, the mean across all successful
        cases of each design type is the most stable aggregate.  Individual
        case scatterplots are better suited for an interactive dashboard
        (future version).

    Args:
        case_results:     All case results.
        case_definitions: Optional; required to group by design type.
        metric_name:      ``MetricSet`` attribute name (e.g. ``"max_von_mises_stress_mpa"``).
        y_label:          Y-axis label string.
        output_path:      Destination ``.png`` path.

    Returns:
        Written file path string, or ``None`` if insufficient data.
    """
    if not case_definitions:
        logger.warning(
            "plot_metric_by_design (%s): no case_definitions; skipping.",
            metric_name,
        )
        return None

    groups = _group_case_ids_by_design(case_definitions)
    design_means: dict[str, float] = {}

    for design, case_ids in sorted(groups.items()):
        values = [
            _metric_from_case_result(case_results[cid], metric_name)
            for cid in case_ids
            if cid in case_results and case_results[cid].success
        ]
        mean_val = _mean_of_available(values)
        if mean_val is not None:
            design_means[design] = mean_val

    if not design_means:
        logger.warning(
            "plot_metric_by_design (%s): no valid data; skipping plot.",
            metric_name,
        )
        return None

    output_path = Path(output_path)
    _ensure_directory(output_path.parent)

    fig, ax = plt.subplots()
    designs = list(design_means.keys())
    means   = [design_means[d] for d in designs]

    ax.bar(designs, means)
    ax.set_xlabel("Design Type")
    ax.set_ylabel(y_label)
    ax.set_title(f"{y_label} by Design Type (mean across successful cases)")
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)

    logger.info("Plot saved: %s", output_path)
    return str(output_path)


def plot_stress_strain_curves(
    case_results: Mapping[str, CaseResult],
    case_definitions: Mapping[str, CaseDefinition] | None,
    output_directory: str | Path,
    max_cases: int = 10,
) -> list[str]:
    """
    Create one stress-strain curve figure per case that has data.

    ARCHITECTURAL DECISION — one figure per case (no subplots):
        Each case's curve is saved as an independent PNG so it can be
        embedded in per-case reports or slide decks without cropping.

    Args:
        case_results:      All case results.
        case_definitions:  Optional; used for plot titles.
        output_directory:  Directory to write PNGs into.
        max_cases:         Maximum number of curves to generate.

    Returns:
        List of written PNG file paths.
    """
    output_dir = Path(output_directory)
    _ensure_directory(output_dir)
    written: list[str] = []
    count = 0

    for case_id, cr in case_results.items():
        if count >= max_cases:
            break
        if cr.metrics is None:
            continue
        points = cr.metrics.stress_strain_points
        if not points or len(points) < 2:
            continue

        strains = [p[0] for p in points]
        stresses = [p[1] for p in points]

        title = case_id
        if case_definitions is not None:
            defs = case_definitions
            if defs is not None:
                cd = defs.get(case_id)
            if cd is not None:
                title = (
                f"{cd.design_type.value} | {cd.material.name} | "
                f"{cd.load_case.load_case_type.value} | "
                f"t={cd.plate_thickness:.1f}mm"
            )

        out_path = output_dir / f"{case_id}_stress_strain.png"
        fig, ax = plt.subplots()
        ax.plot(strains, stresses, marker="o", markersize=3)
        ax.set_xlabel("Strain [-]")
        ax.set_ylabel("Stress [MPa]")
        ax.set_title(f"Stress-Strain: {title}")
        plt.tight_layout()
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)

        written.append(str(out_path))
        count += 1

    if written:
        logger.info(
            "Stress-strain curves written: %d files → %s", len(written), output_dir
        )
    return written


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def _load_reporting_config(project_root: str | Path | None) -> dict[str, Any]:
    """
    Load the ``reporting`` section from base_config.yaml.

    Falls back to an empty dict (all defaults enabled) on any failure.
    """
    try:
        from utils.config_loader import load_pipeline_config
        cfg = load_pipeline_config(project_root)
        return cfg.reporting
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Main orchestration function
# ---------------------------------------------------------------------------

def generate_reports(
    case_results: Mapping[str, CaseResult],
    ranking_result: RankingResult | None = None,
    case_definitions: Mapping[str, CaseDefinition] | None = None,
    project_root: str | Path | None = None,
    reports_dir: str | Path | None = None,
    top_n_filtered: int = 10,
) -> ReportingResult:
    """
    Generate all version-1 reports for one pipeline run.

    Writes:
      - ``summary.csv``              (always)
      - ``ranked_results.csv``       (when ranking_result provided)
      - ``design_comparison.csv``    (always)
      - ``filtered_candidates.csv``  (when ranking_result provided)
      - Four metric bar charts       (when data available)
      - Per-case stress-strain plots (when data available)

    Args:
        case_results:     Mapping of ``case_id → CaseResult``.
        ranking_result:   Optional completed ``RankingResult``.
        case_definitions: Optional mapping of ``case_id → CaseDefinition``.
        project_root:     Project root for path resolution (optional).
        reports_dir:      Override for reports directory (optional).
        top_n_filtered:   Number of top candidates for filtered CSV.

    Returns:
        ``ReportingResult`` with ``written_files`` and ``warnings``.
    """
    reporting_cfg = _load_reporting_config(project_root)
    paths = resolve_report_paths(project_root, reports_dir)
    result = ReportingResult(success=False)
    result.metadata["report_paths"] = paths.to_dict()

    def _enabled(key: str, default: bool = True) -> bool:
        val = reporting_cfg.get(key, default)
        return bool(val)

    # --- Summary CSV ---
    if _enabled("write_summary_csv"):
        try:
            written = write_summary_csv(
                case_results, paths.summary_csv, case_definitions
            )
            result.written_files.append(written)
        except Exception as exc:
            result.warnings.append(f"summary.csv failed: {exc}")

    # --- Ranked results CSV ---
    if ranking_result is not None and _enabled("write_ranked_results_csv"):
        try:
            written = write_ranked_results_csv(
                ranking_result, paths.ranked_results_csv,
                case_results, case_definitions,
            )
            result.written_files.append(written)
        except Exception as exc:
            result.warnings.append(f"ranked_results.csv failed: {exc}")
    elif ranking_result is None:
        result.warnings.append(
            "ranked_results.csv not written: no ranking_result provided."
        )

    # --- Design comparison CSV ---
    if _enabled("write_design_comparison_csv"):
        try:
            written = write_design_comparison_csv(
                case_results, paths.design_comparison_csv, case_definitions
            )
            result.written_files.append(written)
        except Exception as exc:
            result.warnings.append(f"design_comparison.csv failed: {exc}")

    # --- Filtered candidates CSV ---
    if ranking_result is not None and _enabled("write_filtered_candidates_csv"):
        try:
            written = write_filtered_candidates_csv(
                ranking_result, paths.filtered_candidates_csv, top_n_filtered
            )
            result.written_files.append(written)
        except Exception as exc:
            result.warnings.append(f"filtered_candidates.csv failed: {exc}")

    # --- Metric bar charts ---
    if _enabled("generate_plots"):
        plot_specs = [
            ("effective_modulus_mpa",        "Effective Modulus [MPa]",         "modulus_by_design.png"),
            ("max_von_mises_stress_mpa",      "Max von Mises Stress [MPa]",      "max_stress_by_design.png"),
            ("fatigue_risk_score",            "Fatigue Risk Score (proxy)",       "fatigue_metric_by_design.png"),
            ("max_displacement_mm",           "Max Displacement [mm]",           "displacement_by_design.png"),
        ]
        for metric_attr, y_label, filename in plot_specs:
            out = Path(paths.plots_dir) / filename
            try:
                path_written = plot_metric_by_design(
                    case_results, case_definitions,
                    metric_attr, y_label, out,
                )
                if path_written:
                    result.written_files.append(path_written)
                else:
                    result.warnings.append(
                        f"Plot '{filename}' skipped: insufficient data for '{metric_attr}'."
                    )
            except Exception as exc:
                result.warnings.append(f"Plot '{filename}' failed: {exc}")

        # Stress-strain curves
        try:
            ss_written = plot_stress_strain_curves(
                case_results, case_definitions,
                paths.stress_strain_curves_dir,
            )
            result.written_files.extend(ss_written)
            if not ss_written:
                result.warnings.append(
                    "Stress-strain curves skipped: no cases with sufficient S-S data."
                )
        except Exception as exc:
            result.warnings.append(f"Stress-strain curve generation failed: {exc}")

    result.success = True
    logger.info(
        "Reporting complete: %d files written, %d warnings.",
        len(result.written_files), len(result.warnings),
    )
    return result
