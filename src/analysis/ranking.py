"""
src/analysis/ranking.py
=========================
Design-ranking module for the auxetic plate pipeline.

This module takes structured case metric results, normalises comparable
quantities within the current candidate set, applies configurable weighted
scoring, and returns transparent ranked outputs.

PIPELINE POSITION:
    metrics.py  →  [THIS MODULE]  →  reporting.py
                                  →  CaseResult comparisons

RANKING PHILOSOPHY:
    Rankings are relative within the current sweep's result set — they are
    NOT universal absolute quality scores.  A case that scores 1.0 is the
    best in this run, not the best possible auxetic plate design.  All
    normalisation is per-run min-max, so adding or removing cases changes
    absolute rank scores but not the relative ordering within the kept set.

ARCHITECTURAL DECISION — min-max normalisation with direction inversion:
    For each metric, valid values (non-None) are collected across all cases.
    Min-max normalisation maps the range [min, max] to [0, 1].  Direction
    inversion then ensures "higher normalised score = better outcome" for
    every metric regardless of whether the raw metric is "lower is better"
    or "higher is better":
        lower_is_better:  normalised = (max − x) / (max − min)
        higher_is_better: normalised = (x − min)  / (max − min)

ARCHITECTURAL DECISION — per-case active-weight renormalisation for
missing metrics:
    If a case is missing a metric (value is None), that component is
    excluded from the weighted sum for that case only.  The remaining
    active weights are renormalised so they still sum to 1.0 before
    computing that case's total score.  This is fairer than assigning a
    zero score for an absent metric (which would over-penalise cases that
    simply did not run a particular load case).  Missing metrics are
    recorded in ``RankedCase.missing_metrics`` and warnings.

ARCHITECTURAL DECISION — equal-value edge case assigns neutral score:
    When all valid values for a metric are equal (zero range), normalisation
    is undefined.  All valid cases receive a neutral score of 0.5 for that
    component, and the situation is flagged in ``normalization_metadata``.

UNITS: consistent with MetricSet (MPa, mm, N/mm).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from workflow.case_schema import CaseResult, MetricSet

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Default metric directions — lower values are better unless noted.
# ARCHITECTURAL DECISION — explicit direction map rather than sign convention:
#   Using a string-keyed dict makes it easy for callers to inspect, override,
#   or document the direction assumptions without reading source code.
_DEFAULT_METRIC_DIRECTIONS: dict[str, str] = {
    "fatigue_risk":           "lower_is_better",
    "max_von_mises_stress":   "lower_is_better",
    "effective_stiffness":    "higher_is_better",  # stiffer plate = better support
    "max_displacement":       "lower_is_better",
    "effective_modulus":      "higher_is_better",
    "hotspot_stress":         "lower_is_better",
}

# Default neutral score assigned when a metric has zero variance
_NEUTRAL_NORMALISED_SCORE: float = 0.5


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class RankingError(Exception):
    """
    Raised when ranking fails completely.

    Partial failures (missing metrics for individual cases) are represented
    by warnings in ``RankedCase`` rather than raising.
    """


# ---------------------------------------------------------------------------
# Typed dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RankingWeights:
    """
    Configurable weights for the four core ranking metrics.

    ARCHITECTURAL DECISION — weights need not sum to 1.0:
        ``rank_cases()`` normalises weights internally so callers can set
        intuitive relative values (e.g. 2:1:1:1) without forcing exact
        fractions.  Normalisation is reported in ``RankingResult.normalization_metadata``.

    These defaults are aligned with ``ranking_defaults.weights`` in
    ``base_config.yaml``.
    """

    fatigue_risk:         float = 0.30
    max_von_mises_stress: float = 0.30
    effective_stiffness:  float = 0.20
    max_displacement:     float = 0.20

    def validate(self) -> None:
        """
        Validate that all weights are non-negative and at least one is positive.

        Raises:
            RankingError: if any weight is negative or all are zero.
        """
        fields = {
            "fatigue_risk":         self.fatigue_risk,
            "max_von_mises_stress": self.max_von_mises_stress,
            "effective_stiffness":  self.effective_stiffness,
            "max_displacement":     self.max_displacement,
        }
        for name, value in fields.items():
            if value < 0.0:
                raise RankingError(
                    f"RankingWeights.{name} must be non-negative, got {value}."
                )
        if sum(fields.values()) <= 0.0:
            raise RankingError(
                "All RankingWeights are zero.  At least one must be positive."
            )

    def to_dict(self) -> dict[str, float]:
        return {
            "fatigue_risk":         self.fatigue_risk,
            "max_von_mises_stress": self.max_von_mises_stress,
            "effective_stiffness":  self.effective_stiffness,
            "max_displacement":     self.max_displacement,
        }


@dataclass
class RankedCase:
    """
    Ranking entry for one pipeline case.

    Attributes:
        case_id:                    Case identifier.
        rank:                       Rank in the sorted list (1 = best).
        total_score:                Weighted normalised score in [0, 1].
        normalized_component_scores: Per-metric normalised score after
                                    direction inversion.
        raw_metric_values:          Original extracted metric values.
        missing_metrics:            Metric keys absent for this case.
        warnings:                   Per-case ranking notes.
        metadata:                   Supporting details for traceability.
    """

    case_id: str
    rank: int | None = None
    total_score: float | None = None
    normalized_component_scores: dict[str, float] = field(default_factory=dict)
    raw_metric_values: dict[str, float | None] = field(default_factory=dict)
    missing_metrics: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id":                       self.case_id,
            "rank":                          self.rank,
            "total_score":                   self.total_score,
            "normalized_component_scores":   self.normalized_component_scores,
            "raw_metric_values":             self.raw_metric_values,
            "missing_metrics":               self.missing_metrics,
            "warnings":                      self.warnings,
            "metadata":                      self.metadata,
        }


@dataclass
class RankingResult:
    """
    Complete ranking output for a candidate set.

    Attributes:
        success:                 True if ranking completed.
        ranked_cases:            Cases sorted best-first (rank 1 = highest score).
        weights:                 Effective (normalised) weights used.
        normalization_metadata:  Per-metric normalisation details.
        warnings:                Global ranking warnings.
        error_message:           Error description if ``success`` is False.
    """

    success: bool
    ranked_cases: list[RankedCase] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    normalization_metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success":                  self.success,
            "ranked_cases":             [c.to_dict() for c in self.ranked_cases],
            "weights":                  self.weights,
            "normalization_metadata":   self.normalization_metadata,
            "warnings":                 self.warnings,
            "error_message":            self.error_message,
        }


# ---------------------------------------------------------------------------
# Metric extraction helper
# ---------------------------------------------------------------------------

def extract_rankable_metrics(metric_set: MetricSet) -> dict[str, float | None]:
    """
    Extract the core rankable metric values from a ``MetricSet``.

    The four primary ranking metrics are:
        - ``fatigue_risk``          ← ``fatigue_risk_score``
        - ``max_von_mises_stress``  ← ``max_von_mises_stress_mpa``
        - ``effective_stiffness``   ← ``effective_stiffness_n_per_mm``
        - ``max_displacement``      ← ``max_displacement_mm``

    Optional secondary metrics (included but lower priority):
        - ``effective_modulus``     ← ``effective_modulus_mpa``
        - ``hotspot_stress``        ← ``hotspot_stress_mpa``

    Args:
        metric_set: ``MetricSet`` from ``case_schema.py``.

    Returns:
        Dict mapping ranking metric key → raw value (or None).
    """
    return {
        "fatigue_risk":         metric_set.fatigue_risk_score,
        "max_von_mises_stress": metric_set.max_von_mises_stress_mpa,
        "effective_stiffness":  metric_set.effective_stiffness_n_per_mm,
        "max_displacement":     metric_set.max_displacement_mm,
        # Secondary metrics — not weighted by default but available for
        # inspection and optional custom weight configurations.
        "effective_modulus":    metric_set.effective_modulus_mpa,
        "hotspot_stress":       metric_set.hotspot_stress_mpa,
    }


# ---------------------------------------------------------------------------
# Direction map helper
# ---------------------------------------------------------------------------

def default_metric_directions() -> dict[str, str]:
    """
    Return the default metric direction mapping for ranking.

    Returns:
        Dict mapping metric key → ``"lower_is_better"`` or
        ``"higher_is_better"``.
    """
    return dict(_DEFAULT_METRIC_DIRECTIONS)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _safe_min_max(values: list[float]) -> tuple[float, float] | None:
    """
    Return ``(min, max)`` of a finite float list, or ``None`` if empty.

    Args:
        values: List of finite float values.

    Returns:
        Tuple ``(min, max)`` or ``None`` if the list is empty.
    """
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return None
    return min(finite), max(finite)


def normalise_metric_column(
    case_values: dict[str, float | None],
    direction: str,
) -> tuple[dict[str, float | None], dict[str, Any]]:
    """
    Normalise one metric column across cases to a [0, 1] "higher = better" scale.

    ARCHITECTURAL DECISION — normalisation is within the current candidate set:
        Adding or removing cases changes absolute scores but not relative ordering.

    Args:
        case_values:  Mapping of ``case_id → raw_value`` (None allowed).
        direction:    ``"lower_is_better"`` or ``"higher_is_better"``.

    Returns:
        Tuple:
          - ``normalised``: same keys, values in [0,1] (None if original was None)
          - ``meta``: dict with ``min``, ``max``, ``valid_count``,
            ``equal_values_detected``, ``direction``
    """
    valid_values = [v for v in case_values.values() if v is not None and math.isfinite(v)]
    meta: dict[str, Any] = {
        "direction": direction,
        "valid_count": len(valid_values),
        "equal_values_detected": False,
    }

    if not valid_values:
        meta["min"] = None
        meta["max"] = None
        normalised = {k: None for k in case_values}
        return normalised, meta

    v_min, v_max = min(valid_values), max(valid_values)
    meta["min"] = v_min
    meta["max"] = v_max

    normalised: dict[str, float | None] = {}
    v_range = v_max - v_min

    if v_range == 0.0:
        # All valid values are equal — assign neutral score to all
        meta["equal_values_detected"] = True
        for case_id, value in case_values.items():
            normalised[case_id] = _NEUTRAL_NORMALISED_SCORE if value is not None else None
        return normalised, meta

    for case_id, value in case_values.items():
        if value is None or not math.isfinite(value):
            normalised[case_id] = None
        elif direction == "lower_is_better":
            # Best (lowest raw) → normalised 1.0; worst (highest raw) → 0.0
            normalised[case_id] = (v_max - value) / v_range
        elif direction == "higher_is_better":
            # Best (highest raw) → normalised 1.0; worst (lowest raw) → 0.0
            normalised[case_id] = (value - v_min) / v_range
        else:
            # Unknown direction — treat as neutral
            normalised[case_id] = _NEUTRAL_NORMALISED_SCORE

    return normalised, meta


# ---------------------------------------------------------------------------
# Weight loading helper
# ---------------------------------------------------------------------------

def load_ranking_weights(
    project_root: str | Path | None = None,
) -> RankingWeights:
    """
    Load ranking weights from ``base_config.yaml`` via the config loader.

    Reads ``ranking_defaults.weights`` from the base config.  Falls back
    to ``RankingWeights()`` defaults if the section is absent or the
    config cannot be loaded.

    Args:
        project_root: Project root for config resolution (optional).

    Returns:
        ``RankingWeights`` populated from config or defaults.
    """
    try:
        from utils.config_loader import load_pipeline_config
        cfg = load_pipeline_config(project_root)
        rd = cfg.ranking_defaults
        w = rd.get("weights", {})
        return RankingWeights(
            fatigue_risk=float(w.get("fatigue_risk", 0.30)),
            max_von_mises_stress=float(w.get("max_von_mises_stress", 0.30)),
            effective_stiffness=float(w.get("effective_stiffness", 0.20)),
            max_displacement=float(w.get("displacement", 0.20)),
        )
    except Exception:
        return RankingWeights()


# ---------------------------------------------------------------------------
# Main ranking function
# ---------------------------------------------------------------------------

def rank_cases(
    case_metrics: Mapping[str, MetricSet],
    weights: RankingWeights | None = None,
    metric_directions: dict[str, str] | None = None,
) -> RankingResult:
    """
    Rank a collection of cases by weighted normalised metric scores.

    The four primary metrics weighted by ``weights`` are:
        - ``fatigue_risk``
        - ``max_von_mises_stress``
        - ``effective_stiffness``
        - ``max_displacement``

    Steps:
        1. Extract raw rankable metrics for each case.
        2. For each metric, normalise across all cases with valid values.
        3. For each case, compute a weighted sum using only the metrics
           that have valid values (active-weight renormalisation for partial data).
        4. Sort by total score descending and assign ranks.

    ARCHITECTURAL DECISION — active-weight renormalisation for missing metrics:
        See module docstring.  A case missing one metric does not get a zero
        penalty; its score is computed over the remaining active metrics with
        weights re-scaled to sum to 1.0.

    Args:
        case_metrics:      Mapping of ``case_id → MetricSet``.
        weights:           ``RankingWeights`` (loaded from config if None).
        metric_directions: Direction overrides (uses defaults if None).

    Returns:
        ``RankingResult`` with ranked cases, weights, and normalisation metadata.
    """
    global_warnings: list[str] = []
    norm_meta: dict[str, Any] = {}

    if not case_metrics:
        return RankingResult(
            success=False,
            error_message="No cases provided for ranking.",
        )

    if weights is None:
        weights = load_ranking_weights()

    try:
        weights.validate()
    except RankingError as exc:
        return RankingResult(success=False, error_message=str(exc))

    directions = metric_directions or default_metric_directions()

    # --- Determine which metrics are actually in the weight schema ---
    weight_map: dict[str, float] = {
        k: v for k, v in weights.to_dict().items()
        if v > 0.0
    }

    # Normalise weights to sum to 1.0
    total_weight = sum(weight_map.values())
    effective_weights: dict[str, float] = {
        k: v / total_weight for k, v in weight_map.items()
    }
    norm_meta["effective_weights"] = {k: float(f"{v:.6f}") for k, v in effective_weights.items()}
    norm_meta["original_weights_normalised"] = (abs(total_weight - 1.0) > 1e-6)

    # --- Extract raw metrics per case ---
    all_raw: dict[str, dict[str, float | None]] = {}
    for case_id, ms in case_metrics.items():
        all_raw[case_id] = extract_rankable_metrics(ms)

    # --- Normalise each metric column ---
    # per_metric_normalised: metric_key → {case_id → normalised_score | None}
    per_metric_normalised: dict[str, dict[str, float | None]] = {}

    for metric_key in weight_map:
        column: dict[str, float | None] = {
            case_id: all_raw[case_id].get(metric_key)
            for case_id in all_raw
        }
        direction = directions.get(metric_key, "lower_is_better")
        normalised_col, col_meta = normalise_metric_column(column, direction)
        per_metric_normalised[metric_key] = normalised_col
        norm_meta[metric_key] = col_meta

        if col_meta["valid_count"] == 0:
            global_warnings.append(
                f"Metric '{metric_key}' has no valid values across all cases.  "
                f"This metric will not contribute to any case's score."
            )
        if col_meta.get("equal_values_detected"):
            global_warnings.append(
                f"Metric '{metric_key}': all valid values are equal — "
                f"assigned neutral score ({_NEUTRAL_NORMALISED_SCORE}) to all cases."
            )

    # --- Compute per-case weighted scores ---
    ranked_cases: list[RankedCase] = []

    for case_id in all_raw:
        rc = RankedCase(
            case_id=case_id,
            raw_metric_values=dict(all_raw[case_id]),
        )

        # Identify available and missing weighted metrics for this case
        available: dict[str, float] = {}
        for metric_key, w in effective_weights.items():
            norm_score = per_metric_normalised.get(metric_key, {}).get(case_id)
            if norm_score is not None:
                available[metric_key] = norm_score
                rc.normalized_component_scores[metric_key] = float(f"{norm_score:.6f}")
            else:
                rc.missing_metrics.append(metric_key)
                rc.warnings.append(
                    f"Metric '{metric_key}' is unavailable for this case.  "
                    f"Weight {w:.3f} redistributed to remaining active metrics."
                )

        # Active-weight renormalisation
        active_weight_total = sum(
            effective_weights[mk] for mk in available
        )

        if active_weight_total <= 0.0:
            rc.total_score = None
            rc.warnings.append(
                "No valid metrics available for scoring.  "
                "This case will not be ranked."
            )
        else:
            score = sum(
                (effective_weights[mk] / active_weight_total) * norm_score
                for mk, norm_score in available.items()
            )
            rc.total_score = float(f"{score:.6f}")
            if rc.missing_metrics:
                rc.metadata["active_metrics_fraction"] = float(f"{active_weight_total:.4f}")

        ranked_cases.append(rc)

    # --- Sort by total score descending (None → last) ---
    ranked_cases.sort(
        key=lambda rc: rc.total_score if rc.total_score is not None else -math.inf,
        reverse=True,
    )

    # Assign ranks (cases with None score get no rank)
    valid_cases = [rc for rc in ranked_cases if rc.total_score is not None]
    for i, rc in enumerate(valid_cases, start=1):
        rc.rank = i

    return RankingResult(
        success=True,
        ranked_cases=ranked_cases,
        weights=effective_weights,
        normalization_metadata=norm_meta,
        warnings=global_warnings,
    )


# ---------------------------------------------------------------------------
# CaseResult convenience helper
# ---------------------------------------------------------------------------

def rank_case_results(
    case_results: Mapping[str, CaseResult],
    weights: RankingWeights | None = None,
    metric_directions: dict[str, str] | None = None,
) -> RankingResult:
    """
    Rank ``CaseResult`` objects by extracting their embedded ``MetricSet``.

    Filters out failed/incomplete cases before ranking if their metrics
    are entirely empty.

    Args:
        case_results:      Mapping of ``case_id → CaseResult``.
        weights:           Ranking weights (optional).
        metric_directions: Direction overrides (optional).

    Returns:
        ``RankingResult``.
    """
    metric_map: dict[str, MetricSet] = {}
    for case_id, cr in case_results.items():
        if cr.metrics is not None:
            metric_map[case_id] = cr.metrics

    return rank_cases(metric_map, weights, metric_directions)


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------

def group_ranked_cases_by_design_type(
    ranking_result: RankingResult,
    case_design_map: Mapping[str, str],
) -> dict[str, list[RankedCase]]:
    """
    Group ranked cases by their design type.

    Args:
        ranking_result:   Completed ranking result.
        case_design_map:  Mapping of ``case_id → design_type_string``.

    Returns:
        Dict mapping ``design_type → [RankedCase, ...]``, ordered best-first
        within each group.
    """
    groups: dict[str, list[RankedCase]] = {}
    for rc in ranking_result.ranked_cases:
        design = case_design_map.get(rc.case_id, "unknown")
        groups.setdefault(design, []).append(rc)
    return groups


def best_case_per_design(
    ranking_result: RankingResult,
    case_design_map: Mapping[str, str],
) -> dict[str, RankedCase]:
    """
    Return the highest-scoring case for each design type.

    Args:
        ranking_result:   Completed ranking result.
        case_design_map:  Mapping of ``case_id → design_type_string``.

    Returns:
        Dict mapping ``design_type → best RankedCase``.
    """
    groups = group_ranked_cases_by_design_type(ranking_result, case_design_map)
    best: dict[str, RankedCase] = {}
    for design, cases in groups.items():
        # Cases are already sorted best-first in ranking_result
        ranked = [c for c in cases if c.total_score is not None]
        if ranked:
            best[design] = ranked[0]
    return best


# ---------------------------------------------------------------------------
# Strict wrapper
# ---------------------------------------------------------------------------

def require_ranked_cases(
    case_metrics: Mapping[str, MetricSet],
    weights: RankingWeights | None = None,
    metric_directions: dict[str, str] | None = None,
) -> RankingResult:
    """
    Rank cases and raise ``RankingError`` if ranking fails entirely.

    ARCHITECTURAL DECISION — partial ranking is not a failure:
        Cases with missing metrics are still ranked using their available
        data.  Only a complete failure to produce any ranked output raises.

    Args:
        See ``rank_cases()`` for argument descriptions.

    Returns:
        ``RankingResult`` with ``success=True``.

    Raises:
        RankingError: if ``success=False``.
    """
    result = rank_cases(case_metrics, weights, metric_directions)
    if not result.success:
        raise RankingError(
            f"Ranking failed: {result.error_message}"
        )
    return result
