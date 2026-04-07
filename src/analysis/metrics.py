"""
src/analysis/metrics.py
=========================
Engineering-metrics computation module for the auxetic plate pipeline.

This module takes postprocessed solver outputs and computes the consistent
version-1 engineering metrics used by fatigue-proxy analysis, ranking,
and reporting.

PIPELINE POSITION:
    postprocess.py  →  [THIS MODULE]  →  ranking.py
                                      →  reporting.py
                                      →  CaseResult.metrics

ARCHITECTURAL DECISION — transparent metric computation, no fabrication:
    Every derived metric is computed from explicit inputs only.  When a
    required input (e.g. applied force, displacement) is missing, the
    metric is returned as ``None`` with an explanatory warning.  This
    module never substitutes a placeholder number for an unavailable result.

ARCHITECTURAL DECISION — fatigue integration is optional and clearly gated:
    ``compute_engineering_metrics`` calls ``compute_fatigue_risk_proxy``
    only when a ``MaterialRecord`` is provided.  Without material context,
    the fatigue score is left as ``None`` and a warning is added.  This
    means the metrics layer does not require the full material library to
    be loaded; it degrades gracefully.

ARCHITECTURAL DECISION — effective modulus proxy ≠ constitutive material modulus:
    The ``compute_effective_modulus_proxy`` function estimates a structural
    apparent modulus from the slope of the early stress-strain response.
    This is an engineering approximation of the plate's apparent axial
    stiffness per unit area, not the material's Young's modulus.  It is
    labelled ``effective_modulus_mpa`` to distinguish it from the material
    elastic modulus (``elastic_modulus_mpa``) in ``MaterialDefinition``.

UNITS (consistent with project-wide convention):
    Lengths / displacements : mm
    Forces                  : N
    Stresses / moduli       : MPa
    Stiffness               : N/mm
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from analysis.postprocess import PostprocessResult
from simulation.fatigue_model import (
    FatigueProxyInputs,
    FatigueProxyResult,
    FatigueProxyWeights,
    compute_fatigue_risk_proxy,
    fatigue_inputs_from_scalar_results,
)
from workflow.case_schema import MetricSet


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class MetricsComputationError(Exception):
    """
    Raised only when metrics computation fails completely.

    Partial failures (missing individual metrics) are represented by
    ``None`` values and warnings in ``MetricsComputationResult``.
    """


# ---------------------------------------------------------------------------
# Typed dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MetricsComputationResult:
    """
    Structured result of the engineering metrics computation for one case.

    Attributes:
        success:              True if at least some metrics were computed.
        metric_set:           Typed metric values (see ``MetricSet``).
        fatigue_proxy_result: Full fatigue proxy result (if computed).
        warnings:             Non-fatal notes about missing or approximated metrics.
        metadata:             Intermediate and supporting values for reporting.
        error_message:        Error description if ``success`` is False.
    """

    success: bool
    metric_set: MetricSet = field(default_factory=MetricSet)
    fatigue_proxy_result: FatigueProxyResult | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success":               self.success,
            "metric_set":            self.metric_set.to_dict(),
            "fatigue_proxy_result":  (
                self.fatigue_proxy_result.to_dict()
                if self.fatigue_proxy_result else None
            ),
            "warnings":              self.warnings,
            "metadata":              self.metadata,
            "error_message":         self.error_message,
        }


# ---------------------------------------------------------------------------
# Helper: safe peak
# ---------------------------------------------------------------------------

def safe_peak_value(values: list[float | None]) -> float | None:
    """
    Return the maximum finite value from a list, or ``None`` if none exist.

    Args:
        values: List of floats or Nones.

    Returns:
        Max finite value, or ``None``.
    """
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return max(finite) if finite else None


# ---------------------------------------------------------------------------
# Helper: clean stress-strain points
# ---------------------------------------------------------------------------

def clean_stress_strain_points(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """
    Clean a list of (strain, stress_mpa) pairs.

    Removes:
      - pairs with non-finite values
      - exact duplicate pairs (preserves first occurrence order)

    Args:
        points: Raw list of (strain, stress) tuples.

    Returns:
        Cleaned list, preserving original order.
    """
    seen: set[tuple[float, float]] = set()
    cleaned: list[tuple[float, float]] = []
    for strain, stress in points:
        if not (math.isfinite(strain) and math.isfinite(stress)):
            continue
        pair = (strain, stress)
        if pair not in seen:
            seen.add(pair)
            cleaned.append(pair)
    return cleaned


# ---------------------------------------------------------------------------
# Core derived metric helpers
# ---------------------------------------------------------------------------

def compute_effective_stiffness(
    force_n: float | None,
    displacement_mm: float | None,
) -> float | None:
    """
    Compute effective structural stiffness [N/mm].

    Formula:
        stiffness = force_n / displacement_mm

    ARCHITECTURAL DECISION — direct ratio, not secant modulus:
        This is a global structural stiffness proxy (total applied force /
        peak displacement), not a local secant stiffness or material modulus.
        It is useful for comparing relative flexibility between designs under
        the same load case.

    Args:
        force_n:          Applied force [N].
        displacement_mm:  Maximum displacement [mm].

    Returns:
        Stiffness [N/mm], or ``None`` if either input is missing, zero, or
        non-finite.
    """
    if force_n is None:
        return None
    if displacement_mm is None:
        return None
    if not (math.isfinite(force_n) and math.isfinite(displacement_mm)):
        return None
    if displacement_mm == 0.0:
        return None
    if force_n <= 0.0 or displacement_mm <= 0.0:
        return None
    return force_n / displacement_mm


def compute_effective_modulus_proxy(
    stress_strain_points: list[tuple[float, float]],
) -> float | None:
    """
    Compute a practical slope-based effective modulus proxy [MPa] from early
    stress-strain data.

    ARCHITECTURAL DECISION — this is a structural apparent modulus proxy:
        It estimates the average slope of the early stress-strain response,
        not the material's constitutive elastic modulus.  Returned as
        ``effective_modulus_mpa`` to distinguish it from ``elastic_modulus_mpa``
        in the material library.

    Strategy:
      - Clean the points.
      - Use the first and last of the first three valid points.
      - If fewer than 2 points exist, return None.
      - If delta_strain ≤ 0, return None.

    Args:
        stress_strain_points: List of (strain, stress_mpa) pairs.

    Returns:
        Effective modulus proxy [MPa], or ``None`` if insufficient data.
    """
    cleaned = clean_stress_strain_points(stress_strain_points)
    if len(cleaned) < 2:
        return None

    # Use first few points to estimate initial slope
    sample = [cleaned[i] for i in range(min(5, len(cleaned)))]

    # Least-squares slope over the sample for robustness
    n = len(sample)
    if n < 2:
        return None

    strains = [p[0] for p in sample]
    stresses = [p[1] for p in sample]

    d_strain = strains[-1] - strains[0]
    if d_strain <= 0.0:
        return None

    # Simple rise/run slope estimate
    d_stress = stresses[-1] - stresses[0]
    modulus = d_stress / d_strain

    if not math.isfinite(modulus) or modulus <= 0.0:
        return None

    return modulus


def compute_hotspot_ratio(
    hotspot_stress_mpa: float | None,
    max_von_mises_stress_mpa: float | None,
) -> float | None:
    """
    Compute the hotspot-to-max-stress ratio (dimensionless).

    Useful for quantifying stress concentration relative to the global peak.
    A ratio > 1.0 indicates the hotspot exceeds the global max (which can
    occur if different stress metrics are used for each).

    Args:
        hotspot_stress_mpa:       Local hotspot stress [MPa].
        max_von_mises_stress_mpa: Global peak von Mises stress [MPa].

    Returns:
        Hotspot ratio, or ``None`` if either value is missing or zero.
    """
    if hotspot_stress_mpa is None:
        return None
    if max_von_mises_stress_mpa is None:
        return None
    if max_von_mises_stress_mpa <= 0.0:
        return None
    return hotspot_stress_mpa / max_von_mises_stress_mpa


# ---------------------------------------------------------------------------
# Postprocess → MetricSet direct mapping
# ---------------------------------------------------------------------------

def metric_set_from_postprocess(
    postprocess_result: PostprocessResult,
) -> MetricSet:
    """
    Map directly available postprocessed results into a ``MetricSet``.

    This is a pure pass-through — no new computations are performed here.
    Derived metrics (stiffness, modulus, fatigue) are added later by
    ``compute_engineering_metrics``.

    Args:
        postprocess_result: Completed postprocess result from ``postprocess.py``.

    Returns:
        ``MetricSet`` with directly mapped values (derived fields remain None).
    """
    m = postprocess_result.metrics
    cleaned_ss = clean_stress_strain_points(postprocess_result.stress_strain_points)
    return MetricSet(
        max_von_mises_stress_mpa=m.get("max_von_mises_stress_mpa"),
        max_displacement_mm=m.get("max_displacement_mm"),
        hotspot_stress_mpa=m.get("hotspot_stress_mpa"),
        stress_strain_points=cleaned_ss,
        # Derived fields left as None; filled by compute_engineering_metrics
        effective_stiffness_n_per_mm=None,
        effective_modulus_mpa=None,
        fatigue_risk_score=None,
    )


# ---------------------------------------------------------------------------
# Fatigue input bridge
# ---------------------------------------------------------------------------

def build_fatigue_inputs_from_postprocess(
    postprocess_result: PostprocessResult,
    loadcase: Any | None = None,
) -> FatigueProxyInputs:
    """
    Build ``FatigueProxyInputs`` from a postprocessed result.

    Maps:
      - ``max_von_mises_stress_mpa`` → ``max_von_mises_stress_mpa``
      - ``hotspot_stress_mpa``       → ``hotspot_stress_mpa``
      - ``stress_amplitude_mpa``     → from metadata if present, else None
      - ``mean_stress_mpa``          → from metadata if present, else None

    Args:
        postprocess_result: Completed postprocess result.
        loadcase:           Optional load-case record for amplitude inference.

    Returns:
        ``FatigueProxyInputs`` ready for the fatigue model.
    """
    m = postprocess_result.metrics
    meta = postprocess_result.metadata

    return fatigue_inputs_from_scalar_results(
        max_von_mises_stress_mpa=m.get("max_von_mises_stress_mpa"),
        hotspot_stress_mpa=m.get("hotspot_stress_mpa"),
        # stress_amplitude may have been parsed or inferred; check metadata
        stress_amplitude_mpa=meta.get("stress_amplitude_mpa"),
        mean_stress_mpa=meta.get("mean_stress_mpa"),
        loadcase=loadcase,
    )


# ---------------------------------------------------------------------------
# Main metrics computation function
# ---------------------------------------------------------------------------

def compute_engineering_metrics(
    postprocess_result: PostprocessResult,
    applied_force_n: float | None = None,
    material: Any | None = None,
    loadcase: Any | None = None,
    fatigue_weights: FatigueProxyWeights | None = None,
) -> MetricsComputationResult:
    """
    Compute the full engineering metric set for one pipeline case.

    Derived metrics:
      - ``effective_stiffness_n_per_mm``: force / displacement (if both known)
      - ``effective_modulus_mpa``:        slope of early stress-strain response
      - ``fatigue_risk_score``:           from fatigue_model.py (if material given)

    Args:
        postprocess_result: Completed ``PostprocessResult`` from postprocess.py.
        applied_force_n:    Applied force [N] for stiffness computation.
        material:           ``MaterialRecord`` for fatigue proxy (optional).
        loadcase:           ``LoadCaseRecord`` for fatigue amplitude inference.
        fatigue_weights:    Custom ``FatigueProxyWeights`` (optional).

    Returns:
        ``MetricsComputationResult`` with all computable metrics filled in.
    """
    warnings: list[str] = []
    metadata: dict[str, Any] = {}

    # --- Start from postprocessed base metrics ---
    ms = metric_set_from_postprocess(postprocess_result)

    # --- Effective stiffness ---
    force = applied_force_n
    if force is None and loadcase is not None:
        # Duck-type: try to read force from the loadcase record
        force = getattr(loadcase, "force_n", None)
        if force is None:
            force = getattr(loadcase, "mean_force_n", None)

    stiffness = compute_effective_stiffness(force, ms.max_displacement_mm)
    if stiffness is not None:
        ms.effective_stiffness_n_per_mm = stiffness
        metadata["stiffness_force_n"] = force
        metadata["stiffness_disp_mm"] = ms.max_displacement_mm
    else:
        if force is None:
            warnings.append(
                "Effective stiffness not computed: applied_force_n is None.  "
                "Pass applied_force_n or a loadcase record with force_n."
            )
        elif ms.max_displacement_mm is None:
            warnings.append(
                "Effective stiffness not computed: max_displacement_mm is None.  "
                "Ensure *NODE FILE output includes U in the solver input deck."
            )
        else:
            warnings.append(
                "Effective stiffness not computed: zero or non-positive "
                "force or displacement values."
            )

    # --- Effective modulus proxy ---
    cleaned_ss = clean_stress_strain_points(ms.stress_strain_points)
    modulus = compute_effective_modulus_proxy(cleaned_ss)
    if modulus is not None:
        ms.effective_modulus_mpa = modulus
        metadata["cleaned_stress_strain_point_count"] = len(cleaned_ss)
    else:
        warnings.append(
            "Effective modulus proxy not computed: insufficient stress-strain "
            "data points.  Need at least 2 valid (strain, stress) pairs."
        )
        metadata["cleaned_stress_strain_point_count"] = len(cleaned_ss)

    # --- Hotspot ratio (metadata only — not a MetricSet field) ---
    hotspot_ratio = compute_hotspot_ratio(
        ms.hotspot_stress_mpa,
        ms.max_von_mises_stress_mpa,
    )
    if hotspot_ratio is not None:
        metadata["hotspot_ratio"] = float(f"{hotspot_ratio:.4f}")
    else:
        warnings.append(
            "Hotspot ratio not computed: hotspot_stress_mpa or "
            "max_von_mises_stress_mpa is None."
        )

    # --- Fatigue risk proxy ---
    fatigue_result: FatigueProxyResult | None = None

    if material is None:
        warnings.append(
            "Fatigue risk proxy not computed: no material was provided.  "
            "Pass a MaterialRecord to compute the fatigue_risk_score."
        )
    else:
        fatigue_inputs = build_fatigue_inputs_from_postprocess(
            postprocess_result, loadcase
        )

        # Sanity check: if no stress inputs available, skip with a clear warning
        if all(v is None for v in (
            fatigue_inputs.max_von_mises_stress_mpa,
            fatigue_inputs.hotspot_stress_mpa,
            fatigue_inputs.stress_amplitude_mpa,
        )):
            warnings.append(
                "Fatigue risk proxy not computed: no stress values are available "
                "from postprocessing.  Run the solver and verify *EL FILE output."
            )
        else:
            fatigue_result = compute_fatigue_risk_proxy(
                inputs=fatigue_inputs,
                material=material,
                loadcase=loadcase,
                weights=fatigue_weights,
            )

            if fatigue_result.success and fatigue_result.fatigue_risk_score is not None:
                ms.fatigue_risk_score = fatigue_result.fatigue_risk_score
                metadata["fatigue_proxy_category"] = (
                    fatigue_result.metadata.get("risk_category", "unknown")
                )
                metadata["fatigue_governing_metric"] = (
                    fatigue_result.governing_metric
                )
                # Propagate fatigue warnings
                warnings.extend(
                    f"[fatigue_proxy] {w}" for w in fatigue_result.warnings
                )
            else:
                warnings.append(
                    f"Fatigue proxy returned success=False: "
                    f"{fatigue_result.error_message}"
                )
                warnings.extend(
                    f"[fatigue_proxy] {w}" for w in fatigue_result.warnings
                )

    result = MetricsComputationResult(
        success=True,
        metric_set=ms,
        fatigue_proxy_result=fatigue_result,
        warnings=warnings,
        metadata=metadata,
    )
    return result


# ---------------------------------------------------------------------------
# Convenience wrapper from scalar inputs
# ---------------------------------------------------------------------------

def compute_metrics_from_scalars(
    max_von_mises_stress_mpa: float | None = None,
    max_displacement_mm: float | None = None,
    hotspot_stress_mpa: float | None = None,
    stress_strain_points: list[tuple[float, float]] | None = None,
    applied_force_n: float | None = None,
    material: Any | None = None,
    loadcase: Any | None = None,
) -> MetricsComputationResult:
    """
    Compute engineering metrics from individual scalar inputs.

    Convenience wrapper for tests and simple integration — constructs a
    ``PostprocessResult`` from the given scalars and delegates to
    ``compute_engineering_metrics``.

    Args:
        max_von_mises_stress_mpa: Peak von Mises stress [MPa].
        max_displacement_mm:      Max nodal displacement [mm].
        hotspot_stress_mpa:       Hotspot / local peak stress [MPa].
        stress_strain_points:     List of (strain, stress_mpa) pairs.
        applied_force_n:          Applied load [N] for stiffness.
        material:                 ``MaterialRecord`` for fatigue (optional).
        loadcase:                 ``LoadCaseRecord`` for fatigue (optional).

    Returns:
        ``MetricsComputationResult``.
    """
    from analysis.postprocess import PostprocessResult

    synthetic_metrics: dict[str, Any] = {}
    if max_von_mises_stress_mpa is not None:
        synthetic_metrics["max_von_mises_stress_mpa"] = max_von_mises_stress_mpa
    if max_displacement_mm is not None:
        synthetic_metrics["max_displacement_mm"] = max_displacement_mm
    if hotspot_stress_mpa is not None:
        synthetic_metrics["hotspot_stress_mpa"] = hotspot_stress_mpa

    postprocess_result = PostprocessResult(
        success=True,
        metrics=synthetic_metrics,
        stress_strain_points=stress_strain_points or [],
    )

    return compute_engineering_metrics(
        postprocess_result=postprocess_result,
        applied_force_n=applied_force_n,
        material=material,
        loadcase=loadcase,
    )


# ---------------------------------------------------------------------------
# Strict wrapper
# ---------------------------------------------------------------------------

def require_engineering_metrics(
    postprocess_result: PostprocessResult,
    applied_force_n: float | None = None,
    material: Any | None = None,
    loadcase: Any | None = None,
    fatigue_weights: FatigueProxyWeights | None = None,
) -> MetricsComputationResult:
    """
    Compute engineering metrics and raise ``MetricsComputationError`` if
    the computation fails entirely.

    ARCHITECTURAL DECISION — partial success is not a failure:
        If ``compute_engineering_metrics`` returns ``success=True`` but some
        metrics are ``None`` (e.g. no stiffness because displacement was
        unavailable), that is a valid partial result and this function does
        NOT raise.  Raising is reserved for hard failures (``success=False``).

    Args:
        See ``compute_engineering_metrics`` for argument descriptions.

    Returns:
        ``MetricsComputationResult`` with ``success=True``.

    Raises:
        MetricsComputationError: if ``success=False``.
    """
    result = compute_engineering_metrics(
        postprocess_result=postprocess_result,
        applied_force_n=applied_force_n,
        material=material,
        loadcase=loadcase,
        fatigue_weights=fatigue_weights,
    )
    if not result.success:
        raise MetricsComputationError(
            f"Engineering metrics computation failed: {result.error_message}"
        )
    return result
