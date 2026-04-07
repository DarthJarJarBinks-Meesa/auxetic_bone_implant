"""
src/simulation/fatigue_model.py
==================================
Fatigue-risk proxy module for the auxetic plate pipeline.

This module computes a practical, transparent fatigue-risk proxy score
from available FE stress results, cyclic loading context, and material
fatigue properties.

!! IMPORTANT — PROXY ONLY !!
    ALL outputs from this module are fatigue-risk PROXY scores intended
    for COMPARATIVE RANKING of auxetic designs only.
    They are NOT validated fatigue life predictions.
    They are NOT fatigue design allowables.
    They must NOT be used for regulatory, clinical, or certification purposes.

PROXY PHILOSOPHY:
    The score combines four normalized stress ratios, each relative to a
    material-specific threshold.  Lower scores indicate lower relative
    fatigue risk compared to other cases in the same sweep.

    Components:
      (1) Stress amplitude / fatigue limit    — primary fatigue driver
      (2) Max von Mises / yield strength      — structural severity measure
      (3) Hotspot stress / fatigue limit      — local stress concentration penalty
      (4) Mean-stress penalty                 — simple Goodman-style correction hint

    Weighted sum:
        score = w1·R_amp + w2·R_max + w3·R_hot + w4·P_mean

    where each R is a normalized ratio (value / threshold) and P_mean is
    a dimensionless penalty.

    ARCHITECTURAL DECISION — unbounded score, not clamped to [0,1]:
        Individual ratios > 1.0 indicate the stress has exceeded the
        relevant threshold (fatigue limit or yield strength), which is
        physically meaningful information.  Clamping to [0,1] would hide
        this.  The score may therefore exceed 1.0 for high-risk cases.
        Downstream ranking simply orders scores; the absolute magnitude
        is informational.

ARCHITECTURAL DECISION — yield strength as fallback denominator:
    If a material's fatigue_limit_mpa is None (placeholder not yet set),
    a fraction of yield strength is used as the denominator (configurable
    via ``_PLACEHOLDER_FATIGUE_FRACTION``).  This is clearly flagged in
    warnings so the low-confidence result is traceable in reports.

UNITS: MPa throughout for all stress values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from simulation.loadcases import LoadCaseRecord
from simulation.materials import MaterialRecord

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# When fatigue_limit_mpa is None (placeholder), fall back to this fraction
# of yield strength as the effective fatigue limit for proxy scoring.
# Documented in base_config.yaml placeholder_fatigue_limit_fraction_of_yield.
_PLACEHOLDER_FATIGUE_FRACTION: float = 0.50

# When stress_amplitude_mpa is not available, infer it as this fraction of
# max_von_mises_stress_mpa (conservative approximation for fully-reversed cycling).
_AMPLITUDE_FROM_MAX_STRESS_FRACTION: float = 0.50

# Risk category thresholds (heuristic; clearly labelled in categorize_fatigue_risk)
_RISK_THRESHOLDS: dict[str, float] = {
    "low":      0.30,
    "moderate": 0.60,
    "high":     0.90,
    "severe":   math.inf,
}


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class FatigueModelError(Exception):
    """
    Raised when the fatigue proxy model receives invalid inputs or
    encounters a configuration error it cannot recover from.

    Soft failures (missing optional inputs, fallback assumptions) are
    represented by warnings in ``FatigueProxyResult`` rather than
    raising this exception.
    """


# ---------------------------------------------------------------------------
# Typed dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FatigueProxyInputs:
    """
    Stress-like inputs to the fatigue-risk proxy model.

    All stress values are in MPa.  Any field may be ``None`` if not
    available from the solver results; the model handles missing inputs
    via fallback logic and warnings.

    Attributes:
        max_von_mises_stress_mpa: Peak von Mises stress in the model [MPa].
        hotspot_stress_mpa:       Localised peak stress at auxetic nodes [MPa].
        stress_amplitude_mpa:     Half cyclic stress range [MPa].
                                  If None, inferred from max stress.
        mean_stress_mpa:          Mean (static offset) stress [MPa].
        cycle_count:              Reference cycle count (proxy normalisation only).
        frequency_hz:             Cyclic loading frequency [Hz].
        metadata:                 Open dict for additional context.
    """

    max_von_mises_stress_mpa: float | None = None
    hotspot_stress_mpa: float | None = None
    stress_amplitude_mpa: float | None = None
    mean_stress_mpa: float | None = None
    cycle_count: int | None = None
    frequency_hz: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_von_mises_stress_mpa": self.max_von_mises_stress_mpa,
            "hotspot_stress_mpa":       self.hotspot_stress_mpa,
            "stress_amplitude_mpa":     self.stress_amplitude_mpa,
            "mean_stress_mpa":          self.mean_stress_mpa,
            "cycle_count":              self.cycle_count,
            "frequency_hz":             self.frequency_hz,
            "metadata":                 self.metadata,
        }


@dataclass
class FatigueProxyWeights:
    """
    Weighting factors for the four fatigue-risk proxy components.

    Must sum to 1.0 for the score to be dimensionally consistent.  If
    the sum differs slightly from 1.0, the model normalises automatically.

    Attributes:
        stress_amplitude_weight:  Weight for stress amplitude / fatigue limit.
        max_stress_weight:        Weight for max von Mises / yield strength.
        hotspot_stress_weight:    Weight for hotspot stress / fatigue limit.
        mean_stress_penalty_weight: Weight for mean-stress correction penalty.
    """

    stress_amplitude_weight: float = 0.45
    max_stress_weight: float = 0.25
    hotspot_stress_weight: float = 0.20
    mean_stress_penalty_weight: float = 0.10

    @property
    def total(self) -> float:
        return (
            self.stress_amplitude_weight
            + self.max_stress_weight
            + self.hotspot_stress_weight
            + self.mean_stress_penalty_weight
        )

    def normalized(self) -> "FatigueProxyWeights":
        """Return a copy with weights scaled to sum to exactly 1.0."""
        t = self.total
        if t <= 0.0:
            raise FatigueModelError(
                "FatigueProxyWeights total is zero or negative.  "
                "All weights must be non-negative and at least one must be positive."
            )
        return FatigueProxyWeights(
            stress_amplitude_weight=self.stress_amplitude_weight / t,
            max_stress_weight=self.max_stress_weight / t,
            hotspot_stress_weight=self.hotspot_stress_weight / t,
            mean_stress_penalty_weight=self.mean_stress_penalty_weight / t,
        )


@dataclass
class FatigueProxyResult:
    """
    Structured output of the fatigue-risk proxy computation.

    PROXY DISCLAIMER: All values are proxy-based, not validated fatigue life.

    Attributes:
        success:                       True if the proxy score was computed.
        fatigue_risk_score:            Weighted composite proxy score.
                                       May exceed 1.0 for high-risk cases.
                                       Lower = lower relative risk.
        normalized_stress_amplitude_ratio: R_amp = amplitude / fatigue_limit.
        normalized_max_stress_ratio:       R_max = max_stress / yield_strength.
        normalized_hotspot_ratio:          R_hot = hotspot / fatigue_limit.
        mean_stress_penalty:               P_mean (dimensionless).
        governing_metric:              Name of the highest-contributing component.
        used_fatigue_limit_mpa:        Fatigue limit used (may be a fallback).
        used_yield_strength_mpa:       Yield strength used.
        proxy_mode:                    Always ``"fatigue_risk_proxy"``.
        warnings:                      List of fallback / low-confidence notes.
        metadata:                      Intermediate values for traceability.
        error_message:                 Error description if success is False.
    """

    success: bool
    fatigue_risk_score: float | None = None
    normalized_stress_amplitude_ratio: float | None = None
    normalized_max_stress_ratio: float | None = None
    normalized_hotspot_ratio: float | None = None
    mean_stress_penalty: float | None = None
    governing_metric: str | None = None
    used_fatigue_limit_mpa: float | None = None
    used_yield_strength_mpa: float | None = None
    proxy_mode: str = "fatigue_risk_proxy"  # Always this value; not a real life model
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success":                             self.success,
            "proxy_mode":                          self.proxy_mode,
            "fatigue_risk_score":                  self.fatigue_risk_score,
            "normalized_stress_amplitude_ratio":   self.normalized_stress_amplitude_ratio,
            "normalized_max_stress_ratio":         self.normalized_max_stress_ratio,
            "normalized_hotspot_ratio":            self.normalized_hotspot_ratio,
            "mean_stress_penalty":                 self.mean_stress_penalty,
            "governing_metric":                    self.governing_metric,
            "used_fatigue_limit_mpa":              self.used_fatigue_limit_mpa,
            "used_yield_strength_mpa":             self.used_yield_strength_mpa,
            "warnings":                            self.warnings,
            "metadata":                            self.metadata,
            "error_message":                       self.error_message,
        }


# ---------------------------------------------------------------------------
# Basic validation helpers
# ---------------------------------------------------------------------------

def _require_nonnegative(name: str, value: float | None) -> None:
    """
    Assert that ``value`` is non-negative if provided.

    Raises:
        FatigueModelError: if ``value`` is negative.
    """
    if value is not None and value < 0.0:
        raise FatigueModelError(
            f"Stress input '{name}' must be non-negative, got {value} MPa.  "
            f"Check that absolute stress values are provided."
        )


def _safe_positive_ratio(
    numerator: float | None,
    denominator: float | None,
) -> float | None:
    """
    Compute ``numerator / denominator`` safely.

    Returns:
        Ratio as a float, or ``None`` if either value is None or the
        denominator is zero/negative.
    """
    if numerator is None:
        return None
    if denominator is None:
        return None
    if denominator <= 0.0:
        return None
    return numerator / denominator


def _clamp_nonnegative(value: float) -> float:
    """Return max(0.0, value) — ensures a ratio is never negative."""
    return max(0.0, value)


# ---------------------------------------------------------------------------
# Material property helpers
# ---------------------------------------------------------------------------

def get_material_fatigue_limit(material: MaterialRecord) -> float | None:
    """
    Return the material's fatigue limit [MPa], or ``None`` if not set.

    Args:
        material: ``MaterialRecord`` from ``simulation/materials.py``.

    Returns:
        Fatigue limit in MPa, or ``None``.
    """
    return material.fatigue.fatigue_limit_mpa


def get_material_yield_strength(material: MaterialRecord) -> float | None:
    """
    Return the material's yield strength [MPa].

    Args:
        material: ``MaterialRecord``.

    Returns:
        Yield strength in MPa (should always be positive for valid materials).
    """
    return material.mechanical.yield_strength_mpa


# ---------------------------------------------------------------------------
# Effective fatigue limit resolver
# ---------------------------------------------------------------------------

def _resolve_fatigue_limit(
    material: MaterialRecord,
    warnings: list[str],
) -> tuple[float | None, bool]:
    """
    Resolve the effective fatigue limit for proxy calculation.

    If the material's ``fatigue_limit_mpa`` is set and not a placeholder,
    it is used directly.  If it is a placeholder or None, the fallback is
    ``yield_strength × _PLACEHOLDER_FATIGUE_FRACTION``.

    Args:
        material: MaterialRecord.
        warnings: Mutable list; fallback notes are appended here.

    Returns:
        Tuple (effective_fatigue_limit_mpa, is_fallback).
    """
    fatigue_limit = material.fatigue.fatigue_limit_mpa
    is_placeholder = material.fatigue.fatigue_limit_is_placeholder

    if fatigue_limit is not None and not is_placeholder:
        return fatigue_limit, False

    # Fallback path
    yield_strength = material.mechanical.yield_strength_mpa
    fallback_limit = yield_strength * _PLACEHOLDER_FATIGUE_FRACTION

    if fatigue_limit is not None and is_placeholder:
        # There is a placeholder numeric value — use it but warn
        warnings.append(
            f"Material '{material.name}': fatigue_limit_mpa "
            f"({fatigue_limit:.1f} MPa) is marked as a placeholder.  "
            f"Proxy score is LOW CONFIDENCE.  "
            f"Replace with validated fatigue data before interpreting results."
        )
        return fatigue_limit, True

    # fatigue_limit is None entirely — use yield fraction fallback
    warnings.append(
        f"Material '{material.name}' has no fatigue_limit_mpa.  "
        f"Using fallback: yield_strength × {_PLACEHOLDER_FATIGUE_FRACTION:.2f} = "
        f"{fallback_limit:.1f} MPa.  "
        f"Proxy score is LOW CONFIDENCE.  "
        f"Set fatigue_limit_mpa in materials.yaml when data is available."
    )
    return fallback_limit, True


# ---------------------------------------------------------------------------
# Mean-stress penalty
# ---------------------------------------------------------------------------

def compute_mean_stress_penalty(
    mean_stress_mpa: float | None,
    fatigue_limit_mpa: float | None,
    method: str | None = None,
) -> float:
    """
    Compute a simple mean-stress penalty for the fatigue proxy.

    PROXY SIMPLIFICATION:
        A simplified Goodman-inspired ratio is used when both mean stress
        and fatigue limit are available.  The penalty is NOT the Goodman
        correction factor — it is a proxy penalty bounded to [0, 1).
        ``method = "goodman_placeholder"`` is documented but uses the same
        simplified formula.

        Formula::

            P_mean = clamp(|mean_stress| / (2 × fatigue_limit), 0, 0.99)

        Rationale: as mean stress approaches fatigue_limit, the penalty
        approaches 0.5, which is a reasonable upper bound for a simple
        proxy.  The clamp prevents absurd values.

    Args:
        mean_stress_mpa:   Mean stress [MPa] (unsigned; compressive offset
                           is also penalising for this proxy).
        fatigue_limit_mpa: Effective fatigue limit [MPa].
        method:            Name of the correction method from the material
                           config (e.g. ``"goodman_placeholder"``).  Not
                           used to select different formula in version 1.

    Returns:
        Dimensionless penalty in [0.0, 0.99].
    """
    if mean_stress_mpa is None:
        return 0.0
    if fatigue_limit_mpa is None:
        return 0.0
    if fatigue_limit_mpa <= 0.0:
        return 0.0

    # Use absolute value: compressive mean stress also reduces fatigue life
    penalty = abs(mean_stress_mpa) / (2.0 * fatigue_limit_mpa)
    return min(_clamp_nonnegative(penalty), 0.99)


# ---------------------------------------------------------------------------
# Stress amplitude inference
# ---------------------------------------------------------------------------

def infer_stress_amplitude(
    inputs: FatigueProxyInputs,
    loadcase: LoadCaseRecord | None = None,
) -> tuple[float | None, bool]:
    """
    Return the effective stress amplitude for the proxy and whether it
    was inferred (rather than explicitly provided).

    Preference order:
      1. ``inputs.stress_amplitude_mpa`` if explicitly set.
      2. For CYCLIC load cases: amplitude force / mean force × max stress
         (simple proportional scaling, if the load-case amplitudes are known).
      3. Conservative fallback: ``_AMPLITUDE_FROM_MAX_STRESS_FRACTION × max_stress``.

    TRANSPARENCY NOTE:
        Callers that want to know if fallback was used should check the
        second return value (``inferred: bool``).

    Args:
        inputs:   Stress proxy inputs.
        loadcase: Optional load-case record for cyclic amplitude scaling.

    Returns:
        Tuple (stress_amplitude_mpa, inferred) where ``inferred`` is True
        if the amplitude was derived rather than directly provided.
    """
    # Explicit value — use as-is
    if inputs.stress_amplitude_mpa is not None:
        return inputs.stress_amplitude_mpa, False

    # Cyclic load case: proportional scaling from load amplitudes
    if loadcase is not None:
        mean_f = loadcase.mean_force_n
        amp_f = loadcase.amplitude_force_n
        max_stress = inputs.max_von_mises_stress_mpa
        if (
            mean_f is not None and mean_f > 0.0
            and amp_f is not None and amp_f > 0.0
            and max_stress is not None and max_stress > 0.0
        ):
            # Stress amplitude ≈ (amplitude_force / mean_force) × max_stress
            # This is a very rough proportionality; suitable for proxy only.
            ratio = amp_f / (mean_f + amp_f)  # amplitude fraction of peak
            return ratio * max_stress, True

    # Conservative fallback: 50% of max stress (fully-reversed assumption)
    fallback_stress = inputs.max_von_mises_stress_mpa
    if fallback_stress is not None:
        return (
            _AMPLITUDE_FROM_MAX_STRESS_FRACTION * fallback_stress,
            True,
        )

    return None, False


# ---------------------------------------------------------------------------
# Main fatigue-risk computation
# ---------------------------------------------------------------------------

def compute_fatigue_risk_proxy(
    inputs: FatigueProxyInputs,
    material: MaterialRecord,
    loadcase: LoadCaseRecord | None = None,
    weights: FatigueProxyWeights | None = None,
) -> FatigueProxyResult:
    """
    Compute the fatigue-risk proxy score for one pipeline case.

    PROXY DISCLAIMER: output is NOT validated fatigue life prediction.

    Score components::

        R_amp  = stress_amplitude / fatigue_limit
        R_max  = max_von_mises  / yield_strength
        R_hot  = hotspot_stress / fatigue_limit   (or max stress if no hotspot)
        P_mean = mean_stress_penalty (bounded [0, 0.99])

        score = w_amp·R_amp + w_max·R_max + w_hot·R_hot + w_mean·P_mean

    Lower scores indicate lower relative fatigue risk.  Scores > 1.0
    indicate at least one stress component exceeds its threshold.

    Args:
        inputs:    Stress proxy inputs (from FE results or estimates).
        material:  MaterialRecord with mechanical and fatigue properties.
        loadcase:  Optional load-case record for amplitude inference.
        weights:   Weighting factors.  Uses ``FatigueProxyWeights()``
                   defaults if None.

    Returns:
        ``FatigueProxyResult`` — always returned, never raises for soft
        failures (missing inputs, fallback assumptions).
    """
    warnings: list[str] = []
    metadata: dict[str, Any] = {"inputs": inputs.to_dict()}
    result = FatigueProxyResult(success=False, proxy_mode="fatigue_risk_proxy")

    # --- Validate non-negative stress inputs ---
    try:
        for name, value in (
            ("max_von_mises_stress_mpa", inputs.max_von_mises_stress_mpa),
            ("hotspot_stress_mpa",       inputs.hotspot_stress_mpa),
            ("stress_amplitude_mpa",     inputs.stress_amplitude_mpa),
            ("mean_stress_mpa",          inputs.mean_stress_mpa),
        ):
            _require_nonnegative(name, value)
    except FatigueModelError as exc:
        result.error_message = str(exc)
        return result

    # Check that at least one meaningful stress input is available
    if all(v is None for v in (
        inputs.max_von_mises_stress_mpa,
        inputs.hotspot_stress_mpa,
        inputs.stress_amplitude_mpa,
    )):
        result.error_message = (
            "No stress inputs provided.  At least one of max_von_mises_stress_mpa, "
            "hotspot_stress_mpa, or stress_amplitude_mpa must be set to compute "
            "a fatigue-risk proxy score."
        )
        return result

    # --- Resolve material thresholds ---
    fatigue_limit, fl_is_fallback = _resolve_fatigue_limit(material, warnings)
    yield_strength = material.mechanical.yield_strength_mpa
    result.used_fatigue_limit_mpa = fatigue_limit
    result.used_yield_strength_mpa = yield_strength

    # --- Normalised weights ---
    w = (weights or FatigueProxyWeights()).normalized()

    # --- Component 1: Stress amplitude ratio ---
    amplitude, amplitude_inferred = infer_stress_amplitude(inputs, loadcase)
    if amplitude_inferred and amplitude is not None:
        warnings.append(
            f"stress_amplitude_mpa was not explicitly provided.  "
            f"Inferred as {amplitude:.3f} MPa using "
            f"{'cyclic load proportion' if loadcase is not None else 'conservative max-stress fraction'}.  "
            f"Provide measured stress amplitude for higher confidence."
        )

    r_amp = _safe_positive_ratio(amplitude, fatigue_limit)
    if r_amp is None:
        warnings.append(
            "Stress amplitude ratio (R_amp) could not be computed.  "
            "Component contribution is zero."
        )
        r_amp = 0.0

    result.normalized_stress_amplitude_ratio = r_amp
    metadata["stress_amplitude_used_mpa"] = amplitude
    metadata["amplitude_inferred"] = amplitude_inferred

    # --- Component 2: Max stress / yield ratio ---
    r_max = _safe_positive_ratio(inputs.max_von_mises_stress_mpa, yield_strength)
    if r_max is None:
        warnings.append(
            "Max stress ratio (R_max) could not be computed.  "
            "Component contribution is zero."
        )
        r_max = 0.0

    result.normalized_max_stress_ratio = r_max

    # --- Component 3: Hotspot stress ratio ---
    # Use hotspot_stress if available; fall back to max stress
    hotspot = inputs.hotspot_stress_mpa
    hotspot_fallback = False
    if hotspot is None and inputs.max_von_mises_stress_mpa is not None:
        hotspot = inputs.max_von_mises_stress_mpa
        hotspot_fallback = True
        warnings.append(
            "hotspot_stress_mpa not provided.  "
            "Using max_von_mises_stress_mpa as hotspot proxy.  "
            "Provide actual hotspot stress from postprocessor for higher fidelity."
        )

    r_hot = _safe_positive_ratio(hotspot, fatigue_limit)
    if r_hot is None:
        r_hot = 0.0

    result.normalized_hotspot_ratio = r_hot
    metadata["hotspot_fallback_used"] = hotspot_fallback

    # --- Component 4: Mean stress penalty ---
    mean_correction_method = material.fatigue.mean_stress_correction
    p_mean = compute_mean_stress_penalty(
        inputs.mean_stress_mpa,
        fatigue_limit,
        method=mean_correction_method,
    )

    if inputs.mean_stress_mpa is None:
        warnings.append(
            "mean_stress_mpa not provided.  Mean-stress penalty = 0.0."
        )
    elif mean_correction_method and "placeholder" in mean_correction_method.lower():
        warnings.append(
            f"Mean-stress correction method '{mean_correction_method}' is a "
            f"placeholder.  A simplified absolute-ratio penalty was applied.  "
            f"Replace with validated Goodman or Morrow correction when available."
        )

    result.mean_stress_penalty = p_mean

    # --- Weighted composite score ---
    score = (
        w.stress_amplitude_weight   * _clamp_nonnegative(r_amp)
        + w.max_stress_weight       * _clamp_nonnegative(r_max)
        + w.hotspot_stress_weight   * _clamp_nonnegative(r_hot)
        + w.mean_stress_penalty_weight * p_mean
    )

    result.fatigue_risk_score = float(f"{score:.6f}")

    # --- Governing metric ---
    contributions = {
        "stress_amplitude":  w.stress_amplitude_weight   * _clamp_nonnegative(r_amp),
        "max_stress":        w.max_stress_weight         * _clamp_nonnegative(r_max),
        "hotspot_stress":    w.hotspot_stress_weight     * _clamp_nonnegative(r_hot),
        "mean_stress":       w.mean_stress_penalty_weight * p_mean,
    }
    result.governing_metric = max(contributions, key=contributions.get)  # type: ignore[arg-type]

    metadata["weight_contributions"] = {k: float(f"{v:.6f}") for k, v in contributions.items()}
    metadata["weights_used"] = {
        "stress_amplitude": w.stress_amplitude_weight,
        "max_stress":       w.max_stress_weight,
        "hotspot_stress":   w.hotspot_stress_weight,
        "mean_stress":      w.mean_stress_penalty_weight,
    }
    metadata["fatigue_limit_is_fallback"] = fl_is_fallback
    metadata["risk_category"] = categorize_fatigue_risk(score)

    # Standard disclaimer always appended to metadata
    metadata["proxy_disclaimer"] = (
        "FATIGUE RISK PROXY ONLY — not validated fatigue life prediction. "
        "For comparative ranking purposes only."
    )

    result.warnings = warnings
    result.metadata = metadata
    result.success = True

    return result


# ---------------------------------------------------------------------------
# Load-case integration convenience constructor
# ---------------------------------------------------------------------------

def fatigue_inputs_from_scalar_results(
    max_von_mises_stress_mpa: float | None = None,
    hotspot_stress_mpa: float | None = None,
    stress_amplitude_mpa: float | None = None,
    mean_stress_mpa: float | None = None,
    loadcase: LoadCaseRecord | None = None,
) -> FatigueProxyInputs:
    """
    Convenience constructor for ``FatigueProxyInputs`` from scalar FE results.

    Automatically populates ``cycle_count`` and ``frequency_hz`` from the
    load case record if provided.

    Args:
        max_von_mises_stress_mpa: Peak von Mises stress [MPa].
        hotspot_stress_mpa:       Hotspot / local peak stress [MPa].
        stress_amplitude_mpa:     Half cyclic stress range [MPa].
        mean_stress_mpa:          Mean (static offset) stress [MPa].
        loadcase:                 Optional load-case record.

    Returns:
        ``FatigueProxyInputs`` ready for ``compute_fatigue_risk_proxy``.
    """
    cycle_count: int | None = None
    frequency_hz: float | None = None

    if loadcase is not None:
        cycle_count = loadcase.cycle_count
        frequency_hz = loadcase.frequency_hz

    return FatigueProxyInputs(
        max_von_mises_stress_mpa=max_von_mises_stress_mpa,
        hotspot_stress_mpa=hotspot_stress_mpa,
        stress_amplitude_mpa=stress_amplitude_mpa,
        mean_stress_mpa=mean_stress_mpa,
        cycle_count=cycle_count,
        frequency_hz=frequency_hz,
    )


# ---------------------------------------------------------------------------
# Risk category helper
# ---------------------------------------------------------------------------

def categorize_fatigue_risk(score: float | None) -> str:
    """
    Map a fatigue proxy score to a descriptive risk category.

    ARCHITECTURAL DECISION — heuristic thresholds, clearly labelled:
        These thresholds are empirical starting points for version-1
        comparative screening.  They have no regulatory or clinical basis.
        Adjust them in a future version once baseline sweep results are
        available to calibrate against observed design behaviour.

    Thresholds (score < threshold → category):
        < 0.30 → low
        < 0.60 → moderate
        < 0.90 → high
        ≥ 0.90 → severe

    Args:
        score: Fatigue proxy score (may be > 1.0 for high-risk cases).

    Returns:
        Category string: ``"unknown"``, ``"low"``, ``"moderate"``,
        ``"high"``, or ``"severe"``.
    """
    if score is None:
        return "unknown"
    for category, threshold in _RISK_THRESHOLDS.items():
        if score < threshold:
            return category
    return "severe"


# ---------------------------------------------------------------------------
# Hard-fail wrapper
# ---------------------------------------------------------------------------

def require_fatigue_risk_proxy(
    inputs: FatigueProxyInputs,
    material: MaterialRecord,
    loadcase: LoadCaseRecord | None = None,
    weights: FatigueProxyWeights | None = None,
) -> FatigueProxyResult:
    """
    Compute the fatigue proxy and raise ``FatigueModelError`` if unsuccessful.

    Use this when the calling code cannot continue without a valid proxy
    score (e.g. if the ranking module requires all cases to have a score).

    Args:
        inputs:   Fatigue proxy inputs.
        material: Material record.
        loadcase: Optional load-case record.
        weights:  Proxy weights.

    Returns:
        ``FatigueProxyResult`` with ``success=True``.

    Raises:
        FatigueModelError: if the proxy computation fails.
    """
    result = compute_fatigue_risk_proxy(inputs, material, loadcase, weights)
    if not result.success:
        raise FatigueModelError(
            f"Fatigue proxy computation failed: {result.error_message}"
        )
    return result
