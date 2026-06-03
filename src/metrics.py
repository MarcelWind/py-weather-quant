"""Evaluation metrics for probabilistic temperature forecasts.

All functions operate on arrays/Series and return scalar values or
DataFrame-friendly structures for downstream plotting.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from .data import SIGMA_MIN_SQ

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CRPS — Continuous Ranked Probability Score (Gaussian closed form)
# ═══════════════════════════════════════════════════════════════════════════════


def crps_gaussian(
    mu: np.ndarray, sigma: np.ndarray, y: np.ndarray,
) -> np.ndarray:
    """CRPS for N(μ, σ²) at observation y (Gneiting & Raftery 2007).

    Closed form: σ[ z(2Φ(z)−1) + 2φ(z) − 1/√π ], z = (y−μ)/σ
    """
    sigma = np.maximum(sigma, math.sqrt(SIGMA_MIN_SQ))
    z = (y - mu) / sigma
    return sigma * (
        z * (2.0 * norm.cdf(z) - 1.0)
        + 2.0 * norm.pdf(z)
        - 1.0 / math.sqrt(math.pi)
    )


def mean_crps(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> float:
    """Mean CRPS over valid (finite) triples."""
    mask = np.isfinite(mu) & np.isfinite(sigma) & np.isfinite(y)
    if not mask.any():
        return float("nan")
    return float(np.mean(crps_gaussian(mu[mask], sigma[mask], y[mask])))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Brier Score — binary probability forecast skill
# ═══════════════════════════════════════════════════════════════════════════════


def brier_score(pred: np.ndarray, outcome: np.ndarray) -> float:
    """Mean Brier score: mean((p − o)²)."""
    mask = np.isfinite(pred) & np.isfinite(outcome)
    if not mask.any():
        return float("nan")
    return float(np.mean((pred[mask] - outcome[mask]) ** 2))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Log-Loss — logarithmic scoring rule
# ═══════════════════════════════════════════════════════════════════════════════


def log_loss(pred: np.ndarray, outcome: np.ndarray, eps: float = 1e-15) -> float:
    """Mean log-loss: −mean( o⋅log(p) + (1−o)⋅log(1−p) )."""
    pred = np.clip(pred, eps, 1.0 - eps)
    mask = np.isfinite(pred) & np.isfinite(outcome)
    if not mask.any():
        return float("nan")
    p, o = pred[mask], outcome[mask]
    return float(-np.mean(o * np.log(p) + (1.0 - o) * np.log(1.0 - p)))


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Calibration Curve — predicted probability vs observed frequency
# ═══════════════════════════════════════════════════════════════════════════════


def calibration_curve(
    pred: np.ndarray, outcome: np.ndarray, n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin predicted probabilities and compute observed frequency per bin.

    Returns
    -------
    bin_centers : ndarray  — center of each bin
    observed_freq : ndarray — fraction of positive outcomes in each bin
    counts : ndarray       — number of samples in each bin
    """
    mask = np.isfinite(pred) & np.isfinite(outcome)
    pred, outcome = pred[mask], outcome[mask]
    if len(pred) == 0:
        return np.array([]), np.array([]), np.array([])

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(pred, bins) - 1, 0, n_bins - 1)

    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    observed_freq = np.array([
        float(outcome[bin_indices == i].mean()) if (bin_indices == i).any() else float("nan")
        for i in range(n_bins)
    ])
    counts = np.array([int((bin_indices == i).sum()) for i in range(n_bins)])
    return bin_centers, observed_freq, counts


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Reliability Diagram Data — decomposition of Brier score
# ═══════════════════════════════════════════════════════════════════════════════


def reliability_diagram_data(
    pred: np.ndarray, outcome: np.ndarray, n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reliability diagram components.

    Returns
    -------
    bin_centers : ndarray
    observed_freq : ndarray
    gaps : ndarray          — observed_freq − bin_center (reliability gap)
    counts : ndarray
    """
    bin_centers, observed_freq, counts = calibration_curve(pred, outcome, n_bins)
    gaps = observed_freq - bin_centers
    return bin_centers, observed_freq, gaps, counts


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Sharpness & Entropy — concentration of predictive distributions
# ═══════════════════════════════════════════════════════════════════════════════


def sharpness(sigma: np.ndarray) -> float:
    """Sharpness = mean variance of predictive distribution.

    Lower is sharper (more concentrated). For Gaussian predictions,
    sharpness = mean(σ²).
    """
    mask = np.isfinite(sigma)
    if not mask.any():
        return float("nan")
    return float(np.mean(sigma[mask] ** 2))


def prediction_entropy(sigma: np.ndarray) -> float:
    """Mean differential entropy of Gaussian predictions.

    Entropy = 0.5 ⋅ log(2πeσ²).
    Lower entropy = sharper predictions.
    """
    mask = np.isfinite(sigma) & (sigma > 0)
    if not mask.any():
        return float("nan")
    return float(np.mean(0.5 * np.log(2.0 * math.pi * math.e * sigma[mask] ** 2)))


# ═══════════════════════════════════════════════════════════════════════════════
# Bracket-level utilities (convert continuous predictions to binary outcomes)
# ═══════════════════════════════════════════════════════════════════════════════


def bracket_probability(
    mu: np.ndarray, sigma: np.ndarray, low: float, high: float,
) -> np.ndarray:
    """P(low ≤ T < high) under N(μ, σ²) predictive distribution."""
    sigma = np.maximum(sigma, math.sqrt(SIGMA_MIN_SQ))
    cdf_high = norm.cdf((high - mu) / sigma) if math.isfinite(high) else 1.0
    cdf_low = norm.cdf((low - mu) / sigma) if math.isfinite(low) else 0.0
    return cdf_high - cdf_low


def bracket_outcome(
    T_obs: np.ndarray, low: float, high: float,
) -> np.ndarray:
    """Binary: is T_obs in [low, high)?"""
    in_bracket = np.ones(len(T_obs), dtype=bool)
    if math.isfinite(low):
        in_bracket &= (T_obs >= low)
    if math.isfinite(high):
        in_bracket &= (T_obs < high)
    return in_bracket.astype(float)


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregate evaluation for a single model cell
# ═══════════════════════════════════════════════════════════════════════════════

# Default Celsius brackets for daily-max temperature
DEFAULT_BRACKETS = [
    ("≤0°C", float("-inf"), 1.0),
    ("1-4°C", 1.0, 5.0),
    ("5-9°C", 5.0, 10.0),
    ("10-14°C", 10.0, 15.0),
    ("15-19°C", 15.0, 20.0),
    ("20-24°C", 20.0, 25.0),
    ("25-29°C", 25.0, 30.0),
    ("30-34°C", 30.0, 35.0),
    ("≥35°C", 35.0, float("inf")),
]

# Default Fahrenheit brackets
DEFAULT_BRACKETS_F = [
    ("≤32°F", float("-inf"), 33.0),
    ("33-40°F", 33.0, 41.0),
    ("41-50°F", 41.0, 51.0),
    ("51-60°F", 51.0, 61.0),
    ("61-70°F", 61.0, 71.0),
    ("71-80°F", 71.0, 81.0),
    ("81-90°F", 81.0, 91.0),
    ("91-100°F", 91.0, 101.0),
    (">100°F", 101.0, float("inf")),
]


def _detect_brackets(T_obs: np.ndarray) -> list[tuple[str, float, float]]:
    """Detect whether data is Fahrenheit or Celsius and return brackets."""
    if np.nanmax(T_obs) > 50:
        return DEFAULT_BRACKETS_F
    return DEFAULT_BRACKETS


def evaluate_model(
    mu: np.ndarray,
    sigma: np.ndarray,
    y: np.ndarray,
    model_name: str = "",
) -> dict:
    """Compute all evaluation metrics for a model's predictions.

    Parameters
    ----------
    mu : ndarray  — predicted means
    sigma : ndarray — predicted standard deviations
    y : ndarray  — observed values
    model_name : str — label for the results dict

    Returns
    -------
    dict with keys: crps, brier, log_loss, sharpness, entropy,
                    calibration_x, calibration_y, reliability_gap, n
    """
    mask = np.isfinite(mu) & np.isfinite(sigma) & np.isfinite(y)
    mu, sigma, y = mu[mask], sigma[mask], y[mask]
    n = len(mu)

    # CRPS (continuous)
    crps_val = mean_crps(mu, sigma, y)

    # Bracket-level metrics (Brier, log-loss, calibration)
    brackets = _detect_brackets(y)
    all_p_yes: list[float] = []
    all_outcomes: list[float] = []

    for _, low, high in brackets:
        p_yes = bracket_probability(mu, sigma, low, high)
        outcome = bracket_outcome(y, low, high)
        all_p_yes.extend(p_yes.tolist())
        all_outcomes.extend(outcome.tolist())

    p_yes_arr = np.array(all_p_yes)
    outcome_arr = np.array(all_outcomes)

    brier_val = brier_score(p_yes_arr, outcome_arr)
    ll_val = log_loss(p_yes_arr, outcome_arr)

    # Calibration
    bin_c, obs_f, _ = calibration_curve(p_yes_arr, outcome_arr)

    # Reliability gap (mean absolute gap)
    gap = np.nanmean(np.abs(obs_f - bin_c)) if len(bin_c) > 0 else float("nan")

    # Sharpness & entropy
    sharp_val = sharpness(sigma)
    ent_val = prediction_entropy(sigma)

    return {
        "model": model_name,
        "n": n,
        "crps": crps_val,
        "brier": brier_val,
        "log_loss": ll_val,
        "sharpness": sharp_val,
        "entropy": ent_val,
        "reliability_gap": gap,
        "calibration_x": bin_c.tolist(),
        "calibration_y": obs_f.tolist(),
    }


def bracket_calibration(
    mu: np.ndarray,
    sigma: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calibration curve from Gaussian forecasts via temperature brackets.

    Converts continuous Gaussian predictions (μ, σ) into bracket-level
    probability forecasts, then bins these against binary outcomes and returns
    the calibration curve components.  Suitable for aggregate calibration
    plotting across multiple data cells.

    Parameters
    ----------
    mu : ndarray  — predicted means
    sigma : ndarray — predicted standard deviations
    y : ndarray  — observed values
    n_bins : int — number of probability bins (default 10)

    Returns
    -------
    bin_centers : ndarray  — centre of each probability bin
    observed_freq : ndarray — fraction of positive outcomes per bin
    counts : ndarray       — number of samples per bin
    """
    mask = np.isfinite(mu) & np.isfinite(sigma) & np.isfinite(y)
    mu, sigma, y = mu[mask], sigma[mask], y[mask]

    brackets = _detect_brackets(y)
    all_p_yes: list[float] = []
    all_outcomes: list[float] = []

    for _, low, high in brackets:
        p_yes = bracket_probability(mu, sigma, low, high)
        outcome = bracket_outcome(y, low, high)
        all_p_yes.extend(p_yes.tolist())
        all_outcomes.extend(outcome.tolist())

    return calibration_curve(
        np.array(all_p_yes), np.array(all_outcomes), n_bins,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. PIT Histogram — probability integral transform for Gaussian forecasts
# ═══════════════════════════════════════════════════════════════════════════════


def pit_histogram(
    mu: np.ndarray,
    sigma: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """PIT histogram for Gaussian predictive distributions.

    The Probability Integral Transform PIT = Φ((y − μ) / σ) should be
    uniform under perfect calibration.  Returns bin edges *frequencies*
    (not densities) for direct plotting.

    Parameters
    ----------
    mu : ndarray  — predicted means
    sigma : ndarray — predicted standard deviations
    y : ndarray  — observed values
    n_bins : int — number of histogram bins (default 10)

    Returns
    -------
    bin_edges : ndarray  — (n_bins+1,) edges of the histogram bins
    hist : ndarray       — (n_bins,) frequency in each bin
    """
    mask = np.isfinite(mu) & np.isfinite(sigma) & np.isfinite(y)
    mu, sigma, y = mu[mask], sigma[mask], y[mask]

    sigma = np.maximum(sigma, math.sqrt(SIGMA_MIN_SQ))
    pit = norm.cdf((y - mu) / sigma)

    hist, bin_edges = np.histogram(pit, bins=n_bins, range=(0.0, 1.0))
    return bin_edges, hist
