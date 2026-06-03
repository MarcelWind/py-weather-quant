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
# Bracket calibration — reliability of bracket-level probabilities
# ═══════════════════════════════════════════════════════════════════════════════


def bracket_calibration(
    mu: np.ndarray,
    sigma: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
    n_thresholds: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calibration curve for bracket (threshold exceedance) probabilities.

    Uses n_thresholds equally-spaced percentiles of the observed y as binary
    events, computes P(T > threshold | μ, σ) for each, pools across all
    thresholds, then bins the predicted probabilities into n_bins and computes
    the observed frequency per bin.

    Returns
    -------
    bin_centers : ndarray
    observed_freq : ndarray
    counts : ndarray
    """
    mask = np.isfinite(mu) & np.isfinite(sigma) & np.isfinite(y)
    mu_v = mu[mask]
    sigma_v = sigma[mask]
    y_v = y[mask]
    n = len(mu_v)
    if n < 3:
        return np.array([]), np.array([]), np.array([])

    thresholds = np.percentile(y_v, np.linspace(5, 95, n_thresholds))
    all_pred: list[float] = []
    all_outcome: list[float] = []

    for thresh in thresholds:
        p_above = 1.0 - norm.cdf(
            (thresh - mu_v) / np.maximum(sigma_v, math.sqrt(SIGMA_MIN_SQ))
        )
        outcome = (y_v >= thresh).astype(float)
        all_pred.extend(p_above.tolist())
        all_outcome.extend(outcome.tolist())

    return calibration_curve(
        np.array(all_pred), np.array(all_outcome), n_bins=n_bins
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 7. evaluate_model — unified cell-level metric computation
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_model(
    mu: np.ndarray,
    sigma: np.ndarray,
    y: np.ndarray,
    model_name: str = "",
) -> dict:
    """Compute all evaluation metrics for a single (model, cell) result.

    Parameters
    ----------
    mu : ndarray — predicted means
    sigma : ndarray — predicted standard deviations
    y : ndarray — observed temperatures
    model_name : str — model identifier for the output dict

    Returns
    -------
    dict with keys:
        model, n, crps, brier, log_loss, sharpness, entropy,
        reliability_gap, calibration_x, calibration_y
    """
    mask = np.isfinite(mu) & np.isfinite(sigma) & np.isfinite(y)
    mu_v = mu[mask]
    sigma_v = sigma[mask]
    y_v = y[mask]
    n = len(mu_v)

    if n < 3:
        return {
            "model": model_name, "n": n,
            "crps": float("nan"), "brier": float("nan"),
            "log_loss": float("nan"), "sharpness": float("nan"),
            "entropy": float("nan"), "reliability_gap": float("nan"),
            "calibration_x": [], "calibration_y": [],
        }

    # CRPS
    crps_val = mean_crps(mu_v, sigma_v, y_v)

    # Sharpness & entropy
    sharp_val = sharpness(sigma_v)
    entropy_val = prediction_entropy(sigma_v)

    # For Brier / log-loss / calibration we need a binary event.
    # Use "temperature above the observed median" as the event threshold.
    threshold = float(np.median(y_v))
    p_above = 1.0 - norm.cdf((threshold - mu_v) / np.maximum(sigma_v, math.sqrt(SIGMA_MIN_SQ)))
    outcome_above = (y_v >= threshold).astype(float)

    brier_val = brier_score(p_above, outcome_above)
    ll_val = log_loss(p_above, outcome_above)

    # Reliability gap: mean absolute difference between predicted probability
    # and observed frequency (from calibration curve)
    _, obs_freq, _ = calibration_curve(p_above, outcome_above, n_bins=10)
    bin_edges = np.linspace(0.0, 1.0, 11)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    valid_bins = np.isfinite(obs_freq)
    if valid_bins.any():
        reliability_gap = float(np.mean(np.abs(obs_freq[valid_bins] - bin_centers[valid_bins])))
    else:
        reliability_gap = float("nan")

    # Calibration data (for plotting calibration curves / reliability diagrams)
    calib_x, calib_y, _ = calibration_curve(p_above, outcome_above, n_bins=10)
    calibration_x = calib_x.tolist() if len(calib_x) > 0 else []
    calibration_y = calib_y.tolist() if len(calib_y) > 0 else []

    return {
        "model": model_name,
        "n": n,
        "crps": crps_val,
        "brier": brier_val,
        "log_loss": ll_val,
        "sharpness": sharp_val,
        "entropy": entropy_val,
        "reliability_gap": reliability_gap,
        "calibration_x": calibration_x,
        "calibration_y": calibration_y,
    }


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
# 8. Model-vs-Market Comparison — bracket-level metrics against Polymarket
# ═══════════════════════════════════════════════════════════════════════════════


def model_vs_market_brier(
    p_model: np.ndarray, p_market: np.ndarray, outcome: np.ndarray,
) -> float:
    """Mean Brier score for model predictions vs market-implied probabilities.

    Lower is better. A Brier score of 0 means perfect prediction; the benchmark
    is the market-implied Brier score (using p_market instead of p_model).
    """
    return brier_score(p_model, outcome)


def calibration_gap(
    p_model: np.ndarray, p_market: np.ndarray,
) -> float:
    """Mean absolute difference between model and market probabilities.

    Measures how much the model disagrees with the market consensus.
    Lower values mean the model is closer to market pricing.
    """
    mask = np.isfinite(p_model) & np.isfinite(p_market)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(p_model[mask] - p_market[mask])))


def edge_accuracy(
    p_model: np.ndarray, p_market: np.ndarray,
    outcome: np.ndarray, threshold: float = 0.05,
) -> dict:
    """Evaluate model vs market edge: does the model's high-conviction
    call (|p_model - 0.5| > threshold) outperform the market?

    Parameters
    ----------
    p_model : ndarray — model-implied bracket probabilities
    p_market : ndarray — market-implied bracket probabilities
    outcome : ndarray — binary outcomes
    threshold : float — minimum distance from 0.5 to count as "high conviction"

    Returns
    -------
    dict with keys:
        n_calls          — number of high-conviction calls
        model_brier      — Brier score on calls using p_model
        market_brier     — Brier score on calls using p_market
        edge             — market_brier - model_brier (positive = model adds value)
        n_correct_model  — count where model predicted >0.5 and outcome=1
        n_correct_market — count where market predicted >0.5 and outcome=1
    """
    mask = (
        np.isfinite(p_model) & np.isfinite(p_market)
        & np.isfinite(outcome) & (np.abs(p_model - 0.5) > threshold)
    )
    if not mask.any():
        return {
            "n_calls": 0, "model_brier": float("nan"), "market_brier": float("nan"),
            "edge": float("nan"), "n_correct_model": 0, "n_correct_market": 0,
        }

    pm = p_model[mask]
    pk = p_market[mask]
    oc = outcome[mask]

    model_brier_val = float(np.mean((pm - oc) ** 2))
    market_brier_val = float(np.mean((pk - oc) ** 2))

    return {
        "n_calls": int(mask.sum()),
        "model_brier": model_brier_val,
        "market_brier": market_brier_val,
        "edge": market_brier_val - model_brier_val,
        "n_correct_model": int(((pm > 0.5) & (oc == 1)).sum()),
        "n_correct_market": int(((pk > 0.5) & (oc == 1)).sum()),
    }


def compare_model_to_market(
    mu: np.ndarray,
    sigma: np.ndarray,
    T_obs: np.ndarray,
    polymarket_prices: pd.DataFrame,
    target_dates: np.ndarray,
    model_name: str = "",
) -> dict:
    """Compare model bracket probabilities to market prices and outcomes.

    For each (date, bracket) pair, computes:
      - Model probability P(T in bracket | μ, σ)
      - Market price P_market
      - Binary outcome: was T_obs in that bracket?

    Parameters
    ----------
    mu : ndarray — predicted means per row
    sigma : ndarray — predicted stds per row
    T_obs : ndarray — observed temperatures per row
    polymarket_prices : pd.DataFrame — from load_polymarket_prices()
    target_dates : ndarray — date strings/values matching each row
    model_name : str

    Returns
    -------
    dict with keys: model, n, brier_model, brier_market, calib_gap,
                    edge_stats, per_bracket DataFrame
    """
    # Build lookup: (date_str, low, high) → market_price
    price_map: dict[tuple[str, float, float], float] = {}
    for _, row in polymarket_prices.iterrows():
        d = row["target_date"]
        ds = str(d.date()) if hasattr(d, "date") else str(d)
        price_map[(ds, row["low"], row["high"])] = row["price"]

    rows: list[dict] = []
    for i in range(len(mu)):
        if not (np.isfinite(mu[i]) and np.isfinite(sigma[i]) and np.isfinite(T_obs[i])):
            continue
        ds = str(pd.Timestamp(target_dates[i]).date())
        for _, pr in polymarket_prices.iterrows():
            pr_ds = str(pr["target_date"].date()) if hasattr(pr["target_date"], "date") else str(pr["target_date"])
            if pr_ds != ds:
                continue
            low, high = pr["low"], pr["high"]
            p_model = bracket_probability(
                np.array([mu[i]]), np.array([sigma[i]]), low, high
            )[0]
            outcome = bracket_outcome(np.array([T_obs[i]]), low, high)[0]
            market_price = price_map.get((ds, low, high), float("nan"))
            rows.append({
                "target_date": ds,
                "bracket_label": pr["bracket_label"],
                "low": low,
                "high": high,
                "p_model": p_model,
                "p_market": market_price,
                "outcome": outcome,
            })

    if not rows:
        return {"model": model_name, "n": 0, "brier_model": float("nan"),
                "brier_market": float("nan"), "calib_gap": float("nan"),
                "edge": {}, "per_bracket": pd.DataFrame()}

    df = pd.DataFrame(rows)
    valid = df.dropna(subset=["p_model", "p_market", "outcome"])

    if len(valid) < 5:
        return {"model": model_name, "n": len(valid), "brier_model": float("nan"),
                "brier_market": float("nan"), "calib_gap": float("nan"),
                "edge": {}, "per_bracket": df}

    p_mod = valid["p_model"].to_numpy()
    p_mkt = valid["p_market"].to_numpy()
    outc = valid["outcome"].to_numpy()

    brier_mod = brier_score(p_mod, outc)
    brier_mkt = brier_score(p_mkt, outc)
    cal_gap = calibration_gap(p_mod, p_mkt)
    edge = edge_accuracy(p_mod, p_mkt, outc)

    return {
        "model": model_name,
        "n": len(valid),
        "brier_model": brier_mod,
        "brier_market": brier_mkt,
        "calib_gap": cal_gap,
        "edge": edge,
        "per_bracket": df,
    }
