"""Probabilistic temperature forecast models.

Three model variants with a unified ``.fit(predictors, targets)`` /
``.predict(predictors) -> (mu, sigma)`` interface.

Variants
--------
B0 (Raw Ensemble)
    Use ensemble mean and std directly. No training required.
    μ = F_e, σ = s_e

B2 (Simple Gaussian)
    Use simple deterministic forecast as mean, constant variance from
    training MSE. μ = F_s, σ = √(MSE_train)

B3 (Hybrid Bates-Granger)
    Bias-correct each model, combine via inverse-variance weighting
    per lead-time bucket. μ = wₑ(F_e - δₑ) + wₛ(F_s - δₛ),
    σ² = wₑ² vₑ + wₛ² vₛ
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .data import SIGMA_MIN_SQ

# ── Per-bucket statistics ───────────────────────────────────────────────────


def _per_bucket_stats(train: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Per lead-time bucket: bias_e, bias_s, var_e, var_s."""
    out: dict[str, dict[str, float]] = {}
    for bucket, grp in train.groupby("bucket"):
        if len(grp) < 5:
            continue
        e_err = grp["F_e"].to_numpy() - grp["T_obs"].to_numpy()
        s_err = grp["F_s"].to_numpy() - grp["T_obs"].to_numpy()
        out[bucket] = {
            "bias_e": float(np.mean(e_err)),
            "bias_s": float(np.mean(s_err)),
            "var_e": float(np.var(e_err, ddof=1)) if len(grp) > 1 else float("nan"),
            "var_s": float(np.var(s_err, ddof=1)) if len(grp) > 1 else float("nan"),
            "n": int(len(grp)),
        }
    # Global fallback
    if not train.empty:
        e_err = train["F_e"].to_numpy() - train["T_obs"].to_numpy()
        s_err = train["F_s"].to_numpy() - train["T_obs"].to_numpy()
        out["__global__"] = {
            "bias_e": float(np.mean(e_err)),
            "bias_s": float(np.mean(s_err)),
            "var_e": float(np.var(e_err, ddof=1)) if len(train) > 1 else float("nan"),
            "var_s": float(np.var(s_err, ddof=1)) if len(train) > 1 else float("nan"),
            "n": int(len(train)),
        }
    return out


# ── Base class ───────────────────────────────────────────────────────────────


class ForecastModel(ABC):
    """Abstract base for a weather forecast model."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> None:
        """Fit model parameters on training data."""

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (μ, σ) arrays over the feature rows."""


# ── B0: Raw Ensemble ─────────────────────────────────────────────────────────


class B0RawEnsemble(ForecastModel):
    """Use ensemble mean and std directly.

    μ = F_e, σ = s_e
    No training required.
    """

    @property
    def name(self) -> str:
        return "B0 Raw Ensemble"

    def fit(self, train: pd.DataFrame) -> None:
        pass  # no parameters to fit

    def predict(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        mu = features["F_e"].to_numpy().copy()
        sigma = np.maximum(features["s_e"].to_numpy(), math.sqrt(SIGMA_MIN_SQ))
        return mu, sigma


# ── B2: Simple Gaussian ──────────────────────────────────────────────────────


class B2SimpleGaussian(ForecastModel):
    """Simple deterministic model with constant Gaussian uncertainty.

    μ = F_s, σ = √(MSE of F_s vs T_obs on training set)
    """

    def __init__(self) -> None:
        self._var: float = SIGMA_MIN_SQ

    @property
    def name(self) -> str:
        return "B2 Simple Gaussian"

    def fit(self, train: pd.DataFrame) -> None:
        s_err = train["F_s"].to_numpy() - train["T_obs"].to_numpy()
        var = float(np.var(s_err, ddof=1)) if len(train) > 1 else 1.0
        self._var = max(var, SIGMA_MIN_SQ)

    def predict(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        mu = features["F_s"].to_numpy().copy()
        sigma = np.full_like(mu, math.sqrt(self._var))
        return mu, sigma


# ── B3: Hybrid Bates-Granger ─────────────────────────────────────────────────


class B3HybridBatesGranger(ForecastModel):
    """Bates-Granger combination with bias correction.

    Per lead-time bucket:
      1. Remove per-model bias: F_e' = F_e - bias_e, F_s' = F_s - bias_s
      2. Inverse-variance weights: wₑ = vₛ / (vₑ + vₛ), wₛ = 1 - wₑ
      3. μ = wₑ ⋅ F_e' + wₛ ⋅ F_s'
      4. σ² = wₑ² ⋅ vₑ + wₛ² ⋅ vₛ

    Falls back to global (cross-bucket) statistics when a bucket has <5 rows.
    """

    def __init__(self) -> None:
        self._bucket_stats: dict[str, dict[str, float]] = {}

    @property
    def name(self) -> str:
        return "B3 Hybrid Bates-Granger"

    def fit(self, train: pd.DataFrame) -> None:
        self._bucket_stats = _per_bucket_stats(train)

    def predict(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        mu = np.empty(len(features))
        sigma = np.empty(len(features))
        for i, (_, row) in enumerate(features.iterrows()):
            bucket = row["bucket"]
            cell = (
                self._bucket_stats.get(bucket)
                if self._bucket_stats.get(bucket, {}).get("n", 0) >= 5
                else self._bucket_stats.get("__global__")
            )
            if cell is None:
                mu[i] = float("nan")
                sigma[i] = float("nan")
                continue
            ve, vs = cell["var_e"], cell["var_s"]
            if not (math.isfinite(ve) and math.isfinite(vs) and ve + vs > 0):
                mu[i] = float("nan")
                sigma[i] = float("nan")
                continue
            we = vs / (ve + vs)
            ws = 1.0 - we
            m = we * (row["F_e"] - cell["bias_e"]) + ws * (row["F_s"] - cell["bias_s"])
            v = we**2 * ve + ws**2 * vs
            mu[i] = float(m)
            sigma[i] = math.sqrt(max(v, SIGMA_MIN_SQ))
        return mu, sigma


# ── Factory ───────────────────────────────────────────────────────────────────


def all_models() -> list[ForecastModel]:
    """Return a list of all model instances."""
    return [B0RawEnsemble(), B2SimpleGaussian(), B3HybridBatesGranger()]
