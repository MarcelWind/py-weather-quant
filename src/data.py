"""Data loading and preprocessing for the weather forecast model comparison.

Provides a clean interface to load the joined dataset and prepare
train/test splits for model fitting.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

# ── Lead-time buckets (reused from hybrid_eval_variants.py) ──────────────────

LEAD_BUCKETS = [
    (96, float("inf"), ">96h"),
    (48, 96, "48-96h"),
    (24, 48, "24-48h"),
    (12, 24, "12-24h"),
    (6, 12, "6-12h"),
    (0, 6, "0-6h"),
    (float("-inf"), 0, "<=0h"),
]
LEAD_ORDER = [b[2] for b in LEAD_BUCKETS]
SIGMA_MIN_SQ = 0.01


def lead_bucket(h: float) -> str:
    """Assign hours-until-close to a lead-time bucket label."""
    if not math.isfinite(h):
        return "no_tz"
    for lo, hi, label in LEAD_BUCKETS:
        if lo <= h < hi:
            return label
    return "??"


def load_joined(path: str | Path) -> pd.DataFrame:
    """Load the joined dataset CSV.

    Columns: city, target_date, ens_model, simple_model, runtime_utc,
             F_e, s_e, F_s, T_obs
    """
    df = pd.read_csv(path, parse_dates=["target_date"])
    return df


def compute_lead_buckets(
    df: pd.DataFrame,
    runtime_col: str = "runtime_utc",
    target_date_col: str = "target_date",
) -> pd.DataFrame:
    """Compute hours-until-close and lead-time bucket for each row.

    Uses midnight UTC of the target date as 'close' time (simplification).
    For a real deployment, this would use per-city timezone-aware close times.
    """
    df = df.copy()
    close = pd.to_datetime(df[target_date_col]) + pd.Timedelta(days=1)
    # Make close tz-aware (UTC) so it can be subtracted from tz-aware runtime
    close = close.dt.tz_localize("UTC")
    runtime = pd.to_datetime(df[runtime_col], utc=True)
    df["hours_until_close"] = (close - runtime).dt.total_seconds() / 3600.0
    df["bucket"] = df["hours_until_close"].map(lead_bucket)
    return df


def filter_valid(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with missing values."""
    mask = (
        df["F_e"].notna()
        & df["s_e"].notna()
        & df["F_s"].notna()
        & df["T_obs"].notna()
    )
    return df[mask].copy()


def train_test_split_chronological(
    df: pd.DataFrame, train_frac: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/test split sorted by target_date + runtime."""
    df = df.sort_values(["target_date", "runtime_utc"]).reset_index(drop=True)
    split = int(len(df) * train_frac)
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()
    return train, test


def get_cell(
    df: pd.DataFrame, city: str, ens_model: str, simple_model: str,
) -> pd.DataFrame:
    """Filter joined data to one (city, ens_model, simple_model) cell."""
    return df[
        (df["city"] == city)
        & (df["ens_model"] == ens_model)
        & (df["simple_model"] == simple_model)
    ].copy()
