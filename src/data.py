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


# ── Polymarket price loading ─────────────────────────────────────────────────


POLYMARKET_SLUG_PREFIX = "highest-temperature-in-"
POLYMARKET_SLUG_SUFFIX = "-on-"


def parse_polymarket_bracket(label: str) -> tuple[float, float]:
    """Parse a Polymarket bracket label into (low, high) temperature range.

    Supports formats:
      "XX°F or below" → (-inf, XX+1)
      "XX-YY°F"       → (XX, YY+1)
      "XX°F or higher" → (XX, +inf)
    """
    label = label.strip()
    if "or below" in label:
        # "55°F or below"
        val_str = label.split("°")[0].strip()
        val = float(val_str)
        return (float("-inf"), val + 1.0)
    elif "or higher" in label:
        # "74°F or higher"
        val_str = label.split("°")[0].strip()
        val = float(val_str)
        return (val, float("inf"))
    elif "-" in label:
        # "56-57°F" or "12-13°F"
        parts = label.split("-")
        lo = float(parts[0].strip())
        hi_str = parts[1].split("°")[0].strip()
        hi = float(hi_str)
        return (lo, hi + 1.0)
    else:
        raise ValueError(f"Cannot parse bracket label: {label}")


def load_polymarket_prices(
    polymarket_root: str | Path,
    city: str,
) -> pd.DataFrame:
    """Load Polymarket YES-token prices for a city.

    Reads all CSV files from ``{polymarket_root}/{slug_prefix}{city}{slug_suffix}/``
    or from ``{polymarket_root}/*.csv`` directly.

    Returns a DataFrame with columns:
        target_date  – date string YYYY-MM-DD
        bracket_label – e.g. "56-57°F"
        price        – last YES-token price (0-1) for this bracket
        low          – lower bound of temperature bracket
        high         – upper bound of temperature bracket
    """
    root = Path(polymarket_root)
    candidates: list[Path] = []

    # Try the subdirectory pattern first
    subdir_name = f"{POLYMARKET_SLUG_PREFIX}{city}{POLYMARKET_SLUG_SUFFIX}"
    subdir = root / subdir_name
    if subdir.is_dir():
        candidates = sorted(subdir.glob("*.csv"))
    else:
        # Flat directory
        candidates = sorted(root.glob("*.csv"))

    if not candidates:
        raise FileNotFoundError(
            f"No Polymarket CSV files found for city '{city}' in {root}"
        )

    rows: list[dict] = []
    for path in candidates:
        target_date = path.stem  # YYYY-MM-DD
        try:
            df = pd.read_csv(
                path,
                usecols=["market", "price", "timestamp"],
                parse_dates=["timestamp"],
            )
        except (ValueError, KeyError):
            # Some files have extra columns; try without usecols
            df = pd.read_csv(path, parse_dates=["timestamp"])

        # Only YES-side data (side column may or may not exist)
        if "side" in df.columns:
            df = df[df["side"].str.upper() == "YES"].copy()

        # Get last price per market (bracket)
        last = df.groupby("market")["price"].last().reset_index()
        last = last.dropna(subset=["price"])

        for _, row in last.iterrows():
            label = str(row["market"]).strip()
            try:
                low, high = parse_polymarket_bracket(label)
            except (ValueError, IndexError):
                continue
            rows.append({
                "target_date": target_date,
                "bracket_label": label,
                "price": float(row["price"]),
                "low": low,
                "high": high,
            })

    if not rows:
        raise ValueError(
            f"No valid Polymarket bracket prices found for city '{city}'"
        )

    result = pd.DataFrame(rows)
    result["target_date"] = pd.to_datetime(result["target_date"])
    return result.sort_values(["target_date", "low"]).reset_index(drop=True)


def polymarket_bracket_outcome(
    T_obs: float, low: float, high: float,
) -> float:
    """Binary outcome: is T_obs in [low, high)?"""
    if not math.isfinite(T_obs):
        return float("nan")
    if math.isfinite(low) and T_obs < low:
        return 0.0
    if math.isfinite(high) and T_obs >= high:
        return 0.0
    return 1.0
