#!/usr/bin/env python3
"""Regenerate data/joined_data.csv from the bundled raw data in data/raw/.

Reads ensemble forecast CSVs, simple-model forecast CSVs, and a trimmed
observations file — all shipped inside this repo under ``data/raw/``.
No external dependencies required.

Produces ``data/joined_data.csv`` — a clean table of ensemble forecasts,
simple-model forecasts, and observed daily-max temperatures for 3 cities.

Usage:
    python extract_data.py --raw data/raw --dest data/
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# ── Config ───────────────────────────────────────────────────────────────────

CITIES = ["mexico-city", "chicago", "tokyo"]
ENSEMBLE_MODELS = ["ecmwf_ifs025_ensemble", "icon_global_eps"]
SIMPLE_MODELS = {
    "mexico-city": "ecmwf_ifs",
    "chicago": "gfs_hrrr",
    "tokyo": "ecmwf_ifs",
}
DATA_WINDOW_DAYS = 60  # Use the most recent N days of data


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract showcase dataset from bundled raw data")
    ap.add_argument("--raw", default="data/raw",
                    help="Path to raw data directory (default data/raw)")
    ap.add_argument("--dest", default="data",
                    help="Destination directory for extracted joined_data.csv")
    return ap.parse_args(argv)


def ensemble_stats(temps: list[float], masses: list[float]) -> tuple[float, float]:
    """E[T] and Var[T] from discrete PMF."""
    t = np.asarray(temps, dtype=float)
    m = np.asarray(masses, dtype=float)
    total = m.sum()
    if total <= 0 or len(t) == 0:
        return float("nan"), float("nan")
    m = m / total
    fe = float(np.dot(t, m))
    var = float(np.dot(t ** 2, m) - fe ** 2)
    return fe, max(var, 0.0)


def parse_ensemble_csv(path: Path) -> dict:
    """Parse one ensemble CSV file into {target_date, temps, masses, runtime}.

    Handles files that embed both max_temp and min_temp sections (common for
    Tokyo/Asia). Only reads max_temp rows.
    """
    out: dict = {"temps": [], "masses": []}
    max_col = "max_temp"
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw = row.get(max_col, "")
            try:
                val = float(raw)
            except (ValueError, TypeError):
                continue  # skip min_temp section rows
            out["target_date"] = row.get("date", out.get("target_date", ""))
            out["temps"].append(val)
            out["masses"].append(float(row.get("mass_probability", "0")))
            out["runtime"] = row.get("runtime", out.get("runtime", ""))
    return out


def simple_daily_max(path: Path, model_col: str) -> float:
    """Daily max of a model column in an Open-Meteo forecast CSV."""
    vals: list[float] = []
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            v = row.get(model_col)
            if v:
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
    return float(max(vals)) if vals else float("nan")


def load_observations_from_backtest(
    backtest_csv: Path, cities: list[str],
) -> dict[tuple[str, str], float]:
    """Extract T_obs_actual per (city, target_date) from backtest CSV."""
    obs: dict[tuple[str, str], float] = {}
    with backtest_csv.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            city = row.get("city", "")
            if city not in cities:
                continue
            date = row.get("target_date", "")
            key = (city, date)
            if key not in obs:
                try:
                    obs[key] = float(row.get("T_obs_actual", "nan"))
                except (ValueError, TypeError):
                    pass
    return obs


def main() -> None:
    args = parse_args()
    raw = Path(args.raw).resolve()
    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    obs_csv = raw / "observations.csv"
    if not obs_csv.is_file():
        print(f"ERROR: {obs_csv} not found.", file=sys.stderr)
        sys.exit(1)

    print("Loading observations ...", file=sys.stderr)
    obs_map = load_observations_from_backtest(obs_csv, CITIES)
    print(f"  Loaded {len(obs_map)} (city, date) observation pairs", file=sys.stderr)

    # Collect date range
    all_dates: set[str] = set()
    for (_, date) in obs_map:
        all_dates.add(date)
    sorted_dates = sorted(all_dates)
    if DATA_WINDOW_DAYS and len(sorted_dates) > DATA_WINDOW_DAYS:
        keep_dates = set(sorted_dates[-DATA_WINDOW_DAYS:])
    else:
        keep_dates = all_dates

    print(f"  Date range: {min(keep_dates)} to {max(keep_dates)} ({len(keep_dates)} days)", file=sys.stderr)

    # Build joined rows
    rows: list[dict] = []

    for city in CITIES:
        simple_model = SIMPLE_MODELS[city]
        simple_dir = raw / "simple" / city

        for ens_model in ENSEMBLE_MODELS:
            ens_dir = raw / "ensembles" / city / ens_model
            if not ens_dir.is_dir():
                print(f"  SKIP {city}/{ens_model}: directory not found", file=sys.stderr)
                continue

            # Group ensemble files by target_date, pick latest iteration per date
            date_files: dict[str, list[Path]] = defaultdict(list)
            for f in sorted(ens_dir.iterdir()):
                if f.suffix.lower() != ".csv":
                    continue
                # Parse filename: {city}_forecast_{date}_no{iteration}.csv
                parts = f.stem.split("_no")
                if len(parts) != 2:
                    continue
                try:
                    date_part = f.stem.split("_forecast_")[1].split("_no")[0]
                except IndexError:
                    continue
                date_files[date_part].append(f)

            for target_date, files in date_files.items():
                if target_date not in keep_dates:
                    continue

                # Use latest iteration
                latest = sorted(files)[-1]
                parsed = parse_ensemble_csv(latest)
                fe, var_e = ensemble_stats(parsed["temps"], parsed["masses"])
                se = math.sqrt(var_e) if var_e > 0 else 0.0
                if not math.isfinite(fe):
                    continue

                # Simple model daily max
                simple_path = simple_dir / f"{city}_forecast_{target_date}.csv"
                if not simple_path.is_file():
                    continue
                fs = simple_daily_max(simple_path, simple_model)
                if not math.isfinite(fs):
                    continue

                # Observation
                tobs = obs_map.get((city, target_date), float("nan"))
                if not math.isfinite(tobs):
                    continue

                rows.append({
                    "city": city,
                    "target_date": target_date,
                    "ens_model": ens_model,
                    "simple_model": simple_model,
                    "runtime_utc": parsed.get("runtime", ""),
                    "F_e": fe,
                    "s_e": se,
                    "F_s": fs,
                    "T_obs": tobs,
                })

    # Write
    out_path = dest / "joined_data.csv"
    fieldnames = [
        "city", "target_date", "ens_model", "simple_model",
        "runtime_utc", "F_e", "s_e", "F_s", "T_obs",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWritten {len(rows)} rows to {out_path}", file=sys.stderr)
    print(f"  Cities: {set(r['city'] for r in rows)}", file=sys.stderr)
    print(f"  Ensemble models: {set(r['ens_model'] for r in rows)}", file=sys.stderr)
    print(f"  Date range: {min(r['target_date'] for r in rows)} to {max(r['target_date'] for r in rows)}", file=sys.stderr)


if __name__ == "__main__":
    main()
