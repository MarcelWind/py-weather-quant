#!/usr/bin/env python3
"""Weather forecast model comparison — main entry point.

Loads joined data, fits B0/B2/B3 models per (city, ens_model) cell,
computes all evaluation metrics, and generates publication-quality figures.

Usage:
    python src/run.py --data data/joined_data.csv --output output/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import (
    load_joined,
    compute_lead_buckets,
    filter_valid,
    train_test_split_chronological,
    get_cell,
)
from src.models import all_models
from src.metrics import evaluate_model
from src.plots import (
    plot_time_series,
    plot_crps_comparison,
    plot_calibration_curves,
    plot_reliability_diagrams,
    plot_sharpness_entropy,
    plot_metric_heatmap,
    plot_per_city_metrics,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Weather model comparison showcase")
    ap.add_argument("--data", default="data/joined_data.csv",
                    help="Path to joined dataset CSV")
    ap.add_argument("--output", default="output/",
                    help="Output directory for results and figures")
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load & prepare data ────────────────────────────────────────────────
    print("Loading data ...", file=sys.stderr)
    df_raw = load_joined(data_path)
    df = compute_lead_buckets(df_raw)
    df = filter_valid(df)
    print(f"  {len(df)} valid rows after filtering", file=sys.stderr)

    # Unique cells: (city, ens_model, simple_model)
    cells = df[["city", "ens_model", "simple_model"]].drop_duplicates()
    print(f"  {len(cells)} (city, ens_model, simple_model) cells", file=sys.stderr)

    # ── Iterate cells, fit & evaluate ──────────────────────────────────────
    all_results: list[dict] = []
    _plotted_cities: set[str] = set()
    # For calibration curves (aggregate by model across all cells)
    calib_accum: dict[str, dict[str, list]] = {}
    rel_accum: dict[str, dict[str, list]] = {}

    for _, cell_row in cells.iterrows():
        city = cell_row["city"]
        ens_model = cell_row["ens_model"]
        simple_model = cell_row["simple_model"]
        cell_key = f"{city}/{ens_model}"

        cell_df = get_cell(df, city, ens_model, simple_model)
        if len(cell_df) < 10:
            print(f"  SKIP {cell_key}: only {len(cell_df)} rows", file=sys.stderr)
            continue

        train, test = train_test_split_chronological(cell_df)
        print(f"  {cell_key}: {len(train)} train / {len(test)} test", file=sys.stderr)

        models = all_models()
        mu_dict: dict[str, np.ndarray] = {}
        sigma_dict: dict[str, np.ndarray] = {}

        for model in models:
            model.fit(train)
            mu, sigma = model.predict(test)
            mu_dict[model.name] = mu
            sigma_dict[model.name] = sigma

            result = evaluate_model(mu, sigma, test["T_obs"].to_numpy(),
                                    model_name=model.name)
            result["city"] = city
            result["ens_model"] = ens_model
            result["simple_model"] = simple_model
            result["n_train"] = len(train)
            result["n_test"] = len(test)
            all_results.append(result)

            # Accumulate calibration data
            calib_accum.setdefault(model.name, {"x": [], "y": []})
            calib_accum[model.name]["x"].extend(result["calibration_x"])
            calib_accum[model.name]["y"].extend(result["calibration_y"])

            rel_accum.setdefault(model.name, {"x": [], "y": []})
            rel_accum[model.name]["x"].extend(result["calibration_x"])
            rel_accum[model.name]["y"].extend(result["calibration_y"])

        # ── Time series plot (first cell per city) ─────────────────────────
        # Use test portion for visualization
        test_dates = pd.to_datetime(test["target_date"])
        obs = test["T_obs"].to_numpy()
        if city not in _plotted_cities:
            _plotted_cities.add(city)
            plot_time_series(
                test_dates, obs, mu_dict, sigma_dict, city,
            )

    # ── Build results DataFrame ────────────────────────────────────────────
    # Strip calibration arrays (too large for CSV) and internal flags
    df_out = pd.DataFrame(all_results)
    export_cols = ["city", "ens_model", "simple_model", "model",
                   "n", "crps", "brier", "log_loss", "sharpness",
                   "entropy", "reliability_gap", "n_train", "n_test"]
    df_export = df_out[export_cols].copy()
    df_export.to_csv(out_dir / "results.csv", index=False)
    print(f"\nResults written to {out_dir / 'results.csv'}", file=sys.stderr)

    # Print summary
    print("\n── Summary ─────────────────────────────────────────────", file=sys.stderr)
    summary = df_export.groupby("model")[
        ["crps", "brier", "log_loss", "sharpness", "entropy"]
    ].mean()
    print(summary.round(4).to_string(), file=sys.stderr)

    # ── Generate all figures ───────────────────────────────────────────────
    print("\nGenerating figures ...", file=sys.stderr)

    fig1 = plot_crps_comparison(df_export)
    print(f"  {fig1} CRPS comparison", file=sys.stderr)

    # Calibration curves (re-bin aggregated data)
    calib_plot_data: dict[str, dict] = {}
    for model_name, accum in calib_accum.items():
        x = np.array(accum["x"])
        y = np.array(accum["y"])
        # Bin into 10 equal-width bins
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        binned_y = np.full(10, np.nan)
        for i in range(10):
            mask = (x >= bins[i]) & (x < bins[i + 1])
            if mask.any():
                binned_y[i] = np.nanmean(y[mask])
        calib_plot_data[model_name] = {"x": bin_centers, "y": binned_y}

    fig2 = plot_calibration_curves(calib_plot_data)
    print(f"  {fig2} calibration curves", file=sys.stderr)

    fig3 = plot_reliability_diagrams(calib_plot_data)
    print(f"  {fig3} reliability diagrams", file=sys.stderr)

    fig4 = plot_sharpness_entropy(df_export)
    print(f"  {fig4} sharpness & entropy", file=sys.stderr)

    fig5 = plot_metric_heatmap(df_export)
    print(f"  {fig5} metric heatmap", file=sys.stderr)

    fig6 = plot_per_city_metrics(df_export)
    print(f"  {fig6} per-city metrics", file=sys.stderr)

    print(f"\nAll figures saved to {out_dir / 'figures'}/", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
