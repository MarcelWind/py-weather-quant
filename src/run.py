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
    load_polymarket_prices,
)
from src.models import all_models
from src.metrics import evaluate_model, compare_model_to_market
from src.plots import (
    plot_time_series,
    plot_crps_comparison,
    plot_calibration_curves,
    plot_reliability_diagrams,
    plot_sharpness_entropy,
    plot_metric_heatmap,
    plot_per_city_metrics,
    plot_bracket_probabilities,
    plot_model_vs_market_scatter,
    plot_brier_comparison,
    plot_lead_time_bracket_evolution,
)


POLYMARKET_SLUG_PREFIX = "highest-temperature-in-"
POLYMARKET_SLUG_SUFFIX = "-on-"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Weather model comparison showcase")
    ap.add_argument("--data", default="data/joined_data.csv",
                    help="Path to joined dataset CSV")
    ap.add_argument("--output", default="output/",
                    help="Output directory for results and figures")
    ap.add_argument("--polymarket", default=None,
                    help="Path to Polymarket price data directory. "
                         "If provided, compares model probabilities to market prices.")
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

    # ═══════════════════════════════════════════════════════════════════════
    # Polymarket comparison (if --polymarket path provided)
    # ═══════════════════════════════════════════════════════════════════════
    if args.polymarket:
        polymarket_root = Path(args.polymarket)
        if not polymarket_root.is_dir():
            print(f"WARNING: Polymarket directory not found: {polymarket_root}",
                  file=sys.stderr)
        else:
            print("\n── Polymarket comparison ────────────────────────────",
                  file=sys.stderr)

            # Polymarket data is only available for Chicago currently
            pm_cities = set()
            # Detect which cities have polymarket data
            for city in df["city"].unique():
                subdir = polymarket_root / f"{POLYMARKET_SLUG_PREFIX}{city}{POLYMARKET_SLUG_SUFFIX}"
                if subdir.is_dir() and list(subdir.glob("*.csv")):
                    pm_cities.add(city)
                elif list(polymarket_root.glob("*.csv")):
                    pm_cities.add(city)

            if not pm_cities:
                print("  No Polymarket data found for any city.", file=sys.stderr)
            else:
                print(f"  Cities with Polymarket data: {pm_cities}", file=sys.stderr)
                pm_prices_cache: dict[str, pd.DataFrame] = {}

                for city in pm_cities:
                    try:
                        pm_prices_cache[city] = load_polymarket_prices(
                            polymarket_root, city,
                        )
                        print(f"  Loaded Polymarket prices for {city}: "
                              f"{len(pm_prices_cache[city])} bracket-date pairs",
                              file=sys.stderr)
                    except (FileNotFoundError, ValueError) as e:
                        print(f"  SKIP {city}: {e}", file=sys.stderr)

                # Compare each model's predictions to market for each (city, ens_model) cell
                comparison_results: list[dict] = []
                all_per_bracket_dfs: list[pd.DataFrame] = []
                plotted_bracket_dates: set[str] = set()

                for _, cell_row in cells.iterrows():
                    city = cell_row["city"]
                    if city not in pm_prices_cache:
                        continue

                    ens_model = cell_row["ens_model"]
                    simple_model = cell_row["simple_model"]
                    cell_key = f"{city}/{ens_model}"

                    cell_df = get_cell(df, city, ens_model, simple_model)
                    if len(cell_df) < 10:
                        continue

                    train, test = train_test_split_chronological(cell_df)

                    models = all_models()
                    for model in models:
                        model.fit(train)
                        mu, sigma = model.predict(test)

                        comp = compare_model_to_market(
                            mu, sigma,
                            test["T_obs"].to_numpy(),
                            pm_prices_cache[city],
                            test["target_date"].to_numpy(),
                            model_name=model.name,
                        )
                        comp["city"] = city
                        comp["ens_model"] = ens_model
                        comp["simple_model"] = simple_model
                        comparison_results.append(comp)

                        per_bracket = comp.get("per_bracket")
                        if per_bracket is not None and not per_bracket.empty:
                            per_bracket["city"] = city
                            per_bracket["model"] = model.name
                            per_bracket["ens_model"] = ens_model
                            all_per_bracket_dfs.append(per_bracket)

                            # Plot bracket probabilities for a few example dates
                            example_dates = sorted(per_bracket["target_date"].unique())
                            for ex_date in example_dates[:3]:  # first 3 dates
                                date_key = f"{city}_{ex_date}_{model.name}"
                                if date_key not in plotted_bracket_dates:
                                    plotted_bracket_dates.add(date_key)
                                    plot_bracket_probabilities(
                                        per_bracket, city, ex_date, model.name,
                                    )

                # ── Aggregate comparison results ──────────────────────────
                if comparison_results:
                    # Filter valid results
                    valid_comp = [r for r in comparison_results
                                  if r["n"] >= 5 and np.isfinite(r.get("brier_model", float("nan")))]
                    if valid_comp:
                        # Print comparison table
                        comp_rows = []
                        for r in valid_comp:
                            comp_rows.append({
                                "city": r.get("city", ""),
                                "model": r["model"],
                                "n": r["n"],
                                "brier_model": r["brier_model"],
                                "brier_market": r["brier_market"],
                                "calib_gap": r["calib_gap"],
                                "edge": r.get("edge", {}).get("edge", float("nan")),
                            })
                        comp_df = pd.DataFrame(comp_rows)
                        comp_df.to_csv(out_dir / "polymarket_comparison.csv",
                                       index=False)
                        print("\n  Polymarket Comparison Summary:",
                              file=sys.stderr)
                        print(comp_df.groupby("model")[
                            ["brier_model", "brier_market", "calib_gap", "edge"]
                        ].mean().round(4).to_string(), file=sys.stderr)

                        # Generate comparison plots
                        plot_brier_comparison(valid_comp)
                        print("  brier_comparison plot", file=sys.stderr)

                        if all_per_bracket_dfs:
                            combined_pb = pd.concat(all_per_bracket_dfs,
                                                    ignore_index=True)
                            for city in pm_cities:
                                city_pb = combined_pb[
                                    combined_pb["city"] == city
                                ]
                                if city_pb.empty:
                                    continue
                                plot_model_vs_market_scatter(city_pb, city)
                                print(f"  model_vs_market_scatter_{city}",
                                      file=sys.stderr)
                                plot_lead_time_bracket_evolution(
                                    city_pb, np.array([]), city,
                                )
                                print(f"  lead_time_bracket_evolution_{city}",
                                      file=sys.stderr)

    print(f"\nAll figures saved to {out_dir / 'figures'}/", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
