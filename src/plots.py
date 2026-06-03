"""Publication-quality visualizations for the weather forecast model showcase.

Generates:
  - Time series with ±SD uncertainty bands
  - CRPS comparison bar chart
  - Calibration curves
  - Reliability diagrams
  - Sharpness & entropy comparison
  - Metric summary heatmap
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── Global style ─────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = {"B0 Raw Ensemble": "#4C72B0", "B2 Simple Gaussian": "#DD8452",
          "B3 Hybrid Bates-Granger": "#55A868"}
CITY_LABELS = {"nyc": "New York", "london": "London", "tokyo": "Tokyo",
                "chicago": "Chicago", "mexico-city": "Mexico City"}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = _PROJECT_ROOT / "output" / "figures"


def _save(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Time Series with ±SD bands
# ═══════════════════════════════════════════════════════════════════════════════


def plot_time_series(
    dates: np.ndarray,
    obs: np.ndarray,
    predictions: dict[str, np.ndarray],
    sigmas: dict[str, np.ndarray],
    city: str,
) -> plt.Figure:
    """One panel per model: predicted mean ± 1σ band + observed values."""
    models = list(predictions.keys())
    fig, axes = plt.subplots(len(models), 1, figsize=(12, 3 * len(models)),
                             sharex=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        mu = predictions[model]
        sigma = sigmas[model]
        color = COLORS.get(model, "#333333")

        ax.fill_between(dates, mu - sigma, mu + sigma,
                        alpha=0.25, color=color, label=f"{model} ±1σ")
        ax.plot(dates, mu, color=color, linewidth=1.8, label=f"{model} mean")
        ax.scatter(dates, obs, color="black", s=15, zorder=5,
                   label="Observed", marker=".")

        ax.set_ylabel("Temperature")
        ax.legend(loc="upper left", fontsize=9)
        ax.set_title(f"{model}", fontsize=11, fontweight="bold")

    axes[-1].set_xlabel("Date")
    fig.suptitle(f"{CITY_LABELS.get(city, city)} — Forecast Time Series",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, f"time_series_{city}")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CRPS Comparison Bar Chart
# ═══════════════════════════════════════════════════════════════════════════════


def plot_crps_comparison(metrics_df: pd.DataFrame) -> plt.Figure:
    """Grouped bar chart: CRPS per model × city."""
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = metrics_df.pivot_table(
        index="city", columns="model", values="crps", aggfunc="mean",
    )
    pivot.index = [CITY_LABELS.get(c, c) for c in pivot.index]
    pivot.plot(kind="bar", ax=ax, color=[COLORS.get(c, "#999") for c in pivot.columns],
               edgecolor="white", width=0.75)

    ax.set_ylabel("Mean CRPS (lower is better)")
    ax.set_title("Probabilistic Forecast Skill — CRPS Comparison")
    ax.legend(title="Model", fontsize=9)
    ax.set_xlabel("")
    fig.tight_layout()
    _save(fig, "crps_comparison")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Calibration Curves
# ═══════════════════════════════════════════════════════════════════════════════


def plot_calibration_curves(
    calib_data: dict[str, dict],
) -> plt.Figure:
    """One panel per model: predicted prob vs observed frequency."""
    models = list(calib_data.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.5),
                             sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        data = calib_data[model]
        color = COLORS.get(model, "#333")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect calibration")
        ax.plot(data["x"], data["y"], "o-", color=color, linewidth=1.8,
                markersize=5, label=model)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Predicted probability")
        if ax == axes[0]:
            ax.set_ylabel("Observed frequency")
        ax.set_title(model, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_aspect("equal")

    fig.suptitle("Calibration Curves", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "calibration_curves")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Reliability Diagrams
# ═══════════════════════════════════════════════════════════════════════════════


def plot_reliability_diagrams(
    rel_data: dict[str, dict],
) -> plt.Figure:
    """One panel per model with reliability gap bars below."""
    models = list(rel_data.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5),
                             sharey=False)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        data = rel_data[model]
        color = COLORS.get(model, "#333")

        # Main reliability line
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
        ax.plot(data["x"], data["y"], "o-", color=color, linewidth=1.8,
                markersize=5)
        ax.fill_between(data["x"], data["y"], data["x"],
                        where=data["y"] >= data["x"],
                        color=color, alpha=0.15, label="Overconfident")
        ax.fill_between(data["x"], data["y"], data["x"],
                        where=data["y"] < data["x"],
                        color="red", alpha=0.15, label="Underconfident")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Predicted probability")
        if ax == axes[0]:
            ax.set_ylabel("Observed frequency")
        ax.set_title(model, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        ax.set_aspect("equal")

    fig.suptitle("Reliability Diagrams", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "reliability_diagrams")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Sharpness & Entropy Comparison
# ═══════════════════════════════════════════════════════════════════════════════


def plot_sharpness_entropy(metrics_df: pd.DataFrame) -> plt.Figure:
    """Side-by-side bar charts for sharpness and entropy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, metric, title in [
        (ax1, "sharpness", "Sharpness (mean σ², lower is better)"),
        (ax2, "entropy", "Predictive Entropy (lower is better)"),
    ]:
        pivot = metrics_df.pivot_table(
            index="city", columns="model", values=metric, aggfunc="mean",
        )
        pivot.index = [CITY_LABELS.get(c, c) for c in pivot.index]
        pivot.plot(kind="bar", ax=ax,
                   color=[COLORS.get(c, "#999") for c in pivot.columns],
                   edgecolor="white", width=0.75)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.legend(fontsize=8)
        ax.set_ylabel(metric.replace("_", " ").title())

    fig.suptitle("Predictive Concentration", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "sharpness_entropy")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Metric Summary Heatmap
# ═══════════════════════════════════════════════════════════════════════════════


def plot_metric_heatmap(metrics_df: pd.DataFrame) -> plt.Figure:
    """Heatmap of models × metrics, averaged across cities."""
    metric_cols = ["crps", "brier", "log_loss", "sharpness", "entropy",
                   "reliability_gap"]
    display_names = {
        "crps": "CRPS ↓", "brier": "Brier ↓", "log_loss": "Log-Loss ↓",
        "sharpness": "Sharpness ↓", "entropy": "Entropy ↓",
        "reliability_gap": "Rel. Gap ↓",
    }

    pivot = metrics_df.groupby("model")[metric_cols].mean()
    pivot = pivot.rename(columns=display_names)

    fig, ax = plt.subplots(figsize=(8, 2 + 0.5 * len(pivot)))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlOrRd_r",
                ax=ax, cbar_kws={"label": "Value (lower is better)"},
                linewidths=0.5, linecolor="white")
    ax.set_title("Aggregate Model Performance (across cities)", fontsize=12,
                 fontweight="bold")
    ax.set_ylabel("")
    ax.set_xlabel("")
    fig.tight_layout()
    _save(fig, "metric_heatmap")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Per-city metric bar chart (all metrics normalized)
# ═══════════════════════════════════════════════════════════════════════════════


def plot_per_city_metrics(metrics_df: pd.DataFrame) -> plt.Figure:
    """Faceted bar chart of all metrics per city, grouped by model."""
    metric_cols = ["crps", "brier", "log_loss", "sharpness", "entropy",
                   "reliability_gap"]
    display = {
        "crps": "CRPS", "brier": "Brier", "log_loss": "Log-Loss",
        "sharpness": "Sharp.", "entropy": "Entropy", "reliability_gap": "Rel.Gap",
    }
    cities = metrics_df["city"].unique()
    fig, axes = plt.subplots(1, len(cities), figsize=(5 * len(cities), 4),
                             sharey=True)
    if len(cities) == 1:
        axes = [axes]

    for ax, city in zip(axes, cities):
        sub = metrics_df[metrics_df["city"] == city]
        x = np.arange(len(metric_cols))
        width = 0.25

        for i, model in enumerate(sub["model"].unique()):
            vals = sub[sub["model"] == model][metric_cols].values
            if len(vals) > 0:
                ax.bar(x + (i - 1) * width, vals[0], width,
                       label=model, color=COLORS.get(model, "#999"), edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels([display.get(m, m) for m in metric_cols], fontsize=8)
        ax.set_title(CITY_LABELS.get(city, city), fontsize=11, fontweight="bold")
        ax.legend(fontsize=7)
        if ax == axes[0]:
            ax.set_ylabel("Metric value (lower is better)")

    fig.suptitle("Per-City Model Performance", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "per_city_metrics")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Bracket Probability Bar Chart — model vs market
# ═══════════════════════════════════════════════════════════════════════════════


def plot_bracket_probabilities(
    per_bracket_df: pd.DataFrame,
    city: str,
    target_date: str,
    model_name: str = "",
) -> plt.Figure:
    """Grouped bar chart: model probability vs market price per bracket.

    Saves as ``bracket_probs_{city}_{date}.png``.
    """
    sub = per_bracket_df[
        (per_bracket_df["target_date"] == target_date)
    ].sort_values("low")
    if sub.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, f"No data for {city} on {target_date}",
                ha="center", va="center", transform=ax.transAxes)
        _save(fig, f"bracket_probs_{city}_{target_date}")
        return fig

    labels = sub["bracket_label"].tolist()
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, sub["p_model"].values, width,
           label=f"Model ({model_name})" if model_name else "Model",
           color="#4C72B0", alpha=0.85, edgecolor="white")
    ax.bar(x + width / 2, sub["p_market"].values, width,
           label="Polymarket", color="#DD8452", alpha=0.85, edgecolor="white")

    # Mark outcome bracket
    outcome_mask = sub["outcome"].values == 1.0
    if outcome_mask.any():
        for i, ok in enumerate(outcome_mask):
            if ok:
                ax.annotate("◀ Outcome", (i, 1.02), ha="center", fontsize=8,
                            color="green", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Probability / Price")
    ax.set_title(f"{city.title()} — {target_date}: Model vs Market Bracket Probabilities",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    _save(fig, f"bracket_probs_{city}_{target_date}")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Model vs Market Scatter Plot
# ═══════════════════════════════════════════════════════════════════════════════


def plot_model_vs_market_scatter(
    per_bracket_df: pd.DataFrame,
    city: str,
) -> plt.Figure:
    """Scatter of model probability vs market price, colored by outcome.

    Saves as ``model_vs_market_scatter_{city}.png``.
    """
    df = per_bracket_df.dropna(subset=["p_model", "p_market", "outcome"]).copy()
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center",
                transform=ax.transAxes)
        _save(fig, f"model_vs_market_scatter_{city}")
        return fig

    fig, ax = plt.subplots(figsize=(7, 7))
    colors = df["outcome"].map({0.0: "#e74c3c", 1.0: "#2ecc71"})
    ax.scatter(df["p_market"], df["p_model"], c=colors, alpha=0.6,
               edgecolors="gray", linewidth=0.3, s=30)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Perfect agreement")

    ax.set_xlabel("Market price (Polymarket)")
    ax.set_ylabel("Model probability")
    ax.set_title(f"{city.title()} — Model vs Market: Bracket Probability Scatter",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.legend(fontsize=9)

    # Add correlation text
    corr = df["p_model"].corr(df["p_market"])
    ax.text(0.98, 0.02, f"r = {corr:.3f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    _save(fig, f"model_vs_market_scatter_{city}")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Brier Score Comparison — model vs market across models
# ═══════════════════════════════════════════════════════════════════════════════


def plot_brier_comparison(
    comparison_results: list[dict],
) -> plt.Figure:
    """Grouped bar chart: Brier score (model) vs Brier score (market) per model.

    Saves as ``brier_comparison.png``.
    """
    models = []
    model_briers = []
    market_briers = []
    for res in comparison_results:
        if res["n"] >= 5 and np.isfinite(res.get("brier_model", float("nan"))):
            models.append(res["model"])
            model_briers.append(res["brier_model"])
            market_briers.append(res["brier_market"])

    if not models:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes)
        _save(fig, "brier_comparison")
        return fig

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width / 2, model_briers, width, label="Model Brier",
                   color="#4C72B0", alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width / 2, market_briers, width, label="Market Brier",
                   color="#DD8452", alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Brier Score (lower is better)")
    ax.set_title("Model vs Market: Brier Score Comparison", fontsize=12,
                 fontweight="bold")
    ax.legend(fontsize=9)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    _save(fig, "brier_comparison")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Lead-time Evolution of Bracket Probability Error
# ═══════════════════════════════════════════════════════════════════════════════


def plot_lead_time_bracket_evolution(
    per_bracket_df: pd.DataFrame,
    lead_hours: np.ndarray,
    city: str,
) -> plt.Figure:
    """Line plot: model-market probability gap vs hours until close.

    Saves as ``lead_time_bracket_evolution_{city}.png``.
    """
    df = per_bracket_df.dropna(subset=["p_model", "p_market"]).copy()
    if df.empty or len(lead_hours) == 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No data for lead-time evolution",
                ha="center", va="center", transform=ax.transAxes)
        _save(fig, f"lead_time_bracket_evolution_{city}")
        return fig

    # We don't have per-observation lead times in the bracket comparison,
    # so we aggregate by bracket and show mean abs error per bracket
    df["abs_gap"] = np.abs(df["p_model"] - df["p_market"])
    bracket_gaps = df.groupby("bracket_label")["abs_gap"].agg(["mean", "std", "count"])
    bracket_gaps = bracket_gaps.sort_values("mean", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = bracket_gaps.index.tolist()
    x = np.arange(len(labels))
    means = bracket_gaps["mean"].values
    stds = bracket_gaps["std"].values

    colors = plt.cm.RdYlGn_r(1.0 - means / (means.max() if means.max() > 0 else 1))
    ax.bar(x, means, yerr=stds, color=colors, edgecolor="white",
           capsize=4, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean |Model − Market|")
    ax.set_title(f"{city.title()} — Model-Market Disagreement by Bracket",
                 fontsize=12, fontweight="bold")

    # Add count labels
    for i, (_, row) in enumerate(bracket_gaps.iterrows()):
        ax.text(i, row["mean"] + (stds[i] if not np.isnan(stds[i]) else 0) + 0.01,
                f"n={int(row['count'])}", ha="center", fontsize=7)

    fig.tight_layout()
    _save(fig, f"lead_time_bracket_evolution_{city}")
    return fig
