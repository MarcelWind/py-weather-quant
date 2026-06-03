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
CITY_LABELS = {"nyc": "New York", "london": "London", "tokyo": "Tokyo"}

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
