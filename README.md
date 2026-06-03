# py-weather-quant

**Probabilistic weather forecasting — a quantitative comparison of three modeling approaches.**

This project extracts real-world ensemble and deterministic weather forecasts from global and regional providers (ECMWF, ICON, GFS, HRRR), then evaluates how well three different probabilistic models predict the observed maximum temperature. It is a self-contained showcase of rigorous forecast evaluation methodology across three geographically diverse cities.

---

## Models

| Tag | Model | Description |
|-----|-------|-------------|
| **B0** | Raw Ensemble | Ensemble mean $\mu = F_e$, ensemble spread $\sigma = \max(s_e, \sqrt{0.01})$. No training required. |
| **B2** | Simple Gaussian | Deterministic forecast $F_s$ with constant variance $\sigma^2 = \max(\text{Var}(F_s - T_\text{obs}), 0.01)$ estimated from the training set. |
| **B3** | Hybrid Bates–Granger | Per-lead-time-bucket bias correction on both $F_e$ and $F_s$, then inverse-variance weighted combination. Falls back to global statistics when a bucket has fewer than 5 training rows. |

All models share a unified `.fit(train) → .predict(features) → (μ, σ)` interface.

---

## Evaluation Metrics

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| **CRPS** | $\text{CRPS}(\mathcal{N}(\mu,\sigma^2), y) = \sigma\left[ \frac{y-\mu}{\sigma}\big(2\Phi(\frac{y-\mu}{\sigma})-1\big) + 2\phi(\frac{y-\mu}{\sigma}) - \frac{1}{\sqrt{\pi}} \right]$ | Sharpness + calibration combined (negatively oriented) |
| **Brier Score** | $\frac{1}{K}\sum_{k=1}^K (p_k - o_k)^2$ | Probability forecast accuracy at pre-defined temperature brackets |
| **Log Loss** | $-\frac{1}{K}\sum_{k=1}^K [o_k \log p_k + (1-o_k)\log(1-p_k)]$ | Predictive density quality |
| **Sharpness** | $\bar{\sigma}^2$ | Average predictive variance (lower is sharper) |
| **Prediction Entropy** | $\overline{\frac{1}{2}\log(2\pi e \sigma^2)}$ | Differential entropy of the predictive distribution |
| **Calibration** | Reliability diagram | Observed frequency vs. predicted probability (perfect = diagonal) |

---

## Results

Aggregated over all location–model combinations (3 cities × 2 ensemble models × 2 deterministic models):

```
                           crps   brier  log_loss  sharpness  entropy
model
B0 Raw Ensemble          5.0311  0.0994    0.3164    62.1026   3.3591
B2 Simple Gaussian       0.7960  0.0528    0.1897     1.1643   1.4335
B3 Hybrid Bates-Granger  0.7084  0.0553    0.1905     1.0405   1.3603
```

The hybrid model (B3) achieves the lowest CRPS (0.71), 86% lower than the raw ensemble baseline. B2's simple Gaussian already performs very well (CRPS 0.80) with the main gain from B3 coming in sharpness (1.04 vs. 1.16) — the hybrid blend produces tighter predictive distributions.

**Notable:** Chicago with GFS HRRR achieves the best Brier scores (≈0.003), while Mexico City and Tokyo with ECMWF IFS show higher uncertainty (Brier ≈0.06–0.10), reflecting the different forecast skill levels across regions and model types.

---

## Project Structure

```
py-weather-quant/
├── src/
│   ├── data.py          # Data loading, filtering, train/test split
│   ├── models.py        # B0, B2, B3 forecast models
│   ├── metrics.py       # CRPS, Brier, log-loss, calibration, sharpness, entropy
│   ├── plots.py         # 7 publication-quality figure types
│   └── run.py           # End-to-end pipeline: load → fit → evaluate → plot
├── data/
│   ├── joined_data.csv  # Extracted & joined dataset (3 cities, 26 days)
│   └── raw/             # Bundled raw source data (ensembles + forecasts + obs)
├── output/
│   ├── results.csv      # Per-cell evaluation metrics
│   └── figures/         # Generated figures (PNG + PDF)
├── notebooks/
│   └── showcase.ipynb   # Interactive walkthrough
├── extract_data.py      # Regenerate joined_data.csv from data/raw/
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (data/joined_data.csv ships with the repo)
python -m src.run --data data/joined_data.csv --output output/

# 3. Explore interactively
jupyter notebook notebooks/showcase.ipynb
```

### Regenerate the dataset from bundled raw data

```bash
python extract_data.py --raw data/raw --dest data/
```

---

## Figure Gallery

| Figure | Description |
|--------|-------------|
| `time_series_{city}.png` | Test-set predictions with 68%/95% confidence bands vs. observations |
| `crps_comparison.png` | CRPS by model and city, grouped bar chart |
| `calibration_curves.png` | Reliability curves per model across all locations |
| `reliability_diagrams.png` | Reliability diagrams with gap annotations |
| `sharpness_entropy.png` | Sharpness vs. prediction entropy scatter |
| `metric_heatmap.png` | Metric correlation matrix |
| `per_city_metrics.png` | Metric breakdown by city |

---

## Methodology

**Data.** Weather forecasts for Mexico City, Chicago, and Tokyo (April–May 2026). For each city we collect:
- **Ensemble forecasts** from ECMWF IFS ENS (51 members) and ICON global ensemble (40 members): ensemble mean $F_e$ and spread $s_e$.
- **Deterministic forecasts** from GFS HRRR (Chicago), ECMWF IFS (Mexico City, Tokyo): single-value $F_s$.
- **Observations** $T_\text{obs}$ from METAR / synoptic stations.

These three cities represent distinct climate zones and forecast regimes:
| City | Climate | Simple Model | Challenge |
|------|---------|-------------|-----------|
| Mexico City | Subtropical highland (23°C avg) | ECMWF IFS | High-altitude, complex terrain |
| Chicago | Humid continental | GFS HRRR (3 km) | Rapid weather swings, lake effect |
| Tokyo | Humid subtropical | ECMWF IFS | Coastal, typhoon-prone region |

**Train/test split.** Chronological 80/20 per location–model cell. All metrics are reported on the held-out test set (5 days per cell).

**Probabilistic evaluation.** Bracket probabilities are computed analytically via the Gaussian CDF for nine temperature brackets spanning $<0$ to $\geq 35$ °C (all cities report in Celsius).

---

## License

MIT
