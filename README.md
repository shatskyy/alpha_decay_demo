# Alpha Decay Demo

A small, reproducible project to study **alpha decay** around order execution.  
It simulates signals, orders, fills, and market data; builds labels and features; trains simple models; and outputs **interpretable explanation cards**.

> ⚠️ **Synthetic data. This is not a production trading system.**
---

## Why this matters

Most trading signals predictive power **decays** once orders hit the market. Classic execution focuses on **cost minimization** (reduce slippage vs. a benchmark). For short-horizon strategies you must also think about **signal preservation**—capturing as much of the predicted alpha as possible **before it decays**.

This project shows how to **measure** decay, **model** it, and **explain** the drivers so you can reason about the trade-off between **cost** and **timing**.

---

## Quick start

```bash
# 1) Create a virtual environment and install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Run the full pipeline (simulate → DB → labels → features → train → explain)
python -m src.run_demo
```

Outputs are written to `data/` and `db/`.

---

## What you’ll see (sample)

**Console:**

```
Regression MAE (bps) on test: 14.6
Classification ROC-AUC on test: 0.74
Precision/Recall @ tuned threshold: 0.41 / 0.53
Top permutation importances: spread_bp, imbalance, urgency_tag
```

<img src="data/regression_scatter.png" width="420"> 

The model struggles to predict the **exact size** of alpha decay. Most predictions sit near zero, while true values span roughly –50 to +25 bps. This often happens when the **signal-to-noise ratio (SNR)** is low or regularization is strong on a small dataset.

<img src="data/roc_curve.png" width="420">

The **classification** task (“high decay” vs “not”) shows **useful discrimination** (AUC \~0.72), even on a small test set. The stepped shape reflects the small sample size, but it suggests features like **spread**, **imbalance**, and **urgency** carry information about decay risk.

**Explanation card (one line of `data/explanations.jsonl`):**

Each card summarizes the prediction + drivers for a parent order:

```json
{
  "parent_id": "ORD123",
  "prediction_bps": -5.7,
  "risk_bucket": "High decay",
  "top_drivers": ["spread_bp", "urgency_tag", "imbalance"],
  "suggested_tactics": "Slice smaller, avoid top-of-book",
  "guardrails": "<=10% ADV"
}

Each card is a **plain summary** for a parent order: predicted decay (bps), a risk bucket, the top drivers, and suggested tactics/guardrails. Even if the regression is noisy, the cards remain useful for **decision support**.

---

## Run parts of the pipeline

```bash
# 1) Generate synthetic CSVs
python -m src.simulate_data

# 2) Create SQLite DB and ingest tables
python -m src.ingest

# 3) Compute labels & build features
python -m src.label
python -m src.features

# 4) Train models & produce plots
python -m src.train

# 5) Score test set & emit explanation cards
python -m src.predict_explain
```

---

## Repository structure

```
alpha_decay_demo/
  README.md
  requirements.txt
  src/
    simulate_data.py     # signals, market bars, parent orders, child fills → CSVs
    ingest.py            # create SQLite, tables, bulk insert → db/alpha.sqlite
    label.py             # compute alpha at signal vs. exec; alpha_decay labels → parquet
    features.py          # parent-level features from microstructure & order footprint
    train.py             # ElasticNetCV & LogisticRegressionCV; plots & metrics
    predict_explain.py   # score + generate explanation cards (JSONL)
    run_demo.py          # orchestrator: one command runs everything
  data/                  # CSVs, parquet outputs, plots, explanations.jsonl
  db/                    # SQLite database
  notebooks/
    demo.ipynb           # optional interactive exploration
  docs/
    Guide.md             # detailed guide
    Data_Schema.md       # full column definitions for CSVs & SQLite tables
```

---

## Data flow

1. **Simulate** → `data/signals.csv`, `data/orders.csv`, `data/child_fills.csv`, `data/market.csv`
2. **Ingest** → `db/alpha.sqlite` with indices
3. **Label** → `data/labels.parquet` (alpha decay at parent-order level)
4. **Features** → `data/features.parquet` (microstructure + order footprint)
5. **Train** → metrics & plots saved to `data/`
6. **Explain** → `data/explanations.jsonl` (one card per parent order)

---

## Modeling & evaluation

* **Split**: time-aware (last signal date = test; earlier dates = train)
* **Regression**: Elastic Net predicts `alpha_decay` (bps)
* **Classification**: Logistic Regression with a tuned probability threshold
* **Outputs**: scatter + ROC plots, console metrics, and feature importance

---

## Quick glossary

* **Alpha**: expected edge (excess return) from a signal over a short horizon.
* **Alpha decay**: how fast that edge fades after the signal and as you trade.
* **bps**: basis points (1 bps = 0.01%).
* **Label**: the value a model tries to predict (e.g., alpha\_decay).
* **Feature**: an input variable (e.g., spread, imbalance, urgency).
* **ROC-AUC**: 1.0 is perfect, 0.5 is random; measures classification discrimination.
* **Precision/Recall**: how accurate positive flags are, and how many true positives you find.
* **SNR (signal-to-noise ratio)**: how much useful variation exists vs. noise.

---


## Research framing: cost vs. preservation

* **Cost minimization (classic)**: reduce implementation shortfall / slippage vs. benchmarks (arrival, VWAP).
* **Signal preservation (alpha-aware)**: for short-lived signals, prioritize **capturing predicted alpha before it decays**, even if that means accepting higher impact.

This project helps quantify and model decay so execution decisions can trade off **cost** vs. **timing** with eyes open.

---

## Optional: LLM summaries for cards

You can add short LLM-generated blurbs to each card (otherwise the project uses simple rule-based text):

```bash
pip install openai
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini   # optional
export LLM_ENABLE=1               # enable LLM summaries
python -m src.run_demo
```

---

## Known issues & fixes (short)

* **Small test set / high-variance metrics**
  *Fix:* simulate more days/assets, use time-blocked CV, report confidence intervals.

* **Low SNR for the continuous target (near-constant regression predictions)**
  *Fix:* add feature interactions (e.g., `urgency×spread`, `imbalance×vol`), widen regularization grid, and log `std(y_pred)` vs `std(y_true)` to detect collapse.

* **Scaling/join pitfalls**
  *Fix:* ensure per-asset, time-sorted `merge_asof` joins; fit scalers on **train only**.

* **Probability calibration**
  *Fix:* apply Platt/Isotonic calibration on a validation fold and tune thresholds for your objective.

---

## Troubleshooting

* **SQLite ingest errors** → re-run `python -m src.simulate_data` to regenerate CSVs.
* **`merge_asof` sorting** → handled per-asset in `features.py`; ensure input CSVs are fresh.
* **Classification shows 0/0 Precision/Recall** → use the tuned threshold printed to console (don’t default to 0.5).
* **Flat predictions / empty plots** → regenerate data and check feature importance + prediction std.

---
