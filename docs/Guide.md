# Alpha Decay Demo — Project Guide

This guide walks you through the demo end-to-end: what it does, how to run it, and what each module produces.

---

## Why this matters

Signal strength tends to decay quickly once orders hit the market. For execution traders, this decay drives slippage, timing risk, and opportunity cost.
This demo is a **minimal, reproducible sandbox** for studying alpha decay around order execution. It shows how to:

* Generate synthetic but realistic data (signals, orders, fills, market bars)
* Build a database that joins orders, fills, and market context
* Compute alpha decay labels
* Engineer microstructure-based features
* Train baseline models (regression + classification)
* Produce **intepretable explanations** that highlight what drove outcomes

⚠️All data is simulated.

Everything runs in less than a minute on a laptop with pure Python dependencies.

---

## Quick start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the full pipeline:

```bash
python -m src.run_demo
```

3. Inspect outputs:

* Database: `db/alpha.sqlite`
* Data tables: `data/*.csv`, `data/labels.parquet`, `data/features.parquet`
* Plots: `data/regression_scatter.png`, `data/roc_curve.png`
* Explanation cards: `data/explanations.jsonl`

Optional: open the notebook `notebooks/demo.ipynb` for interactive exploration.

---

## Repository structure

```
alpha_decay_demo/
  README.md
  requirements.txt
  src/                 # Core scripts
    simulate_data.py
    ingest.py
    label.py
    features.py
    train.py
    predict_explain.py
    run_demo.py
  data/                # Outputs: CSVs, parquet, plots
  db/                  # SQLite database
  notebooks/           # Optional demo notebook
  docs/
    Guide.md           # This guide
    Data_Schema.md     # Detailed schema reference
```

---

## Data flow

1. `simulate_data.py` → signals, market bars, parent orders, child fills (`data/*.csv`)
2. `ingest.py` → bulk insert into `db/alpha.sqlite`
3. `label.py` → compute alpha decay labels (`data/labels.parquet`)
4. `features.py` → parent-level features from order footprint + market context (`data/features.parquet`)
5. `train.py` → fit regression & classification models, save plots and metrics
6. `predict_explain.py` → generate explanation cards (`data/explanations.jsonl`)

---

## Key schemas

Full column definitions are in `docs/Data_Schema.md`. Highlights:

* **signals**: signal timestamp, asset, side, score, horizon, rank
* **orders**: parent ID, arrival time, side, qty, urgency, algo type, caps, broker, venue hints
* **child\_fills**: parent ID, timestamp, price, qty, venue, order type
* **market**: mid, bid/ask, spread (bps), depth, imbalance, realized vol, ADV, turnover

---

## Modeling and evaluation

* **Train/test split**: last signal date = test, earlier = train
* **Regression**: ElasticNetCV (scaled), predicting alpha decay (bps)
* **Classification**: LogisticRegressionCV, tuned threshold (Max-F1 + base rate)
* **Outputs**: regression scatter, ROC curve, MAE/AUC/Precision/Recall printed to console
* **Diagnostics**: std of regression predictions, top permutation importances

---

## Explanation cards

Each card summarizes prediction + drivers for one parent order.
Example:

```json
{
  "parent_id": "ORD123",
  "prediction_bps": -5.7,
  "risk_bucket": "High decay",
  "top_drivers": ["spread_bp", "urgency_tag", "imbalance"],
  "suggested_tactics": "Slice smaller, avoid top-of-book",
  "guardrails": "Do not exceed 10% ADV"
}
```

* One card per line in `data/explanations.jsonl`
* Useful for **interpreting model outputs** and framing trader tactics/constraints

---

## Questions to explore

This framework provides a safe sandbox to explore real execution problems:

* How quickly does signal strength decay once orders arrive?
* Which microstructure features (spread, imbalance, volatility) drive decay most?
* How stable are decay patterns across assets or time periods?
* How could these insights inform pre-trade estimates or TCA extensions?

---

## Reproducibility

* Fixed seeds where relevant
* Dataset: 5 assets × 5 days → runs in minutes
* Dependencies: pandas, numpy, scikit-learn, matplotlib, sqlite-utils, pyarrow, joblib
* All outputs reproducible under `data/` and `db/`

---

## Troubleshooting

* **SQLite ingest errors** → re-run `python -m src.simulate_data` to regenerate CSVs
* **merge\_asof sorting errors** → fixed in `features.py` (per-asset sorted joins)
* **Classification precision/recall = 0** → use the tuned threshold printed, not 0.5
* **Empty plots or flat predictions** → check prediction std and permutation importances
