# Alpha Decay Demo — Project Guide

This guide walks you through the demo end-to-end: what it does, how to run it, what each module produces, and how to share it on GitHub.

## What this demo does

A minimal, reproducible workflow to study alpha decay around order execution:

- Simulate small but realistic datasets (signals, market bars, orders, child fills)
- Build a local SQLite database
- Compute labels and features
- Train models (regression + classification)
- Score the test set and produce human-readable “explanation cards”

Everything is reproducible, Python-only, and runs in a few minutes on a laptop.

## Quick start

1. Create a virtual environment and install deps

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the full pipeline

```bash
python -m src.run_demo
```

3. Inspect outputs

- SQLite database: `db/alpha.sqlite`
- Data tables: `data/*.csv`, `data/labels.parquet`, `data/features.parquet`
- Plots: `data/regression_scatter.png`, `data/roc_curve.png`
- Explanation cards: `data/explanations.jsonl` (one card per line)

Optional: Open the notebook `notebooks/demo.ipynb` and run all cells.

## Repository structure

```
alpha_decay_demo/
  README.md
  requirements.txt
  .gitignore
  data/                # CSVs, parquet outputs, plots
  db/                  # SQLite database
  src/
    simulate_data.py   # Simulate signals, market, orders, child fills → CSVs
    ingest.py          # Create SQLite, create tables, bulk insert → db/alpha.sqlite
    label.py           # Compute alpha at signal vs exec, alpha_decay, flags → labels.parquet
    features.py        # Parent-level features from microstructure + footprint → features.parquet
    train.py           # Train ElasticNetCV + LogisticRegressionCV; plots & artifacts
    predict_explain.py # Score test set; produce explanation cards (JSONL)
    run_demo.py        # Orchestrator — one command runs everything
    utils.py           # (Reserved) shared helpers
  notebooks/
    demo.ipynb         # Optional companion to explore outputs
  docs/
    Guide.md           # This guide
    Data_Schema.md     # Detailed schema of CSVs and SQLite tables
```

## Data flow (high-level)

1. `simulate_data.py` → writes:
   - `data/signals.csv`, `data/orders.csv`, `data/child_fills.csv`, `data/market.csv`
2. `ingest.py` → `db/alpha.sqlite` with tables and indexes
3. `label.py` → `data/labels.parquet`
4. `features.py` → `data/features.parquet`
5. `train.py` → trains, prints metrics, saves plots and models
6. `predict_explain.py` → `data/explanations.jsonl` and prints samples

## Schemas (summary)

See `docs/Data_Schema.md` for the full column lists and definitions. Key tables:

- `signals(ts_signal, asset, side, signal_score, alpha_horizon_min, signal_strength_rank)`
- `orders(parent_id, ts_arrival, asset, side, qty, urgency_tag, algo_type, participation_cap, broker, venue_hint)`
- `child_fills(parent_id, ts, price, qty, venue, order_type)`
- `market(ts, asset, mid, bid, ask, spread_bp, depth1_bid, depth1_ask, imbalance, rv_5m, rv_30m, adv, turnover)`

## Modeling and evaluation

- Time-aware split: last `ts_signal` date is test; earlier dates are train
- Regression: ElasticNetCV (scaled) on `alpha_decay` (bps); Ridge fallback if needed
- Classification: LogisticRegressionCV (balanced); threshold tuning via Max-F1 + base-rate
- Plots: regression scatter; ROC curve
- Diagnostics printed to console: MAE, AUC, tuned Precision/Recall, std of regression predictions, top permutation importances

## Explanation cards

Each JSON object includes:

- `parent_id`, `prediction_bps`, risk bucket (quantile-based), `top_drivers` (permutation importance), `suggested_tactics`, `guardrails`
- One card per line in `data/explanations.jsonl`

## Reproducibility

- Fixed seeds where relevant; small dataset (5 assets × 5 days) to run quickly
- Pure Python dependencies: pandas, numpy, scikit-learn, matplotlib, sqlite-utils, pyarrow, joblib
- All outputs written under `data/` and `db/`

## Troubleshooting

- If SQLite errors on ingest: ensure `data/*.csv` exist by re-running `python -m src.simulate_data`
- If merge_asof sorting errors: fixed in `features.py` via per-asset sorted as-of joins
- If classification shows 0/0 precision/recall: tuned thresholds are printed; use the tuned threshold rather than 0.5 to convert probabilities to labels
- If plots are empty or models look constant: check the printed std of predictions and permutation importances

## How to publish this to GitHub

1. Create a new GitHub repository (no need to add a README — you already have one)
2. In your local terminal, from the project directory `alpha_decay_demo/`:

```bash
# Initialize git (one-time)
cd /Users/shats/Downloads/AQR\ Prep/Alpha\ Decay/alpha_decay_demo
git init

git add -A
git commit -m "Alpha Decay demo: initial commit"

# Rename default branch to main
git branch -M main

# Add your GitHub remote (replace with your repo URL)
# For HTTPS:
#   git remote add origin https://github.com/<your-user>/alpha_decay_demo.git
# For SSH:
#   git remote add origin git@github.com:<your-user>/alpha_decay_demo.git

git push -u origin main
```

3. Verify on GitHub: README renders, directories visible

### What not to commit

- Your virtual environment `.venv/`
- Large generated files if you want a clean repo (you can keep CSVs small). The `.gitignore` already ignores venvs and common Python artifacts. If you don’t want to commit outputs, uncomment `data/` in `.gitignore`.

### Keeping the project sharable

- Keep `README.md` short and point to `docs/Guide.md`
- Ensure `requirements.txt` installs cleanly in a new environment
- Keep all scripts runnable via the `-m` entrypoints

## Extending the demo

- Replace simulators with real data loaders
- Add additional features (venue, queue estimates, realized slippage vs benchmarks)
- Add backtest harness and cost curves
- Add monitoring notebook for model drift and retraining cadence
