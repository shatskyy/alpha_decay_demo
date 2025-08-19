# Alpha Decay Demo
---
## Improvements in Progress

Implementing fixes for regression performance issues: 
- Improved target variables
- Temporal feature engineering
- Ensemble models
- Realistic data generation

---

## Introduction: Alpha Decay and Signal Preservation in Execution

In trading, **alpha** represents the excess return predicted by a signal beyond a benchmark. The challenge is that most signals are **perishable**, meaning that their predictive power diminishes after generation, otherwise known as **alpha decay**.

One of the goals of execution in insitutional trading is **cost minimization**. The aim is to reduce implementation shortfall or slippage relative to benchmarks such as arrival price or VWAP.

In systematic trading, a signal has a limited lifespan. Slow or overly cost-conscious execution can destroy the alpha that the signal was meant to capture. In this case, execution must be understood as **signal preservation**, or maximizing how much of the forecasted alpha survives through the trading process into realized P\&L.

$$
\text{Realized Alpha} = \text{Predicted Alpha} - (\text{slippage} + \text{market impact} + \text{signal decay})
$$

Execution algorithms, routing logic, and urgency decisions are all factors in the signal preservation problem.

---

### This Project in Context

The Alpha Decay Demo is a minimal, reproducible sandbox designed to illustrate this principle. It simulates signals, orders, fills, and market data, then:

* Computes alpha decay labels at the parent-order level
* Builds features from order characteristics and microstructure
* Trains baseline models to predict decay risk
* Produces explanation cards that translate model outputs into trader-friendly insights and tactics

By walking through this workflow, the demo shows how execution research can evolve from pure cost minimization toward a **signal preservation framework**—bridging research signals, execution tactics, and realized alpha.

## What is this?

A lightweight research proof of concept that demonstrates how alpha-aware execution can be measured, modeled, and explained.

* **Problem traders face**: signals decay quickly after order arrival, driving slippage and timing risk.
* **What this demo does**: simulates orders + fills, computes decay labels, trains models, and produces transparent “explanation cards” that reveal what drove outcomes.
* **Why it matters**: it frames execution as a signal preservation problem.

⚠️ Data is simulated and synthetic

---

## Questions it helps answer

* How quickly do signals decay post-execution?
* Which microstructure features best predict decay?
* How should urgency or participation caps change when decay risk is high?
* Can pre-trade analytics / TCA include decay metrics alongside slippage and cost?

---

## In this Demo:

* **End-to-end workflow**:
  signal → parent order → child fills → market context → labels → features → models → explanations
* **Database**: all orders, fills, and market bars joined in one SQLite file
* **Models**: regression (work in progress) & classification predicting signal decay in bps
* **Visuals**: regression scatter, ROC curve
* **Explanation Cards**: concise summaries of predicted risk and drivers

---

## Quick start


```bash
# 1) Clone repo
# 2) Create a virtual environment and install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3) Run the full pipeline (simulate → DB → labels → features → train → explain)
python -m src.run_demo
```

Outputs are written to `data/` and `db/`.

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

## What you’ll see (example)

**Console:**

```
Regression MAE (bps) on test: 14.6
Classification ROC-AUC on test: 0.74
Precision/Recall @ tuned threshold: 0.41 / 0.53
Top permutation importances: spread_bp, imbalance, urgency_tag
```

<img src="data/roc_curve.png" width="420">

The classification task (“high decay” vs “not”) shows useful discrimination (AUC \~0.72), even on a small test set. The stepped shape reflects the small sample size, but it suggests features like **spread**, **imbalance**, and **urgency** carry information about decay risk.


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
```

Each card is a plain summary for a parent order: predicted decay (bps), a risk bucket, the top drivers, and suggested tactics/guardrails. Even if the regression is noisy, the cards remain useful for decision support.

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

## Troubleshooting

* **SQLite ingest errors** → re-run `python -m src.simulate_data` to regenerate CSVs.
* **`merge_asof` sorting** → handled per-asset in `features.py`; ensure input CSVs are fresh.
* **Classification shows 0/0 Precision/Recall** → use the tuned threshold printed to console (don’t default to 0.5).
* **Flat predictions / empty plots** → regenerate data and check feature importance + prediction std.

---
