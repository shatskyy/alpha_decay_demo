# Alpha Decay Demo (Python)

A minimal, modular Python demo that simulates market and execution data, builds a SQLite database, computes labels and features, trains a scikit-learn model, and analyzes alpha decay with simple explanations.

## Overview

- Simulated datasets for 5–10 equities, 5 trading days, and ~10–20k child fills
- Storage in CSVs (`data/`) and a SQLite DB (`db/`)
- Deterministic runs via fixed random seeds
- Modular scripts in `src/` and a single entrypoint to run end-to-end

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run end-to-end

```bash
python -m src.run_demo
```

## Outputs

- SQLite DB at `db/alpha.sqlite`
- Plots at `data/*.png`
- Explanation cards at `data/explanations.jsonl`
- Metrics printed to console

## Notes

- Only standard scientific Python stack is used: pandas, numpy, scikit-learn, matplotlib, SQLite
- No external services or web apps are required
- The “LLM explanation” is rule-based in this demo to avoid external dependencies, but is API-pluggable.
- This repository is for demonstration purposes only; data are simulated
