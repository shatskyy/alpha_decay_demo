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

## Optional: LLM contextualization (off by default)

You can optionally attach an expert-style paragraph to each explanation card using OpenAI. This is disabled by default.

1. Install the SDK (optional dependency):

```bash
pip install openai
```

2. Set environment variables (do NOT commit keys):

```bash
export OPENAI_API_KEY="<your-key>"
export OPENAI_MODEL="gpt-4o-mini"   # optional; default is gpt-4o-mini
export LLM_ENABLE=1                  # enable LLM summaries
```

3. Re-run the demo:

```bash
python -m src.run_demo
```

Cards will then include an `llm_summary` field when available. If keys or the package are missing, the pipeline silently falls back to rule-based text.

## Outputs

- SQLite DB at `db/alpha.sqlite`
- Plots at `data/*.png`
- Explanation cards at `data/explanations.jsonl`
- Metrics printed to console

## Notes

- Only standard scientific Python stack is used: pandas, numpy, scikit-learn, matplotlib, SQLite
- No external services or web apps are required
- This repository is for demonstration purposes only; data are simulated
