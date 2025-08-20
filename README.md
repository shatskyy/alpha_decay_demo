# Alpha Decay Demo

Institutional execution framed as signal preservation. This system quantifies how trading signals degrade during execution and provides actionable intelligence to optimize the signal preservation vs. cost trade-off.

---

## Executive summary

- **What it is**: A fast, reproducible demo that models and explains parent-order alpha decay from arrival-time microstructure and order parameters.
- **Current Functionality**:
  - Simulates signals, orders, child fills, and minute bars (synthetic but realistic).
  - Computes parent-level alpha-decay labels and builds arrival-only features.
  - Trains regression and classification baselines; saves plots and metrics.
  - Outputs concise per-parent “explanation cards” with predicted decay, drivers, and suggested tactics.

This project aims to provide a framework for ececution traders who want to move beyond pure cost minimization toward preserving predicted alpha. Execution decisions (urgency, participation, venue mix) can either protect or destroy alpha. Measuring, predicting, and explaining decay is the foundation for adaptive, alpha-aware execution.

---

## Introduction: Alpha decay and signal preservation

In systematic trading, a signal’s predictive edge decays quickly. Execution must be understood as preserving that edge, along with minimzing the cost.

$$
\text{Realized Alpha} = \text{Predicted Alpha} - (\text{slippage} + \text{market impact} + \text{signal decay})
$$

---

## Solution Architecture

### 1. Signal-Centric Data Model
- **Causal Design**: Trading signals are embedded directly into synthetic market data with realistic decay patterns
- **Direction-Aware Metrics**: Alpha decay calculations properly account for long/short signal directionality
- **Multi-Horizon Analysis**: Captures decay patterns across different prediction horizons (1min, 5min, 15min, 1hr)

### 2. Advanced Feature Engineering
- **Signal Integration**: Includes signal score and strength as modeling features
- **Microstructure Context**: Spread, imbalance, depth, and volatility measures
- **Regime Detection**: Basic volatility and time-of-day regime indicatorsios


### 3. Enhanced Modeling Framework
- **Direction-Aware Targets**: Properly accounts for signal directionality in alpha decay calculation
- **Monotonic Constraints**: Gradient boosting with constraints on key economic relationships
- **Multiple Target Variants**: Raw, volatility-adjusted, and signal-weighted decay measures

### 4. Institutional Decision Support
- **Risk Bucketing**: Quantile-based classification of decay risk scenarios
- **Tactical Recommendations**: Context-aware suggestions for participation rates, venue selection, and order types
- **Guardrails**: Automated risk controls based on market volatility and execution constraints

---

## Data Architecture & Methodology

### Synthetic Data with Realistic Properties
- **Signal Embedding**: Controlled signal-to-return correlation with realistic decay patterns
- **Microstructure Simulation**: Bid-ask bounce, intraday seasonality, volume patterns
- **Execution Modeling**: Order flow with realistic market impact and temporary/permanent price effects

### Time-Aware Validation
- **No Lookahead Bias**: Strict temporal splits with last day as test set
- **Rolling Origin Evaluation**: Day-by-day validation to assess temporal stability
- **Regime-Aware Metrics**: Performance tracking across different market conditions

### Feature Categories
1. **Arrival Context**: Spread, imbalance, realized volatility, depth, time-of-day
2. **Signal Intelligence**: Score, strength rank, confidence measures
3. **Policy Parameters**: Participation caps, urgency levels, venue preferences
4. **Regime Indicators**: Volatility regimes, open/close periods, spread conditions

---

## Roadmap: AI interpretation and market context (goals)

Integrate AI to interpret every significant output and contextualize risk with current market conditions:

- **Per-parent narratives**: LLM augments cards with concise, trader-friendly explanations citing metrics and uncertainty.
- **Per-asset and day summaries**: roll-up of drivers, regime performance, and what-if scenarios.
- **Market context**: optional feeds (index returns, vol, economic/earnings calendars) to flag relevant shifts that inform execution urgency.
- **Shareable report**: generate a daily HTML/Markdown brief combining diagnostics, context, and cards.

All AI features are opt-in and degrade gracefully to rule-based text when disabled.

---
## Business Applications

### For Execution Traders
- **Pre-Trade Analytics**: Risk assessment before order submission
- **Real-Time Guidance**: Dynamic recommendations during execution
- **Post-Trade Analysis**: Performance attribution including signal preservation metrics

### For Portfolio Managers
- **Strategy Optimization**: Understanding how execution choices affect alpha capture
- **Risk Management**: Identifying scenarios where signal decay risk is elevated
- **Performance Attribution**: Separating execution skill from signal decay effects

### For Quantitative Researchers
- **Algorithm Development**: Framework for testing signal preservation strategies
- **Market Microstructure Research**: Understanding decay patterns across different conditions
- **Model Validation**: Comprehensive backtesting infrastructure with realistic execution simulation

---

## Implementation & Deployment

### Quick Start
```bash
# Setup environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run complete pipeline
python3 -m src.run_demo

# Enhanced mode with LLM explanations
export OPENAI_API_KEY=sk-...
export LLM_ENABLE=1
python3 -m src.run_demo
```

### Modular Execution
```bash
# Individual pipeline components
python -m src.simulate_data    # Generate synthetic data
python -m src.ingest          # Build database
python -m src.label           # Compute alpha decay metrics
python -m src.features        # Engineer features
python -m src.train           # Train ensemble models
python -m src.predict_explain # Generate explanation cards
```
---

## Key Outputs & Deliverables

### 1. Explanation Cards (JSON)
```json
{
  "parent_id": "ORD_2024_001",
  "prediction_bps": -8.3,
  "confidence": 0.73,
  "risk_bucket": "HIGH",
  "top_drivers": [
    {"feature": "signal_score", "importance": 0.034, "sign": "+"},
    {"feature": "spread_bp", "importance": 0.021, "sign": "+"},
    {"feature": "signal_x_urgency", "importance": 0.018, "sign": "+"}
  ],
  "suggested_tactics": [
    "Increase participation to 20-25% given high signal strength",
    "Bias toward dark venues (60%) to reduce market impact",
    "Allow aggressive tactics given signal confidence > 70%"
  ],
  "guardrails": [
    "Do not exceed 25% participation cap",
    "Monitor for regime change - high volatility detected"
  ]
}
```
### 2. Performance Analytics
- **Regression Diagnostics**: Prediction accuracy, residual analysis, feature importance
- **Classification Metrics**: ROC-AUC, precision-recall curves, threshold optimization
- **Economic Significance**: Basis point improvements, Sharpe ratio enhancements

### 3. Visual Diagnostics
- **Prediction Scatter Plots**: Model accuracy visualization
- **ROC Curves**: Classification performance assessment
- **Feature Importance**: Driver identification and ranking

---

## Technical Specifications

### System Requirements
- **Python**: 3.9+
- **Dependencies**: scikit-learn, pandas, numpy, matplotlib, joblib
- **Database**: SQLite (research) / PostgreSQL (production)
- **Compute**: Standard laptop sufficient for demo; production requires dedicated infrastructure

### Data Volumes (Demo)
- **Signals**: ~150 observations across 5 assets, 5 days
- **Orders**: 1:1 with signals
- **Child Fills**: ~10,000 execution records
- **Market Data**: ~9,750 minute bars

---

## Risk Disclaimers & Limitations

⚠️ **Research Framework**: This is a proof-of-concept using synthetic data. Production deployment requires extensive validation with real market data.

### Key Limitations
1. **Synthetic Data**: Relationships may not reflect real market dynamics
2. **Model Assumptions**: Linear and tree-based models may miss complex patterns
3. **Regime Stability**: Performance may degrade during market stress periods
4. **Execution Constraints**: Real-world execution faces additional constraints not modeled

---

## Repository structure

```
alpha_decay_demo/
  README.md
  requirements.txt
  src/
    simulate_data.py     # signals, market bars, parent orders, child fills → CSVs
    ingest.py            # build SQLite and load CSVs → db/alpha.sqlite
    label.py             # compute alpha_decay/flags → parquet
    features.py          # arrival-time features and regime flags
    train.py             # train/evaluate; save models/plots/metrics
    predict_explain.py   # score test, generate explanation cards (JSONL)
    what_if.py           # predictive and structural scenario analysis
    run_demo.py          # orchestrator: run the full pipeline
  data/                  # CSVs, parquet outputs, plots, explanations.jsonl
  db/                    # SQLite database
  docs/
    Data_Schema.md       # column definitions for CSVs & SQLite tables
    examples/            # example plots and sample cards
```
---
