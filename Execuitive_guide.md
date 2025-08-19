# Alpha Decay Demo — Executive Guide

---

## Introduction: Alpha Decay and Signal Preservation in Execution

In trading, **alpha** represents the excess return predicted by a signal beyond a benchmark. The challenge is that most signals are **perishable**, meaning that their predictive power diminishes after generation, otherwise known as **alpha decay**.

One of the goals of execution in insitutional trading is **cost minimization**. The aim is to reduce implementation shortfall or slippage relative to benchmarks such as arrival price or VWAP.

In **systematic trading**, a signal has a limited lifespan—sometimes measured in minutes or seconds. Slow or overly cost-conscious execution can destroy the very alpha the signal was meant to capture. In systematic trading execution must be understood as **signal preservation**, or maximizing how much of the forecasted alpha survives through the trading process into realized P\&L.

$$
\text{Realized Alpha} = \text{Predicted Alpha} - (\text{slippage} + \text{market impact} + \text{signal decay})
$$

Where signals are short-lived and scale matters, the ability to preserve alpha through execution is critical. Execution algorithms, routing logic, and urgency decisions are all factors in the signal preservation problem.

---

### This Project in Context

The **Alpha Decay Demo** is a minimal, reproducible sandbox designed to illustrate this principle. It simulates signals, orders, fills, and market data, then:

* Computes **alpha decay labels** at the parent-order level
* Builds **features** from order characteristics and microstructure
* Trains baseline models to predict decay risk
* Produces **explanation cards** that translate model outputs into trader-friendly insights and tactics

By walking through this workflow, the demo shows how execution research can evolve from pure cost minimization toward a **signal preservation framework**—bridging research signals, execution tactics, and realized alpha.

## What is this?

**A lightweight research sandbox** that shows how signals decay once orders hit the market.
It demonstrates how alpha-aware execution can be measured, modeled, and explained in a fully reproducible way.

* **Problem traders face**: signals decay quickly after order arrival, driving slippage and timing risk.
* **What this demo does**: simulates orders + fills, computes decay labels, trains models, and produces transparent “explanation cards” that reveal what drove outcomes.
* **Why it matters**: it frames execution as a **signal preservation problem**.

⚠️ Data is simulated and synthetic

---

## In this Demo:

* **End-to-end workflow**:
  signal → parent order → child fills → market context → labels → features → models → explanations
* **Database**: all orders, fills, and market bars joined in one SQLite file
* **Models**: regression & classification predicting signal decay in bps
* **Visuals**: regression scatter, ROC curve
* **Explanation Cards**: concise summaries of predicted risk and drivers

---

## Example: Explanation Card

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

Traders can immediately see:

* Expected signal decay (-5.7 bps)
* Why (spread, urgency, imbalance)
* Suggested tactics (slicing, participation cap)

---

## Why traders should care

* **Alpha preservation**: identifies when executing aggressively destroys signal value.
* **Microstructure insight**: shows which features (spread, imbalance, volatility) matter most.
* **Transparency**: avoids “black-box” ML; cards translate model outputs into trader-speak.
* **Prototyping**: quick way to test hypotheses about decay before involving full EMS/OMS stack.

---

## Questions it helps answer

* How quickly do signals decay post-execution?
* Which microstructure features best predict decay?
* How should urgency or participation caps change when decay risk is high?
* Can pre-trade analytics / TCA include decay metrics alongside slippage and cost?

---

## How to use it

1. Clone repo, install Python deps, run one command:

   ```bash
   python -m src.run_demo
   ```
2. Outputs in minutes: database, plots, and JSON explanation cards
3. Use the cards to frame **tactics, guardrails, and trade-offs**

---

## Bottom line

This demo is not a production execution algorithm.
It is a **sandbox for research and dialogue**: a way for execution desks to explore alpha decay, visualize its drivers, and think about how to integrate alpha-awareness into real trading strategies and TCA frameworks.

