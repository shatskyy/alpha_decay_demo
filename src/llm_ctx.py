"""Optional LLM contextualization for explanation cards.

Usage (optional; off by default):
- Set env var LLM_ENABLE=1
- Install the OpenAI SDK: `pip install openai`
- Set OPENAI_API_KEY in your environment
- Optionally set OPENAI_MODEL (default: gpt-4o-mini)

If not enabled or dependencies are missing, functions return None and the
pipeline continues with rule-based content only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import os


@dataclass(frozen=True)
class LLMConfig:
	enabled: bool
	model: str


def _get_config() -> LLMConfig:
	enabled = os.environ.get("LLM_ENABLE", "0") == "1"
	model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
	return LLMConfig(enabled=enabled, model=model)


def maybe_llm_summarize_card(
	card: Dict[str, Any],
	metrics: Dict[str, Any],
	context: Dict[str, Any],
) -> Optional[str]:
	cfg = _get_config()
	if not cfg.enabled:
		return None
	try:
		from openai import OpenAI  # type: ignore
		openai_client = OpenAI()
	except Exception:
		return None

	# Compose a concise expert narrative prompt
	prediction_bps = card.get("prediction_bps")
	risk_bucket = card.get("risk_bucket")
	drivers = card.get("top_drivers", [])
	guardrails = card.get("guardrails", [])
	asset = context.get("asset")
	ts_signal = context.get("ts_signal")
	mae_bps = metrics.get("reg_mae_bps")
	roc_auc = metrics.get("clf_roc_auc")

	drivers_str = ", ".join(
		f"{d.get('feature')}: {d.get('sign','?')} ({abs(float(d.get('importance', 0.0))):.4f})" for d in drivers
	)
	guards_str = "; ".join(str(g) for g in guardrails)

	system = (
		"You are an expert execution trader. Write a single concise paragraph (no bullets) "
		"that interprets predicted alpha decay for one parent order, references model performance, "
		"and connects to the big picture of execution tactics and risk. Be concrete, trader-friendly, and balanced."
	)
	user = (
		f"Asset: {asset}; Signal time: {ts_signal}. "
		f"Predicted decay: {prediction_bps:.2f} bps; Risk bucket: {risk_bucket}. "
		f"Top drivers: {drivers_str}. Guardrails: {guards_str if guards_str else 'n/a'}. "
		f"Model performance on test — Regression MAE: {mae_bps:.2f} bps; Classifier ROC-AUC: {roc_auc:.3f}. "
		"Write one paragraph (3–5 sentences). Avoid jargon, avoid overpromising, acknowledge uncertainty."
	)

	try:
		resp = openai_client.chat.completions.create(
			model=cfg.model,
			messages=[
				{"role": "system", "content": system},
				{"role": "user", "content": user},
			],
			temperature=0.4,
			max_tokens=160,
		)
		text = resp.choices[0].message.content.strip() if resp and resp.choices else None
		return text or None
	except Exception:
		return None