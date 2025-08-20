"""Predict and explain alpha-decay using the trained models.

- Loads trained artifacts and `data/features.parquet`
- Filters to test set (last `ts_signal` date)
- Scores regression and classification models
- Builds per-parent "Explanation Card" dicts with risk buckets, top drivers, suggested tactics, guardrails, and optional LLM summary
- Saves JSONL to `data/explanations.jsonl` and prints 5 sample cards

CLI:
	python -m src.predict_explain
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from . import llm_ctx


# ---------------------------- Paths -----------------------------------------

@dataclass(frozen=True)
class Paths:
	project_root: Path
	data_dir: Path
	db_dir: Path
	db_path: Path


def get_paths() -> Paths:
	src_dir = Path(__file__).resolve().parent
	project_root = src_dir.parent
	data_dir = project_root / "data"
	db_dir = project_root / "db"
	db_path = db_dir / "alpha.sqlite"
	data_dir.mkdir(parents=True, exist_ok=True)
	return Paths(project_root=project_root, data_dir=data_dir, db_dir=db_dir, db_path=db_path)


# ---------------------------- Loading ---------------------------------------

def _load_features(paths: Paths) -> pd.DataFrame:
	feat_path = paths.data_dir / "features.parquet"
	if not feat_path.exists():
		raise FileNotFoundError("features.parquet not found. Run the pipeline up to features.")
	df = pd.read_parquet(feat_path)
	df["ts_signal"] = pd.to_datetime(df["ts_signal"])  # ensure datetime
	return df


def _time_split(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
	dates = df["ts_signal"].dt.normalize().sort_values().unique()
	last_day = dates[-1]
	df_test = df.loc[df["ts_signal"].dt.normalize() == last_day].copy()
	df_train = df.loc[df["ts_signal"].dt.normalize() < last_day].copy()
	return df_train, df_test


def _load_artifacts(paths: Paths) -> Dict[str, object]:
	reg = joblib.load(paths.data_dir / "model_reg.pkl")
	clf = joblib.load(paths.data_dir / "model_clf.pkl")
	with (paths.data_dir / "feature_cols.json").open("r") as f:
		feature_cols: List[str] = json.load(f)
	return {"reg": reg, "clf": clf, "feature_cols": feature_cols}


# ---------------------------- Explanations ----------------------------------

def _risk_bucket_quantile(df_test: pd.DataFrame) -> pd.Series:
	# Quantile-based buckets on predicted bps
	q40 = df_test["pred_bps"].quantile(0.40)
	q75 = df_test["pred_bps"].quantile(0.75)
	def bucket(v: float) -> str:
		if v <= q40:
			return "LOW"
		if v <= q75:
			return "MED"
		return "HIGH"
	return df_test["pred_bps"].apply(bucket)


def _canonical_questions() -> List[str]:
	return [
		"What alpha will we preserve over the next horizon?",
		"How should we adjust participation/urgency/dark share to improve capture?",
		"What is the expected uplift vs a simple POV benchmark?",
	]


def _load_intervals(paths: Paths) -> pd.DataFrame:
	pi_path = paths.data_dir / "reg_intervals.parquet"
	if pi_path.exists():
		try:
			return pd.read_parquet(pi_path)
		except Exception:
			return pd.DataFrame()
	return pd.DataFrame()


def _permutation_drivers(reg_pipeline, X_test: pd.DataFrame, y_test: pd.Series, feature_cols: List[str], top_k: int = 3, random_state: int = 7) -> List[Dict[str, float]]:
	perm = permutation_importance(reg_pipeline, X_test, y_test, n_repeats=10, random_state=7, n_jobs=1)
	pairs = sorted(zip(feature_cols, perm.importances_mean), key=lambda x: abs(x[1]), reverse=True)[:top_k]
	drivers = [
		{"feature": name, "importance": float(val), "sign": "?"}
		for name, val in pairs
	]
	return drivers


def _attach_sign_via_spearman(drivers: List[Dict[str, float]], X_train: pd.DataFrame, y_train: pd.Series) -> List[Dict[str, float]]:
	# Spearman sign via rank correlation (no SciPy): corr of ranks
	y_rank = pd.Series(y_train).rank(method="average")
	for d in drivers:
		name = d["feature"]
		if name in X_train.columns:
			x_rank = pd.Series(X_train[name]).rank(method="average")
			corr = np.corrcoef(x_rank, y_rank)[0, 1]
			d["sign"] = "+" if (corr or 0) >= 0 else "-"
	return drivers


def _policy_levers(df_row: pd.Series) -> List[Dict[str, str]]:
	# Identify 4–6 policy-aware levers and whether they are arrival-only
	levers = []
	def add(name: str, val, arrival_only: bool):
		levers.append({"name": name, "value": str(val), "scope": "arrival-only" if arrival_only else "post-trade"})
	for nm in ["participation_cap", "urgency_tag", "pct_dark", "pct_marketable", "imbalance", "open_close_bucket"]:
		if nm in df_row.index:
			add(nm, df_row.get(nm), nm in ["participation_cap", "urgency_tag", "imbalance", "open_close_bucket"])
	return levers


def _benchmarks(df_test: pd.DataFrame) -> pd.Series:
	# Simple spread-only and POV-15% heuristic baselines in bps
	spread_only = df_test.get("spread_bp", pd.Series(0.0, index=df_test.index)).astype(float)
	# POV baseline proxy: proportional to participation_cap * spread
	pov15 = 0.15 * df_test.get("spread_bp", pd.Series(0.0, index=df_test.index)).astype(float)
	return pd.Series(spread_only, index=df_test.index), pd.Series(pov15, index=df_test.index)


def _confidence_score(row: pd.Series) -> float:
	# Combine interval width and regime flags into a simple 0-1 score
	q10, q90 = row.get("q10", np.nan), row.get("q90", np.nan)
	width = float(q90 - q10) if (pd.notna(q10) and pd.notna(q90)) else np.nan
	base = 0.7 if pd.isna(width) else max(0.1, 1.0 - (abs(width) / 60.0))
	pen = 0.1 * int(row.get("regime_high_vol", 0)) + 0.1 * int(row.get("regime_open", 0))
	return float(np.clip(base - pen, 0.0, 1.0))


def generate_explanations() -> Path:
	paths = get_paths()
	df_all = _load_features(paths)
	df_train, df_test = _time_split(df_all)

	art = _load_artifacts(paths)
	reg = art["reg"]
	clf = art["clf"]
	feature_cols: List[str] = art["feature_cols"]

	# Align columns and ensure presence
	for col in feature_cols:
		if col not in df_test.columns:
			df_test[col] = 0.0
		if col not in df_train.columns:
			df_train[col] = 0.0
	X_test = df_test[feature_cols].copy()
	X_train = df_train[feature_cols].copy()
	y_test = df_test["alpha_decay"].astype(float)
	y_train = df_train["alpha_decay"].astype(float)

	# Score
	pred_bps = reg.predict(X_test)
	df_test["pred_bps"] = pred_bps
	# Attach intervals if available
	intervals = _load_intervals(paths)
	if not intervals.empty:
		intervals = intervals.set_index("parent_id")
		for col in ["q10", "q50", "q90"]:
			if col in intervals.columns:
				df_test[col] = df_test["parent_id"].map(intervals[col])

	# Baselines
	spread_only, pov15 = _benchmarks(df_test)
	df_test["spread_only_bps"] = spread_only
	df_test["pov15_bps"] = pov15

	# Quantile risk buckets on test predictions
	df_test["risk_bucket"] = _risk_bucket_quantile(df_test)

	# Drivers: permutation importance on test; sign via Spearman on train
	drivers = _permutation_drivers(reg, X_test, y_test, feature_cols, top_k=3)
	drivers = _attach_sign_via_spearman(drivers, X_train, y_train)

	# Load participation_cap from orders for guardrail context
	orders_csv = paths.data_dir / "orders.csv"
	pcaps = None
	if orders_csv.exists():
		pcaps = pd.read_csv(orders_csv, usecols=["parent_id", "participation_cap"])
		pcaps = pcaps.set_index("parent_id")

	# 95th percentile of rv_30m on test for guardrail
	rv95 = float(np.nanpercentile(df_test.get("rv_30m", pd.Series(dtype=float)), 95)) if "rv_30m" in df_test.columns and len(df_test) else float("inf")

	# Gather top-level metrics for LLM context
	metrics = {}
	metrics_path = paths.data_dir / "regression_scatter.png"  # dummy trigger that train ran
	try:
		# Pull from train artifacts if needed later; here we pass placeholders
		from sklearn.metrics import roc_auc_score  # noqa: F401
		df_all_test = df_test
		# We can't recompute MAE/AUC cheaply here without reloading models; omit
		metrics = {"reg_mae_bps": np.nan, "clf_roc_auc": np.nan}
	except Exception:
		metrics = {"reg_mae_bps": np.nan, "clf_roc_auc": np.nan}

	# Build cards
	cards: List[Dict] = []
	for idx, row in df_test.iterrows():
		pid = row["parent_id"]
		bucket = row["risk_bucket"]
		row_local = row.copy()
		if pcaps is not None and pid in pcaps.index:
			row_local["participation_cap"] = pcaps.loc[pid, "participation_cap"]
		suggestions = []
		# Use original simple heuristic but with quantile bucket
		hta = float(row.get("horizon_to_age", np.nan)) if pd.notna(row.get("horizon_to_age", np.nan)) else np.inf
		spread_small = (row.get("spread_bp", np.inf) <= 1.5)
		if bucket == "HIGH" and hta < 2.0:
			suggestions.append("Increase POV to ~18–25%, bias DARK ~60%, allow marketable in small clips")
		elif bucket == "LOW" and spread_small:
			suggestions.append("Prefer passive/discretionary tactics; POV ~5–10% with patient posting")
		else:
			suggestions.append("Balanced POV ~10–15%, mix LIT/DARK, throttle marketable flow")

		guards: List[str] = []
		if float(row.get("rv_30m", 0.0)) > rv95:
			guards.append("Avoid aggressive tactics when short-horizon vol is elevated (rv_30m > 95th pctile)")
		pcap = row.get("participation_cap", None)
		if pd.notna(pcap):
			guards.append(f"Do not exceed participation_cap ({int(pcap)}%)")
		else:
			guards.append("Do not exceed participation_cap")
		# Interval-aware caution
		if not np.isnan(row.get("q90", np.nan)) and not np.isnan(row.get("q10", np.nan)):
			width = float(row.get("q90")) - float(row.get("q10"))
			if width > 8.0:  # wide uncertainty band
				guards.append("High uncertainty: prefer robust tactics; avoid overfitting to point estimate")

		card = {
			"parent_id": pid,
			"prediction_bps": float(row["pred_bps"]),
			"interval_bps": {"q10": float(row.get("q10", np.nan)), "q50": float(row.get("q50", np.nan)), "q90": float(row.get("q90", np.nan))},
			"risk_bucket": bucket,
			"top_drivers": drivers,
			"policy_levers": _policy_levers(row),
			"questions": _canonical_questions(),
			"benchmarks": {"spread_only_bps": float(row.get("spread_only_bps", np.nan)), "pov15_bps": float(row.get("pov15_bps", np.nan)), "uplift_vs_pov": float(row.get("pred_bps", np.nan) - row.get("pov15_bps", np.nan))},
			"confidence": _confidence_score(row),
			"suggested_tactics": suggestions,
			"guardrails": guards,
		}

		# Optional LLM summary
		try:
			llm_text = llm_ctx.maybe_llm_summarize_card(card, metrics, {"asset": row.get("asset"), "ts_signal": row.get("ts_signal")})
			if llm_text:
				card["llm_summary"] = llm_text
		except Exception:
			pass

		cards.append(card)

	# Save JSONL
	out_path = paths.data_dir / "explanations.jsonl"
	with out_path.open("w") as f:
		for card in cards:
			f.write(json.dumps(card) + "\n")

	# Print 5 samples
	for sample in cards[:5]:
		print(json.dumps(sample, indent=2))

	return out_path


if __name__ == "__main__":
	generate_explanations()