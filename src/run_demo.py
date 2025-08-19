"""Entrypoint to run the full alpha-decay demo end-to-end.

Usage:
	python -m src.run_demo

Steps:
1) Simulate data and write CSVs to `data/`
2) Ingest CSVs into SQLite under `db/`
3) Compute labels and features
4) Train models and produce plots
5) Predict, explain, and generate explanation cards
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score

from . import simulate_data
from . import ingest
from . import label
from . import features
from . import train
from . import predict_explain


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


def _time_test_set(df: pd.DataFrame) -> pd.DataFrame:
	dates = pd.to_datetime(df["ts_signal"]).dt.normalize().sort_values().unique()
	last_day = dates[-1]
	return df.loc[pd.to_datetime(df["ts_signal"]).dt.normalize() == last_day].copy()


def _print_row_counts(paths: Paths) -> None:
	def _count_csv(p: Path) -> int:
		return int(len(pd.read_csv(p))) if p.exists() else 0

	n_signals = _count_csv(paths.data_dir / "signals.csv")
	n_orders = _count_csv(paths.data_dir / "orders.csv")
	n_child = _count_csv(paths.data_dir / "child_fills.csv")
	n_market = _count_csv(paths.data_dir / "market.csv")
	print(f"Row counts | signals={n_signals}, orders={n_orders}, child_fills={n_child}, market={n_market}")


def _print_metrics(paths: Paths) -> Tuple[float, float]:
	# Load features and artifacts; compute metrics on test split
	feat_path = paths.data_dir / "features.parquet"
	if not feat_path.exists():
		raise FileNotFoundError("features.parquet not found. Run training first.")
	df = pd.read_parquet(feat_path)
	df["ts_signal"] = pd.to_datetime(df["ts_signal"])  # ensure datetime
	with (paths.data_dir / "feature_cols.json").open("r") as f:
		feature_cols: List[str] = json.load(f)
	reg = joblib.load(paths.data_dir / "model_reg.pkl")
	clf = joblib.load(paths.data_dir / "model_clf.pkl")

	df_test = _time_test_set(df)
	for col in feature_cols:
		if col not in df_test.columns:
			df_test[col] = 0.0
	X_test = df_test[feature_cols].copy()
	y_reg_true = df_test["alpha_decay"].astype(float)
	y_cls_true = df_test["decay_flag"].astype(int)

	y_reg_pred = reg.predict(X_test)
	mae_bps = float(mean_absolute_error(y_reg_true, y_reg_pred))
	proba = clf.predict_proba(X_test)[:, 1]
	roc_auc = float(roc_auc_score(y_cls_true, proba)) if y_cls_true.nunique() > 1 else float("nan")
	print(f"Metrics | Regression MAE (bps): {mae_bps:.2f}, Classification ROC-AUC: {roc_auc:.3f}")
	return mae_bps, roc_auc


def main() -> None:
	paths = get_paths()
	print("[1/6] Simulating data → CSVs…")
	simulate_data.generate_all()
	_print_row_counts(paths)

	print("[2/6] Building SQLite DB…")
	ingest.build_database()

	print("[3/6] Computing labels…")
	label.build_labels()

	print("[4/6] Building features…")
	features.build_features()

	print("[5/6] Training models and saving plots…")
	reg_path, clf_path = train.train_models()
	mae_bps, roc_auc = _print_metrics(paths)
	print(f"Artifacts | Regression: {reg_path.name}, Classification: {clf_path.name}")
	print(f"Plots | { (paths.data_dir / 'regression_scatter.png').name }, { (paths.data_dir / 'roc_curve.png').name }")

	print("[6/6] Predicting + generating explanation cards…")
	expl_path = predict_explain.generate_explanations()
	print(f"Explanations | {expl_path}")


if __name__ == "__main__":
	main()