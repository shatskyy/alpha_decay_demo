"""Train models for alpha-decay prediction.

- Loads `data/features.parquet`
- Time-aware split by `ts_signal` date: last day → test, earlier days → train
- Regression: ElasticNetCV predicting alpha_decay (bps)
- Classification: LogisticRegressionCV predicting decay_flag with threshold tuning
- Evaluates MAE (reg) and ROC-AUC/Precision/Recall (clf) on test
- Persists artifacts to `data/`: model_reg.pkl, model_clf.pkl, feature_cols.json
- Saves quick plots to `data/`: regression_scatter.png, roc_curve.png

CLI:
	python -m src.train
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV, RidgeCV
from sklearn.metrics import mean_absolute_error, roc_auc_score, precision_score, recall_score, roc_curve, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance


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


# ---------------------------- Data loading ----------------------------------

def _load_dataset(paths: Paths) -> pd.DataFrame:
	feat_path = paths.data_dir / "features.parquet"
	if not feat_path.exists():
		raise FileNotFoundError("features.parquet not found. Run `python -m src.features` first.")
	df = pd.read_parquet(feat_path)
	# Ensure timestamps
	df["ts_signal"] = pd.to_datetime(df["ts_signal"])  # may be string
	return df


def _select_feature_columns(df: pd.DataFrame) -> List[str]:
	# Exclude non-feature/meta/target columns explicitly
	exclude = {
		"parent_id",
		"asset",
		"side",
		"ts_signal",
		"ts_arrival",
		"horizon",
		"vwap_exec",
		"r_sig",
		"r_exec",
		"alpha_decay",
		"decay_flag",
		"open_close_bucket",
	}
	cand = [c for c in df.columns if c not in exclude]
	# Keep only numeric dtypes
	feature_cols = [c for c in cand if pd.api.types.is_numeric_dtype(df[c])]
	return feature_cols


# ---------------------------- Train/eval ------------------------------------

def _time_aware_split(df: pd.DataFrame) -> Tuple[pd.Index, pd.Index]:
	"""Return train_idx, test_idx where test is the last ts_signal date."""
	dates = df["ts_signal"].dt.normalize().sort_values().unique()
	if len(dates) <= 1:
		# Fallback: 80/20 by time order
		sorted_idx = df.sort_values("ts_signal").index
		cut = int(0.8 * len(sorted_idx))
		return sorted_idx[:cut], sorted_idx[cut:]
	last_day = dates[-1]
	test_idx = df.index[df["ts_signal"].dt.normalize() == last_day]
	train_idx = df.index[df["ts_signal"].dt.normalize() < last_day]
	return train_idx, test_idx


def _pick_thresholds(y_true_train: pd.Series, proba_train: np.ndarray) -> Tuple[float, float]:
	# Max-F1 threshold search on train
	thresholds = np.linspace(0.01, 0.99, 99)
	f1s = [f1_score(y_true_train, (proba_train >= t).astype(int), zero_division=0) for t in thresholds]
	best_t = float(thresholds[int(np.argmax(f1s))])
	# Base-rate match threshold
	pos_rate = float(y_true_train.mean())
	base_t = float(np.quantile(proba_train, 1.0 - pos_rate))
	return best_t, base_t


def train_models() -> Tuple[Path, Path]:
	paths = get_paths()
	df = _load_dataset(paths)
	feature_cols = _select_feature_columns(df)

	X = df[feature_cols].copy()
	y_reg = df["alpha_decay"].astype(float)
	y_cls = df["decay_flag"].astype(int)

	train_idx, test_idx = _time_aware_split(df)
	X_train, X_test = X.loc[train_idx], X.loc[test_idx]
	y_reg_train, y_reg_test = y_reg.loc[train_idx], y_reg.loc[test_idx]
	y_cls_train, y_cls_test = y_cls.loc[train_idx], y_cls.loc[test_idx]

	# Pipelines with imputer + scaler
	reg_pipe: Pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="median")),
			("scaler", StandardScaler(with_mean=True)),
			(
				"model",
				ElasticNetCV(
					l1_ratio=[0.05, 0.2, 0.5, 0.8, 0.95],
					alphas=np.logspace(-3, 1, 20),
					cv=5,
					random_state=7,
				),
			),
		]
	)
	clf_pipe: Pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="median")),
			("scaler", StandardScaler(with_mean=True)),
			("model", LogisticRegressionCV(cv=5, max_iter=2000, class_weight="balanced")),
		]
	)

	# Fit
	reg_pipe.fit(X_train, y_reg_train)
	clf_pipe.fit(X_train, y_cls_train)

	# Evaluate regression with potential Ridge fallback if collapsed
	y_reg_pred_test = reg_pipe.predict(X_test)
	std_pred = float(np.std(y_reg_pred_test))
	if std_pred < 1e-3:
		# Fallback to Ridge for a breathing baseline
		ridge_pipe: Pipeline = Pipeline(
			steps=[
				("imputer", SimpleImputer(strategy="median")),
				("scaler", StandardScaler(with_mean=True)),
				("model", RidgeCV(alphas=np.logspace(-3, 3, 20))),
			]
		)
		ridge_pipe.fit(X_train, y_reg_train)
		reg_pipe = ridge_pipe
		y_reg_pred_test = reg_pipe.predict(X_test)

	mae_bps = float(mean_absolute_error(y_reg_test, y_reg_pred_test))
	print(f"Regression MAE (bps) on test: {mae_bps:.2f}")
	print(f"Std of regression predictions on test: {np.std(y_reg_pred_test):.3f}")

	# Classification: probabilities for class 1 and thresholding
	proba_train = clf_pipe.predict_proba(X_train)[:, 1]
	proba_test = clf_pipe.predict_proba(X_test)[:, 1]
	best_t, base_t = _pick_thresholds(y_cls_train, proba_train)
	y_cls_pred_best = (proba_test >= best_t).astype(int)
	y_cls_pred_base = (proba_test >= base_t).astype(int)

	roc_auc = float(roc_auc_score(y_cls_test, proba_test)) if y_cls_test.nunique() > 1 else float("nan")
	precision_best = float(precision_score(y_cls_test, y_cls_pred_best, zero_division=0))
	recall_best = float(recall_score(y_cls_test, y_cls_pred_best, zero_division=0))
	precision_base = float(precision_score(y_cls_test, y_cls_pred_base, zero_division=0))
	recall_base = float(recall_score(y_cls_test, y_cls_pred_base, zero_division=0))
	print(f"Classification ROC-AUC on test: {roc_auc:.3f}")
	print(f"Classification threshold (Max-F1) = {best_t:.3f} | Precision/Recall: {precision_best:.3f} / {recall_best:.3f}")
	print(f"Classification threshold (Base-rate) = {base_t:.3f} | Precision/Recall: {precision_base:.3f} / {recall_base:.3f}")
	print(f"Test positive rate (decay_flag): {y_cls_test.mean():.3f}")

	# Permutation importance (regression) diagnostics on test
	try:
		perm = permutation_importance(reg_pipe, X_test, y_reg_test, n_repeats=10, random_state=7, n_jobs=1)
		imp_pairs = sorted(zip(feature_cols, perm.importances_mean), key=lambda x: abs(x[1]), reverse=True)[:10]
		print("Top 10 permutation importances (reg):")
		for name, val in imp_pairs:
			print(f"  {name}: {val:.4f}")
	except Exception:
		pass

	# Persist artifacts
	reg_path = paths.data_dir / "model_reg.pkl"
	clf_path = paths.data_dir / "model_clf.pkl"
	joblib.dump(reg_pipe, reg_path)
	joblib.dump(clf_pipe, clf_path)
	with (paths.data_dir / "feature_cols.json").open("w") as f:
		json.dump(feature_cols, f, indent=2)

	# Plots
	plt.figure(figsize=(5, 5))
	plt.scatter(y_reg_test, y_reg_pred_test, s=10, alpha=0.6)
	mn = float(np.nanmin([y_reg_test.min(), y_reg_pred_test.min()]))
	mx = float(np.nanmax([y_reg_test.max(), y_reg_pred_test.max()]))
	plt.plot([mn, mx], [mn, mx], "r--", linewidth=1)
	plt.xlabel("True alpha_decay (bps)")
	plt.ylabel("Predicted (bps)")
	plt.title("Regression: y_true vs y_pred")
	plt.tight_layout()
	plt.savefig(paths.data_dir / "regression_scatter.png", dpi=150)
	plt.close()

	fpr, tpr, _ = roc_curve(y_cls_test, proba_test) if y_cls_test.nunique() > 1 else (np.array([0, 1]), np.array([0, 1]), None)
	plt.figure(figsize=(5, 5))
	plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
	plt.plot([0, 1], [0, 1], "k--", linewidth=1)
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.title("Classification: ROC Curve")
	plt.legend(loc="lower right")
	plt.tight_layout()
	plt.savefig(paths.data_dir / "roc_curve.png", dpi=150)
	plt.close()

	return reg_path, clf_path


if __name__ == "__main__":
	train_models()