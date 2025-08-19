"""Create parent-order level features by joining market snapshots and execution aggregates.

Outputs `data/features.parquet` with features, labels, and metadata for modeling.

Features include:
- Latency: age_sec, exec_dur_sec, horizon_to_age
- Arrival microstructure: spread_bp, imbalance, rv_5m, rv_30m, depth1_bid, depth1_ask
- Footprint: participation_est, pct_dark, pct_marketable, reprice_rate
- Time: minute_of_day, open_close_bucket ∈ {OPEN1H, MID, CLOSE1H}
- Signal context: signal_score, signal_strength_rank, side_sign {+1,-1}

Targets:
- y_reg = alpha_decay (bps)
- y_cls = decay_flag (binary)

CLI:
    python -m src.features
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import sqlite3

import numpy as np
import pandas as pd


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


# ---------------------------- Helpers ---------------------------------------

def _bucket_open_close(ts: pd.Timestamp) -> str:
	hour = ts.hour
	minute = ts.minute
	minutes_since_open = (hour * 60 + minute) - (9 * 60 + 30)
	if 0 <= minutes_since_open < 60:
		return "OPEN1H"
	if 15 * 60 <= hour * 60 + minute < 16 * 60:
		return "CLOSE1H"
	return "MID"


def _arrival_snapshot(market_df: pd.DataFrame, arrivals: pd.DataFrame) -> pd.DataFrame:
	# Robust per-asset merge_asof with proper sorting to avoid "left keys must be sorted"
	market_df = market_df.copy()
	arrivals = arrivals.copy()
	market_df["ts"] = pd.to_datetime(market_df["ts"], errors="coerce")
	arrivals["ts_arrival"] = pd.to_datetime(arrivals["ts_arrival"], errors="coerce")
	arrivals = arrivals.dropna(subset=["asset", "ts_arrival"]).reset_index(drop=True)
	market_df = market_df.dropna(subset=["asset", "ts"]).reset_index(drop=True)

	expected_cols = ["spread_bp", "imbalance", "rv_5m", "rv_30m", "depth1_bid", "depth1_ask", "turnover", "mid"]
	parts: List[pd.DataFrame] = []
	assets = sorted(set(arrivals["asset"]).intersection(set(market_df["asset"])))
	for asset in assets:
		a = arrivals.loc[arrivals["asset"] == asset].sort_values("ts_arrival").reset_index(drop=True)
		m = market_df.loc[market_df["asset"] == asset].sort_values("ts").reset_index(drop=True)
		if a.empty or m.empty:
			continue
		merged = pd.merge_asof(
			a,
			m,
			left_on="ts_arrival",
			right_on="ts",
			direction="backward",
			allow_exact_matches=True,
		)
		parts.append(merged)
	if parts:
		out = pd.concat(parts, ignore_index=True)
	else:
		# No overlaps; return arrivals with NaNs for expected microstructure columns
		for c in expected_cols:
			if c not in arrivals:
				arrivals[c] = np.nan
		out = arrivals
	return out


def _estimate_lit_volume(market_df: pd.DataFrame, asset: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp, lit_fraction: float = 0.7) -> float:
	mask = (market_df.asset == asset) & (market_df.ts >= start_ts.floor("min")) & (market_df.ts <= end_ts.floor("min"))
	sub = market_df.loc[mask, ["turnover", "mid"]]
	if sub.empty:
		return 0.0
	shares_est = (sub["turnover"] / sub["mid"]).sum()
	return float(lit_fraction * shares_est)


# ---------------------------- Core features ---------------------------------

def build_features() -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
	paths = get_paths()
	labels_path = paths.data_dir / "labels.parquet"
	if not labels_path.exists():
		raise FileNotFoundError("labels.parquet not found. Run `python -m src.label` first.")

	labels = pd.read_parquet(labels_path)
	# Ensure dtypes
	labels["ts_signal"] = pd.to_datetime(labels["ts_signal"])
	labels["ts_arrival"] = pd.to_datetime(labels["ts_arrival"])

	# Fetch needed tables from DB
	if not paths.db_path.exists():
		raise FileNotFoundError("alpha.sqlite not found. Run `python -m src.ingest` first.")
	conn = sqlite3.connect(str(paths.db_path))
	try:
		signals = pd.read_sql_query(
			"SELECT asset, ts_signal, side, signal_score, signal_strength_rank FROM signals",
			conn,
			parse_dates=["ts_signal"],
		)
		# Child fills (full; small enough) for footprint metrics
		child = pd.read_sql_query(
			"SELECT parent_id, ts, price, qty, venue, order_type FROM child_fills",
			conn,
			parse_dates=["ts"],
		)
		# Market minute bars
		market = pd.read_sql_query(
			"SELECT ts, asset, spread_bp, imbalance, rv_5m, rv_30m, depth1_bid, depth1_ask, turnover, mid FROM market",
			conn,
			parse_dates=["ts"],
		)
	finally:
		conn.close()

	# Merge signal context into labels (join on asset & ts_signal)
	lbl = labels.merge(
		signals,
		on=["asset", "ts_signal"],
		how="left",
		suffixes=("", "_sig"),
	)

	# Aggregate child fills per parent for qty splits and first/last timestamps
	agg = (
		child.groupby("parent_id")
		.agg(
			child_qty=("qty", "sum"),
			ts_first_fill=("ts", "min"),
			ts_last_fill=("ts", "max"),
			dark_qty=("venue", lambda s: int((child.loc[s.index, "qty"][(child.loc[s.index, "venue"] == "DARK")]).sum() if len(s) else 0)),
			marketable_qty=("order_type", lambda s: int((child.loc[s.index, "qty"][(child.loc[s.index, "order_type"] == "MARKETABLE")]).sum() if len(s) else 0)),
			n_limit_min=("ts", lambda ts: int(pd.to_datetime(child.loc[ts.index].loc[child.loc[ts.index, "order_type"] == "LIMIT", "ts"]).dt.floor("min").nunique())),
		)
		.reset_index()
	)

	# Merge aggregates to labels
	df = lbl.merge(agg, on="parent_id", how="left")

	# Arrival microstructure snapshot via asof-merge
	arrival_keys = df[["parent_id", "asset", "ts_arrival"]].dropna()
	arrival_snap = _arrival_snapshot(
		market_df=market[["asset", "ts", "spread_bp", "imbalance", "rv_5m", "rv_30m", "depth1_bid", "depth1_ask", "turnover", "mid"]],
		arrivals=arrival_keys,
	)
	arrival_cols = [
		"parent_id",
		"spread_bp",
		"imbalance",
		"rv_5m",
		"rv_30m",
		"depth1_bid",
		"depth1_ask",
	]
	df = df.merge(arrival_snap[[*arrival_cols]], on="parent_id", how="left")

	# Latency metrics
	df["age_sec"] = (pd.to_datetime(df["ts_first_fill"]) - pd.to_datetime(df["ts_signal"]) ).dt.total_seconds().clip(lower=0).fillna(0)
	df["exec_dur_sec"] = (pd.to_datetime(df["ts_last_fill"]) - pd.to_datetime(df["ts_first_fill"]) ).dt.total_seconds().clip(lower=0).fillna(0)
	df["horizon_to_age"] = (df["horizon"].astype(float) * 60.0) / (df["age_sec"] + 1.0)

	# Footprint metrics
	df["child_qty"] = df["child_qty"].fillna(0).astype(float)
	df["dark_qty"] = df["dark_qty"].fillna(0).astype(float)
	df["marketable_qty"] = df["marketable_qty"].fillna(0).astype(float)
	df["pct_dark"] = np.where(df["child_qty"] > 0, df["dark_qty"] / df["child_qty"], 0.0)
	df["pct_marketable"] = np.where(df["child_qty"] > 0, df["marketable_qty"] / df["child_qty"], 0.0)

	# Reprice rate: unique minutes with LIMIT child fills divided by execution minutes
	exec_minutes = np.ceil(df["exec_dur_sec"].fillna(0) / 60.0) + 1.0
	df["reprice_rate"] = np.where(exec_minutes > 0, df["n_limit_min"].fillna(0) / exec_minutes, 0.0)

	# Estimate lit volume over execution window (loop is fine: ~O(1e2) parents)
	market_idx = market.sort_values(["asset", "ts"]).reset_index(drop=True)
	lit_volume_est_list: List[float] = []
	for _, row in df.iterrows():
		asset = row["asset"]
		start_ts = pd.to_datetime(row["ts_first_fill"]) if pd.notna(row["ts_first_fill"]) else pd.to_datetime(row["ts_arrival"]) if pd.notna(row["ts_arrival"]) else None
		end_ts = pd.to_datetime(row["ts_last_fill"]) if pd.notna(row["ts_last_fill"]) else start_ts
		if asset is None or start_ts is None or end_ts is None:
			lit_volume_est_list.append(0.0)
			continue
		lit_vol = _estimate_lit_volume(market_idx, asset, start_ts, end_ts, lit_fraction=0.7)
		lit_volume_est_list.append(lit_vol)
	df["lit_volume_est"] = pd.Series(lit_volume_est_list, index=df.index).astype(float)
	# Participation estimate
	df["participation_est"] = df["child_qty"] / (df["lit_volume_est"] + 1.0)

	# Time features
	df["minute_of_day"] = pd.to_datetime(df["ts_arrival"]).dt.hour * 60 + pd.to_datetime(df["ts_arrival"]).dt.minute
	df["open_close_bucket"] = pd.to_datetime(df["ts_arrival"]).apply(_bucket_open_close)

	# Signal side encoding {+1, -1}
	df["side_sign"] = df["side"].map({"BUY": 1, "SELL": -1}).fillna(0).astype(int)

	# Select outputs
	numeric_feature_cols = [
		"age_sec",
		"exec_dur_sec",
		"horizon_to_age",
		"spread_bp",
		"imbalance",
		"rv_5m",
		"rv_30m",
		"depth1_bid",
		"depth1_ask",
		"child_qty",
		"lit_volume_est",
		"participation_est",
		"pct_dark",
		"pct_marketable",
		"reprice_rate",
		"minute_of_day",
		"signal_score",
		"signal_strength_rank",
		"side_sign",
	]
	y_reg = df["alpha_decay"].astype(float)
	y_cls = df["decay_flag"].astype(int)

	meta_cols = [
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
	]

	# Persist full dataset (features + targets + meta)
	out = df[[*meta_cols, *numeric_feature_cols]].copy()
	out_path = paths.data_dir / "features.parquet"
	out.to_parquet(out_path, index=False)
	print(f"Wrote features: {out_path} with {len(out)} rows and {len(out.columns)} columns")

	X = df[numeric_feature_cols].copy()
	return X, y_reg, y_cls, df[meta_cols].copy()


if __name__ == "__main__":
	build_features()