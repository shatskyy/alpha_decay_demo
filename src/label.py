"""Compute alpha-decay labels from the SQLite database.

Outputs `data/labels.parquet` with columns:
- parent_id, asset, side, ts_signal, ts_arrival, alpha_decay, decay_flag,
  horizon, vwap_exec, r_sig, r_exec

Definitions:
- r_sig = 10,000 * (mid(ts_signal + H) - mid(ts_signal)) / mid(ts_signal)
- vwap_exec = sum(price * qty) / sum(qty) across child_fills per parent
- r_exec = 10,000 * (mid(ts_signal + H) - vwap_exec) / mid(ts_signal)
- alpha_decay = r_sig - r_exec = 10,000 * (vwap_exec - mid(ts_signal)) / mid(ts_signal)
  decay_flag = 1 if alpha_decay > 10 else 0

CLI:
    python -m src.label
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import List

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


# ---------------------------- Core labeling ---------------------------------

def _fetch_label_inputs(conn: sqlite3.Connection) -> pd.DataFrame:
	# Use a CTE to define the base parent records, then pull mid at signal, mid at target, and vwap in one pass
	query = """
	WITH base AS (
		SELECT o.parent_id,
		       s.asset,
		       s.side,
		       s.ts_signal,
		       o.ts_arrival,
		       s.alpha_horizon_min AS horizon
		FROM orders o
		JOIN signals s
		  ON s.asset = o.asset AND s.ts_signal = o.ts_arrival
	)
	SELECT
		b.parent_id,
		b.asset,
		b.side,
		b.ts_signal,
		b.ts_arrival,
		b.horizon,
		-- mid at signal (closest at or before)
		(
			SELECT m1.mid FROM market m1
			WHERE m1.asset = b.asset
			  AND m1.ts <= b.ts_signal
			  AND date(m1.ts) = date(b.ts_signal)
			ORDER BY m1.ts DESC
			LIMIT 1
		) AS mid_at_signal,
		-- mid at target horizon (closest at or before target)
		(
			SELECT m2.mid FROM market m2
			WHERE m2.asset = b.asset
			  AND m2.ts <= datetime(b.ts_signal, '+' || b.horizon || ' minutes')
			  AND date(m2.ts) = date(b.ts_signal)
			ORDER BY m2.ts DESC
			LIMIT 1
		) AS mid_at_target,
		-- fixed multi-horizon mids (30m, 120m) for multi-horizon capture
		(
			SELECT m3.mid FROM market m3
			WHERE m3.asset = b.asset
			  AND m3.ts <= datetime(b.ts_signal, '+30 minutes')
			  AND date(m3.ts) = date(b.ts_signal)
			ORDER BY m3.ts DESC
			LIMIT 1
		) AS mid_at_30,
		(
			SELECT m4.mid FROM market m4
			WHERE m4.asset = b.asset
			  AND m4.ts <= datetime(b.ts_signal, '+120 minutes')
			  AND date(m4.ts) = date(b.ts_signal)
			ORDER BY m4.ts DESC
			LIMIT 1
		) AS mid_at_120,
		-- arrival microstructure for adaptive thresholds
		(
			SELECT m5.spread_bp FROM market m5
			WHERE m5.asset = b.asset
			  AND m5.ts <= b.ts_signal
			  AND date(m5.ts) = date(b.ts_signal)
			ORDER BY m5.ts DESC
			LIMIT 1
		) AS spread_bp_at_signal,
		(
			SELECT m6.rv_30m FROM market m6
			WHERE m6.asset = b.asset
			  AND m6.ts <= b.ts_signal
			  AND date(m6.ts) = date(b.ts_signal)
			ORDER BY m6.ts DESC
			LIMIT 1
		) AS rv30_at_signal,
		-- execution VWAP across child prints
		(
			SELECT SUM(cf.price * cf.qty) * 1.0 / NULLIF(SUM(cf.qty), 0)
			FROM child_fills cf
			WHERE cf.parent_id = b.parent_id
		) AS vwap_exec
	FROM base b
	"""
	return pd.read_sql_query(query, conn)


def build_labels() -> Path:
	paths = get_paths()
	if not paths.db_path.exists():
		raise FileNotFoundError(f"Database not found at {paths.db_path}. Run `python -m src.ingest` first.")

	conn = sqlite3.connect(str(paths.db_path))
	try:
		conn.execute("PRAGMA foreign_keys = ON;")
		inputs = _fetch_label_inputs(conn)
		# Load child fills for optional time-weighted capture metrics
		child = pd.read_sql_query(
			"SELECT parent_id, ts, price, qty FROM child_fills",
			conn,
			parse_dates=["ts"],
		)
	finally:
		conn.close()

	# Ensure required fields and drop rows with missing critical values
	inputs = inputs.dropna(subset=["mid_at_signal", "mid_at_target", "vwap_exec"]).copy()

	# Compute returns in basis points with consistent denominator (mid_at_signal)
	inputs["r_sig"] = 10000.0 * (inputs["mid_at_target"] - inputs["mid_at_signal"]) / inputs["mid_at_signal"]
	inputs["r_exec"] = 10000.0 * (inputs["mid_at_target"] - inputs["vwap_exec"]) / inputs["mid_at_signal"]
	inputs["alpha_decay"] = inputs["r_sig"] - inputs["r_exec"]

	# # Signed measures and capture ratio
	# side_sign = inputs["side"].map({"BUY": 1.0, "SELL": -1.0}).fillna(0.0)
	# inputs["alpha_preserved_bps"] = side_sign * inputs["r_exec"]
	# signed_rsig = side_sign * inputs["r_sig"]
	# inputs["capture_ratio"] = np.where(np.abs(signed_rsig) > 1e-6, inputs["alpha_preserved_bps"] / signed_rsig, np.nan)

	#### NEW
	# Fixed  --> Direction-aware calculation from the start
	side_sign = inputs["side"].map({"BUY": 1.0, "SELL": -1.0}).fillna(0.0)

	# Raw returns (direction-neutral, for diagnostics)
	inputs["r_sig_raw"] = 10000.0 * (inputs["mid_at_target"] - inputs["mid_at_signal"]) / inputs["mid_at_signal"]
	inputs["r_exec_raw"] = 10000.0 * (inputs["mid_at_target"] - inputs["vwap_exec"]) / inputs["mid_at_signal"]

	# Direction-aware returns
	inputs["r_sig"] = side_sign * inputs["r_sig_raw"]
	inputs["r_exec"] = side_sign * inputs["r_exec_raw"]

	# Direction-aware alpha decay
	inputs["alpha_decay"] = inputs["r_sig"] - inputs["r_exec"]

	# Alternative slippage-only measure (for comparison)
	inputs["execution_slippage"] = side_sign * 10000.0 * (inputs["vwap_exec"] - inputs["mid_at_signal"]) / inputs["mid_at_signal"]

	###

	# Spread/fee and impact decomposition placeholders (diagnostic, not exact accounting)
	inputs["spread_fee_bps"] = inputs.get("spread_bp_at_signal", pd.Series(0.0, index=inputs.index)).astype(float)
	# impact proxy: alpha_preserved minus spread_fee component (signed)
	inputs["impact_bps"] = inputs["alpha_preserved_bps"] - inputs["spread_fee_bps"].fillna(0.0)

	# Adaptive classification threshold: function of spread and short-horizon vol
	spread = inputs.get("spread_bp_at_signal") if "spread_bp_at_signal" in inputs.columns else pd.Series(2.0, index=inputs.index)
	rv30 = inputs.get("rv30_at_signal") if "rv30_at_signal" in inputs.columns else pd.Series(0.001, index=inputs.index)
	thr = 6.0 + 2.0 * spread.clip(lower=0.5, upper=4.0) + 400.0 * rv30.clip(lower=0.0002, upper=0.003)
	inputs["decay_flag"] = (inputs["alpha_decay"] > thr).astype(int)

	# Multi-horizon capture ratios (30m and 120m) if mids are available
	for H, col_mid in [(30, "mid_at_30"), (120, "mid_at_120")]:
		if col_mid in inputs.columns:
			inputs[f"r_sig_{H}"] = 10000.0 * (inputs[col_mid] - inputs["mid_at_signal"]) / inputs["mid_at_signal"]
			inputs[f"alpha_preserved_{H}_bps"] = side_sign * (10000.0 * (inputs[col_mid] - inputs["vwap_exec"]) / inputs["mid_at_signal"])
			den = side_sign * inputs[f"r_sig_{H}"]
			inputs[f"capture_ratio_{H}"] = np.where(np.abs(den) > 1e-6, inputs[f"alpha_preserved_{H}_bps"] / den, np.nan)

	# Time-weighted capture with half-life on fills
	try:
		HALF_LIFE_MIN = 30.0
		lam = np.log(2.0) / max(1e-9, HALF_LIFE_MIN)
		child_m = child.merge(inputs[["parent_id", "ts_signal", "mid_at_signal", "mid_at_target", "side"]], on="parent_id", how="inner")
		child_m["ts_signal"] = pd.to_datetime(child_m["ts_signal"]).dt.floor("min")
		child_m["ts"] = pd.to_datetime(child_m["ts"]).dt.floor("min")
		child_m["minutes_since_signal"] = (child_m["ts"] - child_m["ts_signal"]).dt.total_seconds() / 60.0
		child_m["w"] = np.exp(-lam * child_m["minutes_since_signal"].clip(lower=0.0))
		child_m["wpx"] = child_m["w"] * child_m["price"] * child_m["qty"]
		child_m["wq"] = child_m["w"] * child_m["qty"]
		wq_sum = child_m.groupby("parent_id")["wq"].sum()
		wv = child_m.groupby("parent_id")["wpx"].sum() / wq_sum.replace(0.0, np.nan)
		wv = wv.rename("wvwap").reset_index()
		inputs = inputs.merge(wv, on="parent_id", how="left")
		ss = inputs["side"].map({"BUY": 1.0, "SELL": -1.0}).fillna(0.0)
		inputs["alpha_preserved_w_bps"] = 10000.0 * (inputs["mid_at_target"] - inputs["wvwap"]) / inputs["mid_at_signal"] * ss
		inputs["capture_ratio_w"] = np.where(np.abs(signed_rsig) > 1e-6, inputs["alpha_preserved_w_bps"] / signed_rsig, np.nan)
	except Exception:
		inputs["alpha_preserved_w_bps"] = np.nan
		inputs["capture_ratio_w"] = np.nan

	# Rename horizon and select columns
	inputs = inputs.rename(columns={"horizon": "horizon"})
	final_cols: List[str] = [
		"parent_id",
		"asset",
		"side",
		"ts_signal",
		"ts_arrival",
		"alpha_decay",
		"decay_flag",
		"horizon",
		"vwap_exec",
		"r_sig",
		"r_exec",
		"alpha_preserved_bps",
		"capture_ratio",
		"alpha_preserved_w_bps",
		"capture_ratio_w",
	]
	labels = inputs.loc[:, final_cols].copy()

	# Persist to Parquet
	out_path = paths.data_dir / "labels.parquet"
	labels.to_parquet(out_path, index=False)
	print(f"Wrote labels: {out_path} with {len(labels)} rows")
	return out_path


if __name__ == "__main__":
	build_labels()
