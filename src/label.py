"""Compute alpha-decay labels from the SQLite database.

Outputs `data/labels.parquet` with columns:
- parent_id, asset, side, ts_signal, ts_arrival, alpha_decay, decay_flag,
  horizon, vwap_exec, r_sig, r_exec

Definitions:
- r_sig = 10,000 * (mid(ts_signal + H) - mid(ts_signal)) / mid(ts_signal)
- vwap_exec = sum(price * qty) / sum(qty) across child_fills per parent
- r_exec = 10,000 * (mid(ts_signal + H) - vwap_exec) / vwap_exec
- alpha_decay = r_sig - r_exec (bps); decay_flag = 1 if alpha_decay > 10 else 0

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
	finally:
		conn.close()

	# Ensure required fields and drop rows with missing critical values
	inputs = inputs.dropna(subset=["mid_at_signal", "mid_at_target", "vwap_exec"]).copy()

	# Compute returns in basis points
	inputs["r_sig"] = 10000.0 * (inputs["mid_at_target"] - inputs["mid_at_signal"]) / inputs["mid_at_signal"]
	inputs["r_exec"] = 10000.0 * (inputs["mid_at_target"] - inputs["vwap_exec"]) / inputs["vwap_exec"]
	inputs["alpha_decay"] = inputs["r_sig"] - inputs["r_exec"]
	inputs["decay_flag"] = (inputs["alpha_decay"] > 10.0).astype(int)

	# Rename horizon for clarity and select final columns in order
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
	]
	labels = inputs.loc[:, final_cols].copy()

	# Persist to Parquet
	out_path = paths.data_dir / "labels.parquet"
	labels.to_parquet(out_path, index=False)
	print(f"Wrote labels: {out_path} with {len(labels)} rows")
	return out_path


if __name__ == "__main__":
	build_labels()