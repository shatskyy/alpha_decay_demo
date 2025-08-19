"""Ingest CSVs into a SQLite database for the alpha-decay demo.

Builds `db/alpha.sqlite` (overwriting if it exists), creates the following tables
with types and indexes, and bulk inserts rows from CSVs in `data/`:

- signals(ts_signal TEXT, asset TEXT, side TEXT, signal_score REAL,
          alpha_horizon_min INTEGER, signal_strength_rank REAL)
  index: (asset, ts_signal)

- orders(parent_id TEXT PRIMARY KEY, ts_arrival TEXT, asset TEXT, side TEXT,
         qty INTEGER, urgency_tag TEXT, algo_type TEXT,
         participation_cap INTEGER, broker TEXT, venue_hint TEXT)
  indexes: (asset, ts_arrival), (parent_id)

- child_fills(parent_id TEXT, ts TEXT, price REAL, qty INTEGER,
              venue TEXT, order_type TEXT)
  index: (parent_id)

- market(ts TEXT, asset TEXT, mid REAL, bid REAL, ask REAL, spread_bp REAL,
         depth1_bid INTEGER, depth1_ask INTEGER, imbalance REAL,
         rv_5m REAL, rv_30m REAL, adv INTEGER, turnover REAL)
  index: (asset, ts)

CLI:
    python -m src.ingest
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Iterable, List, Sequence, Tuple

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
	db_dir.mkdir(parents=True, exist_ok=True)
	db_path = db_dir / "alpha.sqlite"
	return Paths(project_root=project_root, data_dir=data_dir, db_dir=db_dir, db_path=db_path)


# ---------------------------- DB helpers ------------------------------------

def recreate_database(db_path: Path) -> sqlite3.Connection:
	if db_path.exists():
		db_path.unlink()
	conn = sqlite3.connect(str(db_path))
	conn.execute("PRAGMA journal_mode = WAL;")
	conn.execute("PRAGMA synchronous = NORMAL;")
	conn.execute("PRAGMA foreign_keys = ON;")
	return conn


def create_tables(conn: sqlite3.Connection) -> None:
	cur = conn.cursor()
	cur.executescript(
		"""
		CREATE TABLE signals (
			ts_signal TEXT NOT NULL,
			asset TEXT NOT NULL,
			side TEXT NOT NULL,
			signal_score REAL NOT NULL,
			alpha_horizon_min INTEGER NOT NULL,
			signal_strength_rank REAL NOT NULL
		);
		CREATE INDEX idx_signals_asset_ts ON signals(asset, ts_signal);

		CREATE TABLE orders (
			parent_id TEXT PRIMARY KEY,
			ts_arrival TEXT NOT NULL,
			asset TEXT NOT NULL,
			side TEXT NOT NULL,
			qty INTEGER NOT NULL,
			urgency_tag TEXT NOT NULL,
			algo_type TEXT NOT NULL,
			participation_cap INTEGER NOT NULL,
			broker TEXT NOT NULL,
			venue_hint TEXT NOT NULL
		);
		CREATE INDEX idx_orders_asset_ts ON orders(asset, ts_arrival);
		CREATE INDEX IF NOT EXISTS idx_orders_parent_id ON orders(parent_id);

		CREATE TABLE child_fills (
			parent_id TEXT NOT NULL,
			ts TEXT NOT NULL,
			price REAL NOT NULL,
			qty INTEGER NOT NULL,
			venue TEXT NOT NULL,
			order_type TEXT NOT NULL
		);
		CREATE INDEX idx_child_parent ON child_fills(parent_id);

		CREATE TABLE market (
			ts TEXT NOT NULL,
			asset TEXT NOT NULL,
			mid REAL NOT NULL,
			bid REAL NOT NULL,
			ask REAL NOT NULL,
			spread_bp REAL NOT NULL,
			depth1_bid INTEGER NOT NULL,
			depth1_ask INTEGER NOT NULL,
			imbalance REAL NOT NULL,
			rv_5m REAL NOT NULL,
			rv_30m REAL NOT NULL,
			adv INTEGER NOT NULL,
			turnover REAL NOT NULL
		);
		CREATE INDEX idx_market_asset_ts ON market(asset, ts);
		"""
	)
	conn.commit()


# ---------------------------- CSV loading -----------------------------------

def _ensure_iso8601(series: pd.Series) -> pd.Series:
	"""Convert a datetime-like series to ISO8601 strings with minute/second precision."""
	s = pd.to_datetime(series, utc=False)
	# Keep full second-level resolution; inputs are minute-level
	return s.dt.strftime("%Y-%m-%d %H:%M:%S")


def _rows(df: pd.DataFrame, columns: Sequence[str]) -> Iterable[Tuple]:
	for row in df.loc[:, list(columns)].itertuples(index=False, name=None):
		yield tuple(None if pd.isna(v) else v for v in row)


def load_signals(conn: sqlite3.Connection, data_dir: Path) -> int:
	csv_path = data_dir / "signals.csv"
	df = pd.read_csv(csv_path, parse_dates=["ts_signal"]) if csv_path.exists() else pd.DataFrame()
	if df.empty:
		return 0
	df["ts_signal"] = _ensure_iso8601(df["ts_signal"]) 
	cols = ["ts_signal", "asset", "side", "signal_score", "alpha_horizon_min", "signal_strength_rank"]
	conn.executemany(
		"INSERT INTO signals (ts_signal, asset, side, signal_score, alpha_horizon_min, signal_strength_rank) VALUES (?,?,?,?,?,?)",
		list(_rows(df, cols)),
	)
	conn.commit()
	return int(len(df))


def load_orders(conn: sqlite3.Connection, data_dir: Path) -> int:
	csv_path = data_dir / "orders.csv"
	df = pd.read_csv(csv_path, parse_dates=["ts_arrival"]) if csv_path.exists() else pd.DataFrame()
	if df.empty:
		return 0
	df["ts_arrival"] = _ensure_iso8601(df["ts_arrival"]) 
	cols = [
		"parent_id",
		"ts_arrival",
		"asset",
		"side",
		"qty",
		"urgency_tag",
		"algo_type",
		"participation_cap",
		"broker",
		"venue_hint",
	]
	conn.executemany(
		"INSERT INTO orders (parent_id, ts_arrival, asset, side, qty, urgency_tag, algo_type, participation_cap, broker, venue_hint) VALUES (?,?,?,?,?,?,?,?,?,?)",
		list(_rows(df, cols)),
	)
	conn.commit()
	return int(len(df))


def load_child_fills(conn: sqlite3.Connection, data_dir: Path) -> int:
	csv_path = data_dir / "child_fills.csv"
	df = pd.read_csv(csv_path, parse_dates=["ts"]) if csv_path.exists() else pd.DataFrame()
	if df.empty:
		return 0
	df["ts"] = _ensure_iso8601(df["ts"]) 
	cols = ["parent_id", "ts", "price", "qty", "venue", "order_type"]
	conn.executemany(
		"INSERT INTO child_fills (parent_id, ts, price, qty, venue, order_type) VALUES (?,?,?,?,?,?)",
		list(_rows(df, cols)),
	)
	conn.commit()
	return int(len(df))


def load_market(conn: sqlite3.Connection, data_dir: Path) -> int:
	csv_path = data_dir / "market.csv"
	df = pd.read_csv(csv_path, parse_dates=["ts"]) if csv_path.exists() else pd.DataFrame()
	if df.empty:
		return 0
	df["ts"] = _ensure_iso8601(df["ts"]) 
	# Fill NaNs for NOT NULL numeric columns
	for col, val in {
		"mid": 0.0,
		"bid": 0.0,
		"ask": 0.0,
		"spread_bp": 0.0,
		"depth1_bid": 0,
		"depth1_ask": 0,
		"imbalance": 0.0,
		"rv_5m": 1e-6,
		"rv_30m": 1e-6,
		"adv": 0,
		"turnover": 0.0,
	}.items():
		if col in df.columns:
			df[col] = df[col].fillna(val)
	cols = [
		"ts",
		"asset",
		"mid",
		"bid",
		"ask",
		"spread_bp",
		"depth1_bid",
		"depth1_ask",
		"imbalance",
		"rv_5m",
		"rv_30m",
		"adv",
		"turnover",
	]
	conn.executemany(
		"INSERT INTO market (ts, asset, mid, bid, ask, spread_bp, depth1_bid, depth1_ask, imbalance, rv_5m, rv_30m, adv, turnover) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
		list(_rows(df, cols)),
	)
	conn.commit()
	return int(len(df))


# ---------------------------- Orchestration ---------------------------------

def build_database() -> None:
	paths = get_paths()
	conn = recreate_database(paths.db_path)
	try:
		create_tables(conn)
		n_signals = load_signals(conn, paths.data_dir)
		n_orders = load_orders(conn, paths.data_dir)
		n_child = load_child_fills(conn, paths.data_dir)
		n_market = load_market(conn, paths.data_dir)
		print(
			f"Built {paths.db_path.name}: signals={n_signals}, orders={n_orders}, child_fills={n_child}, market={n_market}"
		)
	finally:
		conn.close()


if __name__ == "__main__":
	build_database()