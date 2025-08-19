"""Simulate mock market and execution data for the alpha-decay demo.

Generates CSVs into the project's `data/` directory:

1) signals.csv
   columns: ts_signal, asset, side, signal_score, alpha_horizon_min, signal_strength_rank
   - 5 assets from ['AAPL','MSFT','AMZN','GOOG','META']
   - business days: last 5 sessions (synthetic)
   - one signal per hour between 10:00–15:00
   - signal_score ~ N(0,1), signal_strength_rank in [0,1], horizon = 60 minutes
   - side ∈ {BUY, SELL} correlated with score sign

2) orders.csv
   columns: parent_id, ts_arrival, asset, side, qty, urgency_tag, algo_type, participation_cap, broker, venue_hint
   - 1 parent order per signal; parent_id = SHA1(asset|ts_signal)
   - qty lognormal; urgency_tag ∈ {LOW,MED,HIGH}; algo_type ∈ {POV,SCHEDULE,DISCRETIONARY}
   - participation_cap in [5,35]%

3) child_fills.csv
   columns: parent_id, ts, price, qty, venue, order_type
   - random child schedule over ~15–45 minutes post arrival
   - price path uses market mid plus small impact + noise
   - venue in {LIT,DARK}; order_type in {LIMIT,MARKETABLE,PEG}
   - ensure sum child qty == parent qty

4) market.csv
   columns: ts, asset, mid, bid, ask, spread_bp, depth1_bid, depth1_ask, imbalance, rv_5m, rv_30m, adv, turnover
   - 1-min bars for each asset/day from 09:30–16:00
   - realistic mid with intraday U-shape vol and widening spread near open/close
   - imbalance in [-1,1]; rv_* > 0; spread_bp ~ lognormal around 1–3bp

The data size is intentionally small to run in seconds. Randomness is seeded for
reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
import hashlib
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd


# ---------------------------- Configuration ---------------------------------

ASSETS: List[str] = ["AAPL", "MSFT", "AMZN", "GOOG", "META"]
TRADING_START = time(9, 30)
TRADING_END = time(16, 0)
SIGNAL_HOURS = [10, 11, 12, 13, 14, 15]
ALPHA_HORIZON_MINUTES = 60
RNG_SEED = 42

URGENCY_TAGS = ["LOW", "MED", "HIGH"]
ALGO_TYPES = ["POV", "SCHEDULE", "DISCRETIONARY"]
BROKERS = ["BrokerA", "BrokerB", "BrokerC"]
VENUE_HINTS = ["LIT", "DARK", "SMART"]

CHILD_VENUES = ["LIT", "DARK"]
ORDER_TYPES = ["LIMIT", "MARKETABLE", "PEG"]

# Minute bars per regular US session (09:30–16:00) = 390
MINUTES_PER_SESSION = 390


@dataclass(frozen=True)
class Paths:
	project_root: Path
	data_dir: Path


def get_paths() -> Paths:
	# Locate the project root as the parent of this file's parent directory
	src_dir = Path(__file__).resolve().parent
	project_root = src_dir.parent
	data_dir = project_root / "data"
	data_dir.mkdir(parents=True, exist_ok=True)
	return Paths(project_root=project_root, data_dir=data_dir)


def get_rng(seed: int = RNG_SEED) -> np.random.Generator:
	return np.random.default_rng(seed)


# ---------------------------- Date utilities --------------------------------

def get_last_business_days(n_days: int = 5) -> List[pd.Timestamp]:
	today = pd.Timestamp.today().normalize()
	bdays = pd.bdate_range(end=today, periods=n_days)
	# Return as list of normalized timestamps
	return [pd.Timestamp(d.date()) for d in bdays]


def session_minutes_for_day(day: pd.Timestamp) -> pd.DatetimeIndex:
	start_dt = pd.Timestamp.combine(day, TRADING_START)
	end_dt = pd.Timestamp.combine(day, TRADING_END)
	# Generate a closed-open interval [start, end) to get 390 minutes
	# Pandas date_range is inclusive on both ends if "inclusive='both'",
	# so we end one minute before the session end.
	minutes = pd.date_range(start=start_dt, end=end_dt - timedelta(minutes=1), freq="1min")
	return minutes


# ---------------------------- Market simulation -----------------------------

def _u_shaped_vol_profile(n: int) -> np.ndarray:
	# High volatility near open and close, lower in the middle
	x = np.linspace(0.0, 1.0, n)
	shape = 0.6 + 1.2 * np.abs(x - 0.5) * 2.0  # ~0.6 mid, up to ~1.8 edges
	return shape


def _base_price_for_asset(asset: str, rng: np.random.Generator) -> float:
	base_map = {
		"AAPL": 180.0,
		"MSFT": 390.0,
		"AMZN": 140.0,
		"GOOG": 135.0,
		"META": 480.0,
	}
	base = base_map.get(asset, 100.0)
	jitter = rng.normal(0.0, base * 0.005)
	return max(5.0, base + jitter)


def simulate_market(days: List[pd.Timestamp], rng: np.random.Generator) -> pd.DataFrame:
	records = []
	for asset in ASSETS:
		adv_shares = int(rng.uniform(1.5e6, 7.5e6))  # per-asset ADV
		for day in days:
			minutes = session_minutes_for_day(day)
			n = len(minutes)
			vol_profile = _u_shaped_vol_profile(n)

			base_price = _base_price_for_asset(asset, rng)
			base_daily_vol_bp = rng.uniform(60.0, 120.0)  # daily vol 60–120bp
			per_min_vol = (base_daily_vol_bp / 10000.0) / np.sqrt(MINUTES_PER_SESSION)
			minute_sigma = per_min_vol * vol_profile

			# Generate zero-mean minute returns
			minute_rets = rng.normal(loc=0.0, scale=minute_sigma, size=n)
			prices = base_price * np.exp(np.cumsum(minute_rets))

			# Spread in bp lognormal around 1–3bp, wider at open/close
			spread_ln_mu = np.log(0.0002)  # ~2bp
			spread_ln_sigma = 0.25
			spread_abs = np.exp(rng.normal(spread_ln_mu, spread_ln_sigma, size=n))
			spread_abs *= (0.9 + 1.6 * np.abs(np.linspace(0, 1, n) - 0.5) * 2)

			mid = prices
			spread = spread_abs * mid
			bid = mid - spread / 2.0
			ask = mid + spread / 2.0

			depth_bid = rng.integers(3000, 15000, size=n)
			depth_ask = rng.integers(3000, 15000, size=n)

			# Imbalance in [-1, 1]
			raw_imb = (depth_bid - depth_ask) / (depth_bid + depth_ask + 1e-9)
			noise = rng.normal(0.0, 0.05, size=n)
			imbalance = np.clip(raw_imb + noise, -1.0, 1.0)

			# Rolling realized vol (use log returns)
			log_mid = np.log(mid)
			log_ret = np.diff(log_mid, prepend=log_mid[0])
			rv_5m = pd.Series(log_ret).rolling(5, min_periods=1).std().to_numpy()
			rv_30m = pd.Series(log_ret).rolling(30, min_periods=1).std().to_numpy()

			# Spread in basis points
			spread_bp = (ask - bid) / mid * 10000.0

			# Turnover per minute ~ adv/390 shares traded at price
			turnover = (adv_shares / MINUTES_PER_SESSION) * mid

			for i, ts in enumerate(minutes):
				records.append(
					{
						"ts": ts,
						"asset": asset,
						"mid": float(mid[i]),
						"bid": float(bid[i]),
						"ask": float(ask[i]),
						"spread_bp": float(spread_bp[i]),
						"depth1_bid": int(depth_bid[i]),
						"depth1_ask": int(depth_ask[i]),
						"imbalance": float(imbalance[i]),
						"rv_5m": float(max(rv_5m[i], 1e-6)),
						"rv_30m": float(max(rv_30m[i], 1e-6)),
						"adv": int(adv_shares),
						"turnover": float(turnover[i]),
					}
				)

	market_df = pd.DataFrame.from_records(records)
	return market_df


# ---------------------------- Signals simulation ----------------------------

def simulate_signals(days: List[pd.Timestamp], rng: np.random.Generator) -> pd.DataFrame:
	rows = []
	for day in days:
		for hour in SIGNAL_HOURS:
			ts_base = pd.Timestamp.combine(day, time(hour, 0))
			for asset in ASSETS:
				score = float(rng.normal(0.0, 1.0))
				strength = float(rng.uniform(0.0, 1.0))
				side = "BUY" if score >= 0.0 else "SELL"
				rows.append(
					{
						"ts_signal": ts_base,
						"asset": asset,
						"side": side,
						"signal_score": score,
						"alpha_horizon_min": ALPHA_HORIZON_MINUTES,
						"signal_strength_rank": strength,
					}
				)
	signals_df = pd.DataFrame(rows)
	return signals_df


# ---------------------------- Orders simulation -----------------------------

def _parent_id(asset: str, ts_signal: pd.Timestamp) -> str:
	text = f"{asset}|{ts_signal.isoformat()}".encode("utf-8")
	return hashlib.sha1(text).hexdigest()


def simulate_orders(signals_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
	qty_mu = 9.0  # log scale mean
	qty_sigma = 0.6
	rows = []
	for _, s in signals_df.iterrows():
		parent_id = _parent_id(s["asset"], s["ts_signal"])
		# lognormal in shares, then clip to a reasonable range
		qty = int(max(500, min(50000, rng.lognormal(mean=qty_mu, sigma=qty_sigma))))
		urgency = rng.choice(URGENCY_TAGS, p=[0.4, 0.4, 0.2])
		algo = rng.choice(ALGO_TYPES, p=[0.5, 0.3, 0.2])
		participation_cap = int(rng.integers(5, 36))
		broker = rng.choice(BROKERS)
		venue_hint = rng.choice(VENUE_HINTS, p=[0.6, 0.2, 0.2])
		rows.append(
			{
				"parent_id": parent_id,
				"ts_arrival": s["ts_signal"],
				"asset": s["asset"],
				"side": s["side"],
				"qty": qty,
				"urgency_tag": urgency,
				"algo_type": algo,
				"participation_cap": participation_cap,
				"broker": broker,
				"venue_hint": venue_hint,
			}
		)
	orders_df = pd.DataFrame(rows)
	return orders_df


# ---------------------------- Child fills simulation ------------------------

def _nearest_minute(ts: pd.Timestamp) -> pd.Timestamp:
	return ts.floor("min")


def _child_qty_allocation(total_qty: int, n_child: int, rng: np.random.Generator) -> List[int]:
	# Use a Dirichlet-like approach via Gamma to get positive parts that sum to ~1, then scale
	weights = rng.gamma(shape=1.0, scale=1.0, size=n_child)
	weights = weights / weights.sum()
	raw = np.floor(weights * total_qty).astype(int)
	remainder = int(total_qty - raw.sum())
	# Distribute remainder one by one to random indices
	if remainder > 0:
		for idx in rng.choice(np.arange(n_child), size=remainder, replace=True):
			raw[idx] += 1
	return raw.tolist()


def simulate_child_fills(
	orders_df: pd.DataFrame, market_df: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
	# Build lookup for minute mid prices per asset
	market_lookup: Dict[Tuple[str, pd.Timestamp], float] = {}
	for _, row in market_df[["asset", "ts", "mid"]].iterrows():
		market_lookup[(row["asset"], row["ts"])] = float(row["mid"])

	rows = []
	for _, o in orders_df.iterrows():
		parent_id = o["parent_id"]
		asset = o["asset"]
		side = o["side"]
		total_qty = int(o["qty"])
		arrival = pd.Timestamp(o["ts_arrival"])  # ensure TS type

		# Child schedule 15–45 minutes after arrival
		schedule_len_min = int(rng.integers(15, 46))
		# Many child prints per parent to reach ~10k–20k total child rows
		n_child = int(rng.integers(40, 91))
		offsets = np.sort(rng.integers(0, schedule_len_min + 1, size=n_child))
		child_ts = [arrival + timedelta(minutes=int(off)) for off in offsets]

		child_qtys = _child_qty_allocation(total_qty, n_child, rng)

		# Price dynamics: nearest minute mid, plus small drift (impact) and noise
		def mid_at(ts: pd.Timestamp) -> float:
			key = (asset, _nearest_minute(ts))
			# Fallback: if exact minute missing, try previous minute
			m = market_lookup.get(key)
			if m is None:
				key_prev = (asset, _nearest_minute(ts - timedelta(minutes=1)))
				m = market_lookup.get(key_prev, np.nan)
			return float(m)

		sign = 1.0 if side == "BUY" else -1.0
		start_mid = mid_at(arrival)
		if not np.isfinite(start_mid) or start_mid <= 0:
			# As a last resort, pick median mid for asset
			start_mid = float(
				np.nanmedian(market_df.loc[market_df.asset == asset, "mid"].to_numpy())
			)

		# Progress-based impact: up to ~5 bp over schedule, plus noise ~1 bp
		for i, ts in enumerate(child_ts):
			progress = (i + 1) / n_child
			base_mid = mid_at(ts)
			if not np.isfinite(base_mid) or base_mid <= 0:
				base_mid = start_mid
			impact_bp = 3.0 + rng.uniform(-1.0, 1.5)  # around ~3bp
			impact = base_mid * (impact_bp / 10000.0) * sign * progress
			noise = rng.normal(0.0, base_mid * (0.8 / 10000.0))
			price = max(0.01, base_mid + impact + noise)

			rows.append(
				{
					"parent_id": parent_id,
					"ts": ts,
					"price": float(price),
					"qty": int(child_qtys[i]),
					"venue": rng.choice(CHILD_VENUES, p=[0.8, 0.2]),
					"order_type": rng.choice(ORDER_TYPES, p=[0.6, 0.3, 0.1]),
				}
			)

	child_df = pd.DataFrame(rows)
	return child_df


# ---------------------------- I/O helpers -----------------------------------

def write_csv(df: pd.DataFrame, path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(path, index=False)


# ---------------------------- Orchestration ---------------------------------

def generate_all() -> None:
	paths = get_paths()
	rng = get_rng()

	# Dates
	days = get_last_business_days(5)

	# Market first, so fills can reference mid prices
	market_df = simulate_market(days, rng)

	# Signals and derived entities
	signals_df = simulate_signals(days, rng)
	orders_df = simulate_orders(signals_df, rng)
	child_df = simulate_child_fills(orders_df, market_df, rng)

	# Write outputs
	write_csv(signals_df, paths.data_dir / "signals.csv")
	write_csv(orders_df, paths.data_dir / "orders.csv")
	write_csv(child_df, paths.data_dir / "child_fills.csv")
	write_csv(market_df, paths.data_dir / "market.csv")


if __name__ == "__main__":
	generate_all()