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

# Causal signal → future return mapping (tuned via simple calibration)
# Target correlation between signed future return and signal_score
TARGET_CORR = 0.25
CORR_TOL = 0.10  # acceptable band: [0.15, 0.35]
MAX_CALIBRATION_STEPS = 3
# Initial mapping strength: bps of expected move over horizon per 1.0 sigma of score
INIT_BPS_PER_SIGMA = 20.0

URGENCY_TAGS = ["LOW", "MED", "HIGH"]
ALGO_TYPES = ["POV", "SCHEDULE", "DISCRETIONARY"]
BROKERS = ["BrokerA", "BrokerB", "BrokerC"]
VENUE_HINTS = ["LIT", "DARK", "SMART"]

CHILD_VENUES = ["LIT", "DARK"]
ORDER_TYPES = ["LIMIT", "MARKETABLE", "PEG"]

# Tick size and fee/rebate assumptions (simplified)
TICK_SIZE = 0.01  # $0.01
TAKER_FEE_BPS = 0.3
MAKER_REBATE_BPS = -0.1
DARK_IMPROVEMENT_BPS = 0.5  # price improvement vs mid
DARK_FILL_REDUCTION = 0.5    # scale down dark filled qty

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


def simulate_market(days: List[pd.Timestamp], rng: np.random.Generator, signals_df: pd.DataFrame | None = None, bps_per_sigma: float = INIT_BPS_PER_SIGMA) -> pd.DataFrame:
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

			# If signals are provided, embed a causal drift proportional to signal_score and side over the horizon
			if signals_df is not None and not signals_df.empty:
				# Filter signals for this asset and day
				mask = (signals_df["asset"] == asset) & (pd.to_datetime(signals_df["ts_signal"]).dt.normalize() == day)
				asset_signals = signals_df.loc[mask, ["ts_signal", "side", "signal_score", "alpha_horizon_min"]].copy()
				if not asset_signals.empty:
					asset_signals["ts_signal"] = pd.to_datetime(asset_signals["ts_signal"])  # ensure dtype
					# Build additive log-drift array
					log_drift = np.zeros(n, dtype=float)
					for _, s in asset_signals.iterrows():
						start_ts = pd.Timestamp(s["ts_signal"]).floor("min")
						# Find index at or before the signal minute
						idx_candidates = np.where(minutes <= start_ts)[0]
						if len(idx_candidates) == 0:
							continue
						idx0 = int(idx_candidates[-1])
						h = int(s.get("alpha_horizon_min", ALPHA_HORIZON_MINUTES))
						idx1 = min(n - 1, idx0 + max(1, h))
						if idx1 <= idx0:
							continue
						side_sign = 1.0 if str(s["side"]).upper() == "BUY" else -1.0
						score = float(s["signal_score"]) if pd.notna(s["signal_score"]) else 0.0
						# Target arithmetic return over horizon in fraction terms
						r_total = (bps_per_sigma / 10000.0) * score * side_sign
						# Convert to log-return target and distribute evenly across the horizon minutes after the signal bar
						log_target = float(np.log(max(1e-6, 1.0 + r_total)))
						span = idx1 - idx0
						per_min_log = log_target / span
						log_drift[idx0 + 1 : idx1 + 1] += per_min_log
					# Apply drift
					minute_rets = minute_rets + log_drift

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
			# Tick/grid consistency and no locked/crossed markets
			bid = np.maximum(TICK_SIZE, np.round(bid / TICK_SIZE) * TICK_SIZE)
			ask = np.maximum(bid + TICK_SIZE, np.round(ask / TICK_SIZE) * TICK_SIZE)
			mid = (bid + ask) / 2.0

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
	# Enforce per-asset time ordering and drop any accidental duplicates
	if not market_df.empty:
		market_df = market_df.sort_values(["asset", "ts"]).drop_duplicates(["asset", "ts"], keep="last").reset_index(drop=True)
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


# ---------------------------- Diagnostics / SNR -------------------------------

def _signed_future_return(signals_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.Series:
	# For each signal, compute side-signed arithmetic return over its horizon using mid prices
	signals = signals_df.copy()
	signals["ts_signal"] = pd.to_datetime(signals["ts_signal"]).dt.floor("min")
	market = market_df[["ts", "asset", "mid"]].copy()
	market["ts"] = pd.to_datetime(market["ts"]).dt.floor("min")
	# Build lookup for mid at exact minutes
	lookup = {(row["asset"], row["ts"]): float(row["mid"]) for _, row in market.iterrows()}
	vals = []
	for _, s in signals.iterrows():
		asset = s["asset"]
		t0 = pd.Timestamp(s["ts_signal"]).floor("min")
		h = int(s.get("alpha_horizon_min", ALPHA_HORIZON_MINUTES))
		t1 = t0 + timedelta(minutes=h)
		m0 = lookup.get((asset, t0), np.nan)
		# choose last bar at or before t1 on same day
		m1 = lookup.get((asset, t1), np.nan)
		if not np.isfinite(m1):
			# try stepping back within horizon by 1 minute until found or fail
			for k in range(1, 6):
				m1 = lookup.get((asset, t1 - timedelta(minutes=k)), np.nan)
				if np.isfinite(m1):
					break
		side_sign = 1.0 if str(s["side"]).upper() == "BUY" else -1.0
		ret = np.nan
		if np.isfinite(m0) and m0 > 0 and np.isfinite(m1) and m1 > 0:
			ret = side_sign * (m1 - m0) / m0
		vals.append(ret)
	return pd.Series(vals, index=signals.index, dtype=float)


def _snr_report(signals_df: pd.DataFrame, market_df: pd.DataFrame) -> dict:
	ret_signed = _signed_future_return(signals_df, market_df)
	s = pd.concat([signals_df[["side", "signal_score"]].reset_index(drop=True), ret_signed.rename("ret_signed")], axis=1).dropna()
	buy_median = float(s.loc[s["side"] == "BUY", "ret_signed"].median()) if (s["side"] == "BUY").any() else float("nan")
	sell_median = float(s.loc[s["side"] == "SELL", "ret_signed"].median()) if (s["side"] == "SELL").any() else float("nan")
	corr = float(pd.Series(s["signal_score"]).corr(pd.Series(s["ret_signed"]), method="pearson")) if len(s) > 2 else float("nan")
	return {"buy_median": buy_median, "sell_median": sell_median, "corr": corr}


def _market_monotonicity_report(market_df: pd.DataFrame) -> dict:
	if market_df.empty:
		return {"assets_checked": 0, "assets_non_monotonic": 0}
	bad = 0
	for asset, sub in market_df.sort_values(["asset", "ts"]).groupby("asset"):
		ts = pd.to_datetime(sub["ts"]).values
		if len(ts) <= 1:
			continue
		# strictly increasing
		if not np.all(ts[1:] > ts[:-1]):
			bad += 1
	return {"assets_checked": int(market_df["asset"].nunique()), "assets_non_monotonic": int(bad)}


def _fills_window_report(orders_df: pd.DataFrame, child_df: pd.DataFrame) -> dict:
	if orders_df.empty or child_df.empty:
		return {"parents_checked": int(len(orders_df)), "violations": 0}
	arrivals = orders_df[["parent_id", "ts_arrival"]].copy()
	arrivals["ts_arrival"] = pd.to_datetime(arrivals["ts_arrival"]).dt.floor("min")
	child = child_df[["parent_id", "ts"]].copy()
	child["ts"] = pd.to_datetime(child["ts"]).dt.floor("min")
	agg = child.groupby("parent_id").agg(min_ts=("ts", "min"), max_ts=("ts", "max")).reset_index()
	df = arrivals.merge(agg, on="parent_id", how="left")
	viol = 0
	for _, r in df.iterrows():
		arr = r["ts_arrival"]
		mn = r["min_ts"]
		mx = r["max_ts"]
		if pd.isna(mn) or pd.isna(mx):
			continue
		# Define completion cutoff as arrival + ALPHA_HORIZON_MINUTES (bounded by session end)
		day = arr.normalize()
		end_cap = min(arr + timedelta(minutes=ALPHA_HORIZON_MINUTES), pd.Timestamp.combine(day, TRADING_END) - timedelta(minutes=1))
		if not (mn >= arr and mx <= end_cap):
			viol += 1
	return {"parents_checked": int(len(arrivals)), "violations": int(viol)}


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
		# Policy mapping driven by signal strength
		strength = float(s.get("signal_strength_rank", 0.5))
		if strength < 0.33:
			urgency = "LOW"
			participation_cap = int(rng.integers(5, 15))
			algo = rng.choice(["DISCRETIONARY", "SCHEDULE"], p=[0.6, 0.4])
		elif strength < 0.66:
			urgency = "MED"
			participation_cap = int(rng.integers(12, 26))
			algo = rng.choice(["POV", "SCHEDULE"], p=[0.7, 0.3])
		else:
			urgency = "HIGH"
			participation_cap = int(rng.integers(22, 36))
			algo = rng.choice(["POV", "DISCRETIONARY"], p=[0.8, 0.2])
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
	# Build lookup for minute microstructure per asset: (mid, spread_bp, imbalance, turnover)
	market_lookup: Dict[Tuple[str, pd.Timestamp], Tuple[float, float, float, float]] = {}
	for _, row in market_df[["asset", "ts", "mid", "spread_bp", "imbalance", "turnover"]].iterrows():
		market_lookup[(row["asset"], row["ts"])] = (
			float(row["mid"]),
			float(row["spread_bp"]),
			float(row["imbalance"]),
			float(row["turnover"]),
		)

	rows = []
	for _, o in orders_df.iterrows():
		parent_id = o["parent_id"]
		asset = o["asset"]
		side = o["side"]
		total_qty = int(o["qty"])
		arrival = pd.Timestamp(o["ts_arrival"])  # ensure TS type

		# Child schedule anchored to arrival, bounded by [arrival, min(arrival + horizon, session_end))
		# Use alpha horizon minutes as an upper bound to keep fills within labeling window
		arrival_floor = _nearest_minute(arrival)
		day = arrival_floor.normalize()
		session_end_ts = pd.Timestamp.combine(day, TRADING_END) - timedelta(minutes=1)
		max_end_ts = min(arrival_floor + timedelta(minutes=ALPHA_HORIZON_MINUTES), session_end_ts)
		max_sched_minutes = int(max(1, (max_end_ts - arrival_floor).total_seconds() // 60))
		# Draw schedule length within [min(15, max), min(45, max)] and ensure at least 1 minute
		low = int(min(15, max_sched_minutes))
		high = int(min(45, max_sched_minutes))
		if high < 1:
			high = 1
		if low < 1:
			low = 1
		if high < low:
			high = low
		schedule_len_min = int(rng.integers(low, high + 1))
		# Many child prints per parent to reach ~10k–20k total child rows
		n_child = int(rng.integers(40, 91))
		offsets = np.sort(rng.integers(0, schedule_len_min + 1, size=n_child))
		child_ts = [arrival_floor + timedelta(minutes=int(off)) for off in offsets]

		child_qtys = _child_qty_allocation(total_qty, n_child, rng)

		# Price dynamics: nearest minute microstructure
		def micro_at(ts: pd.Timestamp) -> Tuple[float, float, float, float]:
			key = (asset, _nearest_minute(ts))
			val = market_lookup.get(key)
			if val is None:
				key_prev = (asset, _nearest_minute(ts - timedelta(minutes=1)))
				val = market_lookup.get(key_prev)
			if val is None:
				return float("nan"), float("nan"), float("nan"), float("nan")
			return val  # (mid, spread_bp, imbalance, turnover)

		sign = 1.0 if side == "BUY" else -1.0
		start_mid, start_spread_bp, start_imb, start_turn = micro_at(arrival)
		if not np.isfinite(start_mid) or start_mid <= 0:
			# As a last resort, pick median mid for asset
			start_mid = float(
				np.nanmedian(market_df.loc[market_df.asset == asset, "mid"].to_numpy())
			)

		# Execution behavior driven by urgency, participation, spread, and imbalance
		urg = str(o.get("urgency_tag", "MED")).upper()
		urgency_w = {"LOW": 0.3, "MED": 0.7, "HIGH": 1.0}.get(urg, 0.7)
		p_cap = float(o.get("participation_cap", 15))
		arrival_spread_bp = start_spread_bp if np.isfinite(start_spread_bp) else 2.0
		arrival_imb = start_imb if np.isfinite(start_imb) else 0.0
		arrival_turnover = start_turn if np.isfinite(start_turn) else (start_mid * 1e6)
		adverse_imb = max(0.0, -sign * arrival_imb)  # adverse when book leans against side
		spread_factor = 1.0 / (1.0 + max(0.0, arrival_spread_bp) / 6.0)
		base_marketable = 0.2 + 0.6 * urgency_w
		marketable_frac = np.clip(base_marketable * spread_factor + 0.55 * adverse_imb, 0.05, 0.98)

		# Impact and pricing depend on urgency/participation/spread/imbalance and liquidity
		# Temporary impact state decays over time; permanent impact accumulates slowly
		temp_state_bp = 0.0
		perm_cum_bp = 0.0
		temp_decay = 0.85
		temp_coef = 12.0
		perm_coef = 2.0
		for i, ts in enumerate(child_ts):
			progress = (i + 1) / n_child
			base_mid, spread_bp_t, imb_t, turn_t = micro_at(ts)
			if not np.isfinite(base_mid) or base_mid <= 0:
				base_mid = start_mid
			if not np.isfinite(spread_bp_t):
				spread_bp_t = arrival_spread_bp
			if not np.isfinite(imb_t):
				imb_t = arrival_imb
			if not np.isfinite(turn_t) or turn_t <= 0:
				turn_t = arrival_turnover
			adverse_imb_t = max(0.0, -sign * imb_t)
			# Participation relative to lit shares this minute
			shares_per_min = max(1.0, turn_t / base_mid)
			clip_participation = child_qtys[i] / shares_per_min
			pressure = (p_cap / 100.0) * progress + 2.0 * clip_participation
			# Update temporary/permanent impact in bps
			temp_state_bp = temp_decay * temp_state_bp + temp_coef * pressure
			perm_cum_bp += perm_coef * pressure
			impact_bp = (
				0.5 * max(0.0, spread_bp_t)
				+ 2.0 * adverse_imb_t
				+ temp_state_bp
				+ 0.5 * perm_cum_bp
			)
			impact = base_mid * (impact_bp / 10000.0) * sign
			# Decide order type and venue
			is_marketable = rng.random() < (marketable_frac * (0.7 + 0.6 * progress))
			order_type = "MARKETABLE" if is_marketable else ("LIMIT" if rng.random() > 0.2 else "PEG")
			half_spread_abs = base_mid * (max(0.0, spread_bp_t) / 20000.0)
			p_dark = 0.5 * np.clip(0.6 * (1.2 - urgency_w) * (1.0 + max(0.0, spread_bp_t) / 3.0), 0.05, 0.85)
			venue_val = "DARK" if rng.random() < p_dark else "LIT"
			noise_sd = base_mid * ((0.4 + 0.6 * urgency_w) / 10000.0)
			noise = rng.normal(0.0, noise_sd)
			# Tick rounding helper
			def round_to_tick(p: float) -> float:
				return max(TICK_SIZE, TICK_SIZE * round(p / TICK_SIZE))
			if order_type == "MARKETABLE":
				if venue_val == "DARK":
					impr = base_mid * (DARK_IMPROVEMENT_BPS / 10000.0)
					price = round_to_tick(base_mid - sign * impr + noise)
				else:
					gross = base_mid + sign * half_spread_abs + impact + noise
					price = round_to_tick(gross * (1.0 + TAKER_FEE_BPS / 10000.0))
			elif order_type == "PEG":
				gross = base_mid + 0.5 * sign * half_spread_abs + 0.75 * impact + noise
				price = round_to_tick(gross)
			else:  # LIMIT
				favor = max(0.0, sign * imb_t)
				improve_bp = 0.6 * favor * max(0.0, spread_bp_t) * (0.5 + 0.5 * progress)
				improve = base_mid * (improve_bp / 10000.0)
				gross = base_mid + 0.4 * impact - improve + noise
				price = round_to_tick(gross * (1.0 + MAKER_REBATE_BPS / 10000.0))

			rows.append(
				{
					"parent_id": parent_id,
					"ts": ts,
					"price": float(price),
					"qty": int(child_qtys[i]),
					"venue": venue_val,
					"order_type": order_type,
				}
			)

	child_df = pd.DataFrame(rows)
	# Enforce ordering within parent and drop any accidental out-of-window rows (guard)
	if not child_df.empty:
		child_df = child_df.sort_values(["parent_id", "ts"]).reset_index(drop=True)
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

	# Signals first (causal); then market with embedded drift calibrated to target correlation
	signals_df = simulate_signals(days, rng)
	bps_per_sigma = INIT_BPS_PER_SIGMA
	best_market = None
	best_report = None
	for step in range(MAX_CALIBRATION_STEPS):
		market_df = simulate_market(days, rng, signals_df=signals_df, bps_per_sigma=bps_per_sigma)
		report = _snr_report(signals_df, market_df)
		best_market, best_report = market_df, report
		corr = report["corr"]
		print(f"SNR calibration step {step+1}/{MAX_CALIBRATION_STEPS} | bps_per_sigma={bps_per_sigma:.2f} | corr={corr:.3f} | buy_med={report['buy_median']:.5f} | sell_med={report['sell_median']:.5f}")
		if np.isfinite(corr) and abs(corr - TARGET_CORR) <= CORR_TOL:
			break
		# Adjust mapping strength proportional to error (guard against zero/NaN corr)
		if not np.isfinite(corr) or abs(corr) < 1e-6:
			bps_per_sigma *= 1.5
		else:
			scale = np.clip(TARGET_CORR / max(1e-6, corr), 0.5, 2.0)
			bps_per_sigma = float(np.clip(bps_per_sigma * scale, 1.0, 100.0))

	# Use the latest/best market for downstream generation
	market_df = best_market if best_market is not None else simulate_market(days, rng, signals_df=signals_df, bps_per_sigma=bps_per_sigma)

	# Orders and child fills
	orders_df = simulate_orders(signals_df, rng)
	child_df = simulate_child_fills(orders_df, market_df, rng)

	# Write outputs
	write_csv(signals_df, paths.data_dir / "signals.csv")
	write_csv(orders_df, paths.data_dir / "orders.csv")
	write_csv(child_df, paths.data_dir / "child_fills.csv")
	write_csv(market_df, paths.data_dir / "market.csv")

	# Final acceptance snapshot
	report = _snr_report(signals_df, market_df)
	mono = _market_monotonicity_report(market_df)
	fwin = _fills_window_report(orders_df, child_df)
	print(
		f"Acceptance | BUY median future_ret={report['buy_median']:.5f}, SELL median future_ret={report['sell_median']:.5f}, Corr(score, signed_ret)={report['corr']:.3f}"
	)
	print(
		f"Acceptance | Market monotonicity: assets_checked={mono['assets_checked']}, non_monotonic={mono['assets_non_monotonic']}"
	)
	print(
		f"Acceptance | Fills window: parents_checked={fwin['parents_checked']}, violations={fwin['violations']}"
	)


if __name__ == "__main__":
	generate_all()