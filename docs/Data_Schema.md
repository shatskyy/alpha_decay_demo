# Data Schema Reference

## CSV outputs (from simulate_data.py)

### data/signals.csv

- ts_signal (datetime): hourly signal time (10:00–15:00)
- asset (string): ticker (AAPL, MSFT, AMZN, GOOG, META)
- side (string): BUY/SELL
- signal_score (float): ~N(0,1) score
- alpha_horizon_min (int): horizon in minutes (60)
- signal_strength_rank (float): [0,1]

### data/orders.csv

- parent_id (string): SHA1(asset|ts_signal)
- ts_arrival (datetime): arrival time (equals ts_signal)
- asset (string)
- side (string)
- qty (int): lognormal size (clipped)
- urgency_tag (string): LOW/MED/HIGH
- algo_type (string): POV/SCHEDULE/DISCRETIONARY
- participation_cap (int): [5,35]
- broker (string)
- venue_hint (string): LIT/DARK/SMART

### data/child_fills.csv

- parent_id (string)
- ts (datetime): child fill timestamp
- price (float): execution price
- qty (int): child fill quantity
- venue (string): LIT/DARK
- order_type (string): LIMIT/MARKETABLE/PEG

### data/market.csv

- ts (datetime): minute bar timestamp (09:30–16:00)
- asset (string)
- mid, bid, ask (float)
- spread_bp (float): bid-ask spread in bps
- depth1_bid, depth1_ask (int)
- imbalance (float): [-1,1]
- rv_5m, rv_30m (float): rolling vol proxies (>0)
- adv (int): estimated ADV shares
- turnover (float): per-minute turnover in notional

## SQLite database (db/alpha.sqlite)

Tables created in `ingest.py`:

### signals

- ts_signal TEXT NOT NULL
- asset TEXT NOT NULL
- side TEXT NOT NULL
- signal_score REAL NOT NULL
- alpha_horizon_min INTEGER NOT NULL
- signal_strength_rank REAL NOT NULL
- Index: idx_signals_asset_ts (asset, ts_signal)

### orders

- parent_id TEXT PRIMARY KEY
- ts_arrival TEXT NOT NULL
- asset TEXT NOT NULL
- side TEXT NOT NULL
- qty INTEGER NOT NULL
- urgency_tag TEXT NOT NULL
- algo_type TEXT NOT NULL
- participation_cap INTEGER NOT NULL
- broker TEXT NOT NULL
- venue_hint TEXT NOT NULL
- Indexes: idx_orders_asset_ts (asset, ts_arrival), idx_orders_parent_id (parent_id)

### child_fills

- parent_id TEXT NOT NULL
- ts TEXT NOT NULL
- price REAL NOT NULL
- qty INTEGER NOT NULL
- venue TEXT NOT NULL
- order_type TEXT NOT NULL
- Index: idx_child_parent (parent_id)

### market

- ts TEXT NOT NULL
- asset TEXT NOT NULL
- mid REAL NOT NULL
- bid REAL NOT NULL
- ask REAL NOT NULL
- spread_bp REAL NOT NULL
- depth1_bid INTEGER NOT NULL
- depth1_ask INTEGER NOT NULL
- imbalance REAL NOT NULL
- rv_5m REAL NOT NULL
- rv_30m REAL NOT NULL
- adv INTEGER NOT NULL
- turnover REAL NOT NULL
- Index: idx_market_asset_ts (asset, ts)

## Modeling tables

### data/labels.parquet

- parent_id, asset, side, ts_signal, ts_arrival
- horizon (int)
- vwap_exec (float)
- r_sig (float, bps), r_exec (float, bps)
- alpha_decay (float, bps), decay_flag (int: 0/1)

### data/features.parquet

- Meta: parent_id, asset, side, ts_signal, ts_arrival, horizon, vwap_exec, r_sig, r_exec, alpha_decay, decay_flag, open_close_bucket
- Numeric features: age_sec, exec_dur_sec, horizon_to_age, spread_bp, imbalance, rv_5m, rv_30m, depth1_bid, depth1_ask, child_qty, lit_volume_est, participation_est, pct_dark, pct_marketable, reprice_rate, minute_of_day, signal_score, signal_strength_rank, side_sign
