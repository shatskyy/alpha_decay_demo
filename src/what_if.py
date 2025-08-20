"""What-if simulator.

Two modes:
- Predictive: perturb arrival-only features and score the regression model (fast)
- Structural: re-simulate fills with changed knobs and recompute labels (slow, preferred)

This module uses predictive mode by default for speed, but also supports structural re-simulation if
the caller provides CSV overrides (not wired from run_demo due to runtime).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import json
import numpy as np
import pandas as pd
import joblib
from . import simulate_data


@dataclass(frozen=True)
class Paths:
    project_root: Path
    data_dir: Path


def get_paths() -> Paths:
    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return Paths(project_root=project_root, data_dir=data_dir)


def _time_test(df: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(df["ts_signal"]).dt.normalize().sort_values().unique()
    last_day = dates[-1]
    return df.loc[pd.to_datetime(df["ts_signal"]).dt.normalize() == last_day].copy()


def _load_csv(paths: Paths, name: str, parse_dates: List[str] | None = None) -> pd.DataFrame:
    p = paths.data_dir / f"{name}.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, parse_dates=parse_dates or [])


def _nearest_before(series_ts: pd.Series, target: pd.Timestamp) -> float:
    sub = series_ts[series_ts.index.get_level_values("ts") <= target]
    if len(sub):
        return float(sub.iloc[-1])
    return float("nan")


def _compute_mid_refs(market: pd.DataFrame, asset: str, ts_signal: pd.Timestamp, horizon_min: int) -> Tuple[float, float]:
    m = market.loc[market["asset"] == asset, ["ts", "mid"]].copy()
    m["ts"] = pd.to_datetime(m["ts"]).dt.floor("min")
    m = m.set_index(["ts"]).sort_index()
    m_vals = m["mid"]
    m_vals.index = m_vals.index.set_names(["ts"])
    t0 = ts_signal.floor("min")
    t1 = t0 + pd.Timedelta(minutes=int(horizon_min))
    def last_before(t: pd.Timestamp) -> float:
        sub = m_vals.loc[:t]
        return float(sub.iloc[-1]) if len(sub) else float("nan")
    return last_before(t0), last_before(t1)


def _bump_urgency(row: pd.Series) -> Tuple[int, int]:
    med = int(row.get("urgency_MED", 0))
    high = int(row.get("urgency_HIGH", 0))
    # one-hot states: LOW=(0,0), MED=(1,0), HIGH=(0,1)
    if high == 1:
        return 0, 1
    if med == 1:
        return 0, 1  # MED -> HIGH
    return 1, 0  # LOW -> MED


def run() -> Path:
    paths = get_paths()
    feat_path = paths.data_dir / "features.parquet"
    cols_path = paths.data_dir / "feature_cols.json"
    model_path = paths.data_dir / "model_reg.pkl"
    if not (feat_path.exists() and cols_path.exists() and model_path.exists()):
        raise FileNotFoundError("Features or model artifacts missing. Run the pipeline first.")

    df = pd.read_parquet(feat_path)
    df["ts_signal"] = pd.to_datetime(df["ts_signal"])  # ensure datetime
    with cols_path.open("r") as f:
        feature_cols: List[str] = json.load(f)
    reg = joblib.load(model_path)

    df_test = _time_test(df)
    Xb = df_test[feature_cols].copy()

    # Scenario: lower participation cap if present
    X_cap = Xb.copy()
    if "participation_cap" in X_cap.columns:
        X_cap["participation_cap"] = np.clip(X_cap["participation_cap"].astype(float) - 5.0, 5.0, None)

    # Scenario: bump urgency one notch via dummies
    X_urg = Xb.copy()
    if "urgency_MED" in X_urg.columns and "urgency_HIGH" in X_urg.columns:
        bumped = df_test.apply(_bump_urgency, axis=1, result_type="expand")
        X_urg["urgency_MED"] = bumped[0].astype(int)
        X_urg["urgency_HIGH"] = bumped[1].astype(int)

    # Predict (fast, predictive what-if)
    base = reg.predict(Xb)
    cap_down = reg.predict(X_cap)
    urg_up = reg.predict(X_urg)

    out = pd.DataFrame(
        {
            "parent_id": df_test["parent_id"].values,
            "baseline_bps": base,
            "cap_down_bps": cap_down,
            "urgency_up_bps": urg_up,
            "delta_cap_down": cap_down - base,
            "delta_urgency_up": urg_up - base,
        }
    )
    # Structural re-simulation (slow): re-generate fills for scenario orders on test parents only
    try:
        orders = _load_csv(paths, "orders", parse_dates=["ts_arrival"]).set_index("parent_id")
        signals = _load_csv(paths, "signals", parse_dates=["ts_signal"]).set_index(["asset", "ts_signal"]) if (paths.data_dir / "signals.csv").exists() else pd.DataFrame()
        market = _load_csv(paths, "market", parse_dates=["ts"])  # mids already generated
        # Prepare scenario orders for test set
        test_ids = set(df_test["parent_id"].tolist())
        ord_test = orders.loc[orders.index.intersection(test_ids)].reset_index().copy()
        # Build scenarios
        scen_defs: Dict[str, pd.DataFrame] = {
            "cap_down": ord_test.assign(participation_cap=np.clip(ord_test["participation_cap"].astype(float) - 5.0, 5.0, None)),
            "urgency_up": ord_test.copy(),
        }
        # bump urgency tag
        if "urgency_tag" in scen_defs["urgency_up"].columns:
            m = {"LOW": "MED", "MED": "HIGH", "HIGH": "HIGH"}
            scen_defs["urgency_up"]["urgency_tag"] = scen_defs["urgency_up"]["urgency_tag"].map(m).fillna("MED")

        # Helper: compute scenario alpha_decay structurally
        def scenario_alpha(scen_orders: pd.DataFrame, label: str, seed: int = 1234, n_draws: int = 3) -> pd.DataFrame:
            # Average over a few seeds for stability
            vwap_list = []
            for k in range(n_draws):
                rng = simulate_data.get_rng(seed + k)
                child = simulate_data.simulate_child_fills(scen_orders, market, rng)
                vwap_k = child.groupby("parent_id").apply(lambda g: float((g["price"] * g["qty"]).sum() / max(1, g["qty"].sum())))
                vwap_list.append(vwap_k)
            vwap = pd.concat(vwap_list, axis=1).mean(axis=1).rename("vwap")
            # VWAP per parent
            vwap = (child.groupby("parent_id").apply(lambda g: float((g["price"] * g["qty"]).sum() / max(1, g["qty"].sum())))).rename("vwap")
            # Compute mid_at_signal/target
            rows = []
            for _, r in scen_orders.iterrows():
                pid = r["parent_id"]; asset = r["asset"]; ts0 = pd.to_datetime(r["ts_arrival"]).floor("min")
                mid0, mid1 = _compute_mid_refs(market, asset, ts0, int(df_test.set_index("parent_id").loc[pid, "horizon"]))
                rows.append({"parent_id": pid, "mid0": mid0, "mid1": mid1})
            mids = pd.DataFrame(rows).set_index("parent_id")
            res = pd.DataFrame(index=df_test.set_index("parent_id").index)
            res = res.join(vwap, how="left").join(mids, how="left")
            res[f"alpha_decay_{label}"] = 10000.0 * (res["vwap"] - res["mid0"]) / res["mid0"]
            return res[[f"alpha_decay_{label}"]]

        # Structural baseline vs original orders
        base_struct = scenario_alpha(ord_test.reset_index(), "baseline")
        res_cap = scenario_alpha(scen_defs["cap_down"], "cap_down")
        res_urg = scenario_alpha(scen_defs["urgency_up"], "urgency_up")
        out = out.merge(res_cap, left_on="parent_id", right_index=True, how="left").merge(res_urg, left_on="parent_id", right_index=True, how="left")
        out = out.merge(base_struct, left_on="parent_id", right_index=True, how="left")
        out["delta_cap_down_struct"] = out["alpha_decay_cap_down"] - out["alpha_decay_baseline"]
        out["delta_urgency_up_struct"] = out["alpha_decay_urgency_up"] - out["alpha_decay_baseline"]
    except Exception as e:
        print(f"What-if | structural re-sim skipped: {e}")

    out_path = paths.data_dir / "what_if.csv"
    out.to_csv(out_path, index=False)
    print(f"What-if | Saved scenario estimates to {out_path}")
    # Print small summary
    print(
        "What-if | Median deltas (bps): cap_down = {:.2f}, urgency_up = {:.2f}".format(
            float(np.nanmedian(out["delta_cap_down"])), float(np.nanmedian(out["delta_urgency_up"]))
        )
    )
    return out_path


if __name__ == "__main__":
    run()


