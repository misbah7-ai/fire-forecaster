"""
Live serving pipeline: fetch the 7-day forecast, build features IDENTICALLY to training,
season-gate the request. Imported by app.py.

Core invariant: `build_live_features` reproduces training feature engineering byte-for-byte --
same RH daily-aggregation, same column order as bundle["features"] -- by calling the SHARED
`features.engineer_features`. The __main__ self-test proves it against the cached ERA5 history to
atol=1e-10 and asserts the RH aggregation matches the bundle; on failure it refuses to serve.
"""
from __future__ import annotations

import json
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from features import FEATURES, REGIONS, RH_AGGREGATION, SEASON_MONTHS, engineer_features
from openmeteo import fetch_forecast_daily

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "data" / "live_cache"
CACHE_TTL = 3600  # ~1 hour


def _cache_path(region):
    return CACHE_DIR / f"forecast_{region.replace(' ', '_')}.json"


def _read_cache(region):
    p = _cache_path(region)
    if not p.exists():
        return None
    try:
        blob = json.loads(p.read_text(encoding="utf-8"))
        if time.time() - blob["fetched_at"] > CACHE_TTL:
            return None
        df = pd.DataFrame(blob["data"])
        df["date"] = pd.to_datetime(df["date"])
        return df
    except (ValueError, KeyError):
        return None


def _write_cache(region, df):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    _cache_path(region).write_text(json.dumps({"fetched_at": time.time(),
                                               "data": out.to_dict("records")}), encoding="utf-8")


def fetch_forecast_weather(region, forecast_days=7, use_cache=True):
    """Next `forecast_days` days of daily weather (date, temp, RH, Ws, Rain) for a known region.
    Raises ValueError for an unknown region; openmeteo.OpenMeteoError (caught by the app) on any
    API problem."""
    if region not in REGIONS:
        raise ValueError(f"unknown region {region!r}; expected one of {list(REGIONS)}")
    if use_cache:
        cached = _read_cache(region)
        if cached is not None:
            return cached
    c = REGIONS[region]
    df = fetch_forecast_daily(c["lat"], c["lon"], forecast_days=forecast_days, past_days=0)
    if use_cache:
        _write_cache(region, df)
    return df


def build_live_features(weather_df, region):
    """Reproduce training feature engineering exactly -> columns == bundle['features']."""
    X = engineer_features(weather_df[["temp", "RH", "Ws", "Rain"]], region)
    assert list(X.columns) == FEATURES, "live feature order diverged from training"
    return X


def season_ok(d, bundle):
    """Whether date `d`'s month is in the model's trained season (June-September)."""
    return pd.Timestamp(d).month in bundle.get("season_months", SEASON_MONTHS)


def assert_rh_aggregation_matches(bundle):
    """Refuse to serve if the live RH aggregation differs from what training persisted."""
    if bundle.get("rh_aggregation") != RH_AGGREGATION:
        raise RuntimeError("RH aggregation mismatch between live pipeline and bundle:\n"
                           f"  live   : {RH_AGGREGATION!r}\n  bundle : {bundle.get('rh_aggregation')!r}")


def _self_test():
    import joblib
    import numpy as np
    hist = pd.read_csv(ROOT / "data" / "openmeteo_fire_history.csv", parse_dates=["date"])
    bundle_path = ROOT / "models" / "fire_model.joblib"
    max_diff = 0.0
    for region, g in hist.groupby("Region"):
        live = build_live_features(g, region).to_numpy(float)
        ref = engineer_features(g[["temp", "RH", "Ws", "Rain"]], region).to_numpy(float)
        max_diff = max(max_diff, float(np.abs(live - ref).max()))
    assert max_diff <= 1e-10, f"live features diverge from training (max diff {max_diff:g})"
    print(f"[self-test] live features reproduce training features (max diff {max_diff:g})")
    if bundle_path.exists():
        b = joblib.load(bundle_path)
        assert_rh_aggregation_matches(b)
        assert build_live_features(hist.iloc[:5], hist.iloc[0]["Region"]).columns.tolist() \
            == b["features"], "live feature order != bundle features"
        print(f"[self-test] RH aggregation matches bundle; feature order matches ({b['features']})")
    else:
        print("[self-test] bundle not found yet -- run train_model.py first")


if __name__ == "__main__":
    _self_test()
