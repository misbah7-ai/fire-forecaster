"""
Servable feature contract -- imported by BOTH training and live serving so feature drift is
impossible. The bundle persists FEATURES, RH_AGGREGATION and feature_ranges; the live pipeline
asserts the RH aggregation matches before it will serve.

RAW WEATHER ONLY. The four servable variables + a binary region flag. No Fire-Weather-Index
component is ever a model feature: FWI codes are computed fire-danger scores and the dataset's
label is essentially a threshold on them (see the label audit) -- feeding them in is circular.
The FWI codes appear in the app only as *context* and in the audit as *demonstration*.
"""
from __future__ import annotations

import pandas as pd

# The single RH daily aggregation, reproduced byte-for-byte at train and serve time.
# Daily MINIMUM of hourly relative_humidity_2m = the driest hour, the fire-relevant one.
RH_AGGREGATION = "min of hourly relative_humidity_2m per calendar day (tz Africa/Algiers)"

# The two -- and only two -- regions the model was trained on, with query coordinates.
REGIONS = {
    "Bejaia": {"lat": 36.7525, "lon": 5.0556},
    "Sidi-Bel Abbes": {"lat": 35.1978, "lon": -0.6308},
}

# Exact feature order. temp/RH/Ws/Rain are the raw daily weather; region_sidi is the region flag.
FEATURES = ["temp", "RH", "Ws", "Rain", "region_sidi"]

# The trained season (Algerian fire season). The app season-gates on these months.
SEASON_MONTHS = [6, 7, 8, 9]

# The raw weather columns as they arrive from the Open-Meteo fetchers.
WEATHER_COLS = ["temp", "RH", "Ws", "Rain"]


def engineer_features(weather_df: pd.DataFrame, region: str) -> pd.DataFrame:
    """Return exactly the FEATURES columns, in order, for one region's daily weather frame.

    `weather_df` needs columns temp (temperature_2m_max), RH (daily-min relative humidity),
    Ws (wind_speed_10m_max), Rain (precipitation_sum). The region flag is derived from `region`
    (not the data), so the same frame yields the right flag from either Open-Meteo endpoint.
    """
    if region not in REGIONS:
        raise ValueError(f"unknown region {region!r}; expected one of {list(REGIONS)}")
    missing = [c for c in WEATHER_COLS if c not in weather_df.columns]
    if missing:
        raise ValueError(f"weather frame missing required columns: {missing}")
    out = weather_df.copy()
    out["region_sidi"] = 1 if region == "Sidi-Bel Abbes" else 0
    return out[FEATURES]
