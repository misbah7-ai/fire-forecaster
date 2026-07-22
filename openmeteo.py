"""
Open-Meteo weather fetchers -- ONE module for training (ERA5 archive) and serving (forecast),
so the weather source is internally consistent between them (the project's core discipline).

  * ARCHIVE  -> historical retrain weather  -> archive-api.open-meteo.com  (ERA5 reanalysis)
  * FORECAST -> live 7-day forecast weather  -> api.open-meteo.com

DIFFERENT hostnames; never interchanged. Both return the identical tidy daily schema
(date, temp, RH, Ws, Rain) where temp=temperature_2m_max, Ws=wind_speed_10m_max,
Rain=precipitation_sum, and RH = daily MINIMUM of hourly relative_humidity_2m (one shared
aggregation function, so train and serve cannot diverge). CC BY 4.0.
"""
from __future__ import annotations

import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEZONE = "Africa/Algiers"
_DAILY = "temperature_2m_max,wind_speed_10m_max,precipitation_sum"
_HOURLY = "relative_humidity_2m"
_TIMEOUT = 30


class OpenMeteoError(RuntimeError):
    """HTTP / timeout / malformed-response problems, so callers never see a raw requests
    exception or a KeyError from a missing field."""


def _get(url, params):
    try:
        r = requests.get(url, params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise OpenMeteoError(f"Open-Meteo request failed: {e}") from e
    except ValueError as e:
        raise OpenMeteoError(f"Open-Meteo returned non-JSON: {e}") from e


def _daily_rh_min(payload):
    """Aggregate hourly relative_humidity_2m to a per-calendar-day MINIMUM (driest hour).
    The hourly time strings are local (we pass timezone), so the date prefix keys the day."""
    h = payload.get("hourly") or {}
    times, rh = h.get("time"), h.get("relative_humidity_2m")
    if not times or rh is None:
        raise OpenMeteoError("Open-Meteo response missing hourly relative_humidity_2m")
    frame = pd.DataFrame({"date": pd.to_datetime(pd.Series(times)).dt.normalize(),
                          "rh": pd.to_numeric(pd.Series(rh), errors="coerce")})
    return frame.groupby("date")["rh"].min()


def _assemble(payload):
    d = payload.get("daily") or {}
    for k in ("time", "temperature_2m_max", "wind_speed_10m_max", "precipitation_sum"):
        if k not in d:
            raise OpenMeteoError(f"Open-Meteo daily block missing {k!r}")
    df = pd.DataFrame({
        "date": pd.to_datetime(pd.Series(d["time"])).dt.normalize(),
        "temp": pd.to_numeric(pd.Series(d["temperature_2m_max"]), errors="coerce"),
        "Ws": pd.to_numeric(pd.Series(d["wind_speed_10m_max"]), errors="coerce"),
        "Rain": pd.to_numeric(pd.Series(d["precipitation_sum"]), errors="coerce"),
    })
    df = df.merge(_daily_rh_min(payload).rename("RH"), left_on="date", right_index=True,
                  how="left")
    return df[["date", "temp", "RH", "Ws", "Rain"]].sort_values("date").reset_index(drop=True)


def fetch_archive_daily(lat, lon, start_date, end_date):
    """Historical ERA5 daily weather for [start_date, end_date] inclusive (YYYY-MM-DD)."""
    return _assemble(_get(ARCHIVE_URL, {
        "latitude": lat, "longitude": lon, "start_date": start_date, "end_date": end_date,
        "daily": _DAILY, "hourly": _HOURLY, "timezone": TIMEZONE}))


def fetch_forecast_daily(lat, lon, forecast_days=7, past_days=0):
    """Live forecast daily weather: next `forecast_days` days (+ optional `past_days`)."""
    return _assemble(_get(FORECAST_URL, {
        "latitude": lat, "longitude": lon, "forecast_days": forecast_days,
        "past_days": past_days, "daily": _DAILY, "hourly": _HOURLY, "timezone": TIMEZONE}))
