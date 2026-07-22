"""
Parse the stacked two-region Algerian Forest Fires CSV into a clean, region-tagged, labelled
calendar. The file's weather and FWI columns are kept for LABELS + AUDIT only -- they are never
model features.

Parsing gotchas handled: two regions stacked with a title row + blank line between them, a
repeated header, stray whitespace in column names, trailing spaces + inconsistent case in
`Classes`, and one malformed/shifted numeric row (found via to_numeric(coerce) and dropped).
Result: 243 clean rows (122 Bejaia + 121 Sidi-Bel Abbes), fire base rate ~0.564.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CSV_PATH = Path(__file__).parent / "data" / "Algerian_forest_fires_dataset_UPDATE.csv"
_NUM_COLS = ["day", "month", "year", "Temperature", "RH", "Ws", "Rain",
             "FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]


def load_fire_labels(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """One row per labelled region-day: Region, date, label (1=fire), plus the raw noon weather
    and FWI columns (audit/diagnostic use only)."""
    lines = csv_path.read_text(encoding="utf-8", errors="replace").splitlines()

    blocks, region, header, rows = [], None, None, []

    def flush():
        nonlocal rows
        if region and rows:
            blocks.append((region, header, rows))
        rows = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if "Region Dataset" in line:
            flush()
            region = line.replace("Region Dataset", "").strip()
            header = None
        elif line.lower().startswith("day,"):
            header = [c.strip() for c in line.split(",")]
        elif header is not None:
            rows.append(line.split(","))
    flush()

    frames = []
    for reg, hdr, recs in blocks:
        f = pd.DataFrame(recs, columns=hdr)
        f.columns = [c.strip() for c in f.columns]
        f["Region"] = reg
        frames.append(f)
    full = pd.concat(frames, ignore_index=True)

    full["Classes"] = full["Classes"].astype(str).str.strip().str.lower()
    for c in _NUM_COLS:
        full[c] = pd.to_numeric(full[c], errors="coerce")

    before = len(full)
    good = full["Classes"].isin(["fire", "not fire"]) & full[_NUM_COLS].notna().all(axis=1)
    full = full[good].copy()
    dropped = before - len(full)

    full["label"] = (full["Classes"] == "fire").astype(int)
    full["date"] = pd.to_datetime(dict(year=full.year.astype(int), month=full.month.astype(int),
                                       day=full.day.astype(int)))
    full = full.sort_values(["Region", "date"]).reset_index(drop=True)
    full.attrs["dropped_rows"] = dropped
    return full


if __name__ == "__main__":
    df = load_fire_labels()
    print(f"clean rows: {len(df)} (dropped {df.attrs['dropped_rows']})")
    print(df.groupby("Region").agg(n=("label", "size"), fires=("label", "sum"),
                                   base_rate=("label", "mean"), start=("date", "min"),
                                   end=("date", "max")))
    print(f"overall fire base rate: {df['label'].mean():.3f}")
