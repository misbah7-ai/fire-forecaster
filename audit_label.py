"""
LABEL AUDIT -- is the dataset's `Classes` an observed-fire record, or a thresholded fire-danger
index? We sweep every single threshold on each of the dataset's own FWI-system columns
(FFMC/DMC/DC/ISI/BUI/FWI) and measure how well one threshold reproduces `Classes`.

The inference: a ~98%-deterministic weather-index -> label mapping is incompatible with real
Mediterranean ignition, which is stochastic and overwhelmingly human-caused. So `Classes` is an
INDEX CLASS, not observed fire -- and feeding those indices in as features (as published work
does) is circular: the model reconstructs the labelling rule.

Values are recomputed here; nothing is hardcoded. Each result prints beside its expected value
and flags a mismatch > 0.01.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_prep import load_fire_labels

ROOT = Path(__file__).parent
INDICES = ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]
EXPECTED_ACC = {"FFMC": 0.983, "ISI": 0.979, "FWI": 0.942, "DMC": 0.860, "DC": 0.860,
                "BUI": 0.852}


def best_threshold(values: np.ndarray, y: np.ndarray):
    """Best single threshold for the rule (index >= t -> fire), by accuracy. Returns
    (threshold, accuracy, n_correct, predictions)."""
    order = np.unique(values)
    # candidate cuts = midpoints between consecutive unique values, plus the extremes
    cuts = np.concatenate([[order[0] - 1e-9], (order[:-1] + order[1:]) / 2, [order[-1] + 1e-9]])
    best = None
    for t in cuts:
        pred = (values >= t).astype(int)
        acc = (pred == y).mean()
        if best is None or acc > best[1]:
            best = (float(t), float(acc), int((pred == y).sum()), pred)
    return best


def main():
    pd.set_option("display.width", 200)
    df = load_fire_labels()
    y = df["label"].to_numpy()
    n = len(df)
    print(f"rows: {n}  fires: {y.sum()}  base rate: {y.mean():.3f}\n")

    print("=" * 78)
    print("SINGLE-THRESHOLD REPRODUCTION OF `Classes` (rule: index >= t -> fire)")
    print("=" * 78)
    print(f"{'index':6s} {'thr':>8s} {'accuracy':>9s} {'n_correct':>10s} {'errors':>7s}   "
          f"{'expected':>9s} {'flag':>8s}")
    rows = []
    best_overall = None
    for idx in INDICES:
        t, acc, ncorr, pred = best_threshold(df[idx].to_numpy(float), y)
        exp = EXPECTED_ACC[idx]
        flag = "OK" if abs(acc - exp) <= 0.01 else "CHECK"
        print(f"{idx:6s} {t:8.2f} {acc:9.3f} {ncorr:10d} {n-ncorr:7d}   {exp:9.3f} {flag:>8s}")
        rows.append({"index": idx, "threshold": t, "accuracy": acc, "n_correct": ncorr,
                     "errors": n - ncorr, "expected": exp})
        if best_overall is None or acc > best_overall[1]:
            best_overall = (idx, acc, t, pred)
    pd.DataFrame(rows).to_csv(ROOT / "data" / "audit_label_thresholds.csv", index=False)

    idx, acc, t, pred = best_overall
    print(f"\nBEST RULE: `{idx} >= {t:.2f} -> fire` reproduces Classes at "
          f"{acc:.3f} ({int((pred==y).sum())}/{n}, {n-int((pred==y).sum())} errors)")

    print(f"\n--- EXCEPTION ROWS for the best rule ({idx} >= {t:.2f}) ---")
    exc = df[pred != y][["Region", "date", "Temperature", "RH", "Ws", "Rain", idx, "Classes"]]
    exc = exc.assign(predicted=np.where(pred[pred != y] == 1, "fire", "not fire"))
    exc["date"] = exc["date"].dt.strftime("%Y-%m-%d")
    print(exc.to_string(index=False))

    print("\n--- CLASS-CONDITIONAL INDEX RANGES (min-max) ---")
    for i in INDICES:
        fire = df.loc[y == 1, i]
        notf = df.loc[y == 0, i]
        print(f"  {i:5s}  fire: [{fire.min():6.1f}, {fire.max():6.1f}]   "
              f"not-fire: [{notf.min():6.1f}, {notf.max():6.1f}]")
    fwf, fwn = df.loc[y == 1, "FWI"], df.loc[y == 0, "FWI"]
    print(f"\n  (expected FWI fire ~1.7-31.1, not-fire ~0.0-6.1 -> narrow OVERLAP, not a clean cut;"
          f" got fire [{fwf.min():.1f},{fwf.max():.1f}] not-fire [{fwn.min():.1f},{fwn.max():.1f}])")

    print("\n" + "=" * 78)
    print("INFERENCE: a single weather-index threshold reproduces `Classes` ~98% of the time.")
    print("Real Mediterranean ignition is stochastic and overwhelmingly human-caused and CANNOT")
    print("be that deterministic in weather. => `Classes` is a thresholded fire-DANGER index,")
    print("not an observed-fire record. Using FWI components as features to predict it is circular.")
    print("wrote data/audit_label_thresholds.csv")


if __name__ == "__main__":
    main()
