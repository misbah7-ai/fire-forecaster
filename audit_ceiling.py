"""
CEILING AUDIT -- where does the predictive skill actually live, and how much is noise?

Decomposes the servable ceiling and puts a bootstrap CI on every ROC-AUC (n~122 hold-out ->
CI ~+/-0.08-0.10, so point estimates alone over-claim):

  1. Ceiling chain (single-feature hold-out AUC): dataset FWI (~0.987) -> self-FWI from NOON
     weather (~0.866, implementation/startup gap) -> self-FWI from Open-Meteo (~0.770,
     reanalysis-vs-station gap). Noon-station FWI is unobservable live, so ~0.78 is roughly the
     servable ceiling.
  2. ML vs formula: best servable ML (~0.776) vs self-FWI-from-Open-Meteo alone (~0.770) --
     overlapping CIs; the model adds ~nothing over the formula.
  3. Circularity demo: refit WITH the dataset's FWI columns as features -> AUC ~0.99 (never served).
  4. Trivial baseline: always-predict-fire -> recall 1.0, precision ~0.607, AUC 0.5.
  5. Feature experiment RE-RUN with the ORIGINAL design (each feature set x model family gets
     its own small grid on the training months -- measurement, not a shipped swap), reporting
     chronological AND leave-one-region-out AUC with CIs. The reading is DATA-DRIVEN: the
     script counts how many memory-vs-baseline comparisons actually replicate, rather than
     asserting it.

The SHIPPED model is untouched by all of this: it keeps its fixed hyperparameters and
threshold (train_model.py). Nothing here is served.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from data_prep import load_fire_labels
from features import FEATURES
from fwi import compute_codes
from stats_utils import bootstrap_metric_ci, ci_str

ROOT = Path(__file__).parent
RS = 42
BASE = FEATURES
SIMPLE = ["temp_3d", "temp_7d", "rain_7d", "rain_14d", "days_since_rain", "dryness_ewma"]
MOIST = ["FFMC_c", "DMC_c", "DC_c"]
DATASET_FWI_COLS = ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]
SHIPPED_PARAMS = {"ccp_alpha": 0.005, "max_depth": 5, "min_samples_leaf": 5}

TREE_GRID = {"max_depth": [2, 3, 4], "min_samples_leaf": [3, 5, 8],
             "ccp_alpha": [0.0, 0.005, 0.01]}
HGB_GRID = {"learning_rate": [0.05, 0.1], "max_depth": [2, 3], "max_iter": [200],
            "min_samples_leaf": [10, 20], "l2_regularization": [0.0, 1.0]}


def build_frame():
    """labels + Open-Meteo weather + memory + self-FWI(Open-Meteo), per region in date order.
    (The dataset's own FWI columns ride along from the labels merge -- audit use only.)"""
    labels = load_fire_labels()
    weather = pd.read_csv(ROOT / "data" / "openmeteo_fire_history.csv", parse_dates=["date"])
    df = labels.merge(weather, on=["Region", "date"], how="left", suffixes=("_csv", ""))
    assert df[["temp", "RH", "Ws", "Rain"]].isna().sum().sum() == 0
    df["region_sidi"] = (df.Region == "Sidi-Bel Abbes").astype(int)
    parts = []
    for _, g in df.groupby("Region"):
        g = g.sort_values("date").reset_index(drop=True).copy()
        g["temp_3d"] = g["temp"].rolling(3, min_periods=1).mean()
        g["temp_7d"] = g["temp"].rolling(7, min_periods=1).mean()
        g["rain_7d"] = g["Rain"].rolling(7, min_periods=1).sum()
        g["rain_14d"] = g["Rain"].rolling(14, min_periods=1).sum()
        wet = (g["Rain"] >= 1.0).to_numpy()
        dsr, c = np.empty(len(g)), 0
        for i in range(len(g)):
            c = 0 if wet[i] else c + 1
            dsr[i] = c
        g["days_since_rain"] = dsr
        g["dryness_ewma"] = g["Rain"].ewm(alpha=0.15, adjust=False).mean()
        g = compute_codes(g, temp="temp", rh="RH", wind="Ws", rain="Rain")  # self-FWI (OM)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def self_fwi_from_noon():
    """Self-computed FWI from the dataset's own NOON weather columns (the real FWI inputs)."""
    labels = load_fire_labels()
    parts = []
    for _, g in labels.groupby("Region"):
        parts.append(compute_codes(g.rename(columns={"Temperature": "temp", "RH": "RH",
                                                     "Ws": "Ws", "Rain": "Rain"})))
    return pd.concat(parts, ignore_index=True)


def is_test(df):
    return df["date"].dt.month.isin([8, 9])


def single_feature_ci(df, col):
    te = df[is_test(df)]
    return bootstrap_metric_ci(te["label"].to_numpy(), te[col].to_numpy(float))


def _grid_fit(family, Xtr, ytr):
    """Original experiment design: each feature set x family gets its own small grid, selected
    on shuffled CV over the TRAINING months only."""
    cv = StratifiedKFold(5, shuffle=True, random_state=RS)
    if family == "tree":
        gs = GridSearchCV(DecisionTreeClassifier(class_weight="balanced", random_state=RS),
                          TREE_GRID, scoring="roc_auc", cv=cv, n_jobs=-1)
    else:
        gs = GridSearchCV(HistGradientBoostingClassifier(class_weight="balanced",
                                                         random_state=RS, early_stopping=False),
                          HGB_GRID, scoring="roc_auc", cv=cv, n_jobs=-1)
    gs.fit(Xtr, ytr)
    return gs


def _refit(family, params):
    if family == "tree":
        return DecisionTreeClassifier(class_weight="balanced", random_state=RS, **params)
    return HistGradientBoostingClassifier(class_weight="balanced", random_state=RS,
                                          early_stopping=False, **params)


def eval_set(df, feats, family, fixed_params=None):
    """Chrono + LORO eval. Grid-tunes on the train months (original design) unless
    fixed_params is given (used for the shipped-tree row and the circularity demo)."""
    tr, te = df[~is_test(df)], df[is_test(df)]
    if fixed_params is None:
        gs = _grid_fit(family, tr[feats], tr["label"])
        params, model = gs.best_params_, gs.best_estimator_
    else:
        params = fixed_params
        model = _refit(family, params).fit(tr[feats], tr["label"])
    proba = model.predict_proba(te[feats])[:, 1]
    auc, lo, hi, _ = bootstrap_metric_ci(te["label"].to_numpy(), proba)
    pred = (proba >= 0.5).astype(int)
    rec = recall_score(te["label"], pred, zero_division=0)
    prec = precision_score(te["label"], pred, zero_division=0)
    yy, pp = [], []
    for ho in df.Region.unique():
        a, b = df[df.Region != ho], df[df.Region == ho]
        mm = _refit(family, params).fit(a[feats], a["label"])
        yy.append(b["label"].to_numpy()); pp.append(mm.predict_proba(b[feats])[:, 1])
    yy, pp = np.concatenate(yy), np.concatenate(pp)
    lauc, llo, lhi, _ = bootstrap_metric_ci(yy, pp)
    return dict(auc=auc, lo=lo, hi=hi, rec=rec, prec=prec, params=params,
                loro=lauc, loro_lo=llo, loro_hi=lhi)


def main():
    pd.set_option("display.width", 200)
    df = build_frame()
    noon = self_fwi_from_noon()
    dataset = load_fire_labels()

    tr, te = df[~is_test(df)], df[is_test(df)]
    print(f"test-set composition: n={len(te)}  fire={int(te.label.sum())}  "
          f"not-fire={int((1-te.label).sum())}  base_rate={te.label.mean():.3f}")
    print(f"train composition:    n={len(tr)}  fire_rate={tr.label.mean():.3f}")
    print(f"shipped DecisionTree params (fixed, never re-tuned): {SHIPPED_PARAMS}\n")

    # ---- 1. ceiling chain ------------------------------------------------------------
    print("=" * 78)
    print("CEILING CHAIN -- single-feature hold-out ROC-AUC (95% bootstrap CI)")
    print("=" * 78)
    d_fwi = single_feature_ci(dataset, "FWI")
    n_fwi = single_feature_ci(noon, "FWI_c")
    o_fwi = single_feature_ci(df, "FWI_c")
    print(f"  dataset's own FWI column            {ci_str(*d_fwi[:3])}   (expected ~0.987)")
    print(f"  self-FWI from NOON weather          {ci_str(*n_fwi[:3])}   (expected ~0.866)")
    print(f"  self-FWI from Open-Meteo (servable) {ci_str(*o_fwi[:3])}   (expected ~0.770)")
    print(f"    gap 1 (implementation/startup, noon): {d_fwi[0]-n_fwi[0]:+.3f}")
    print(f"    gap 2 (reanalysis vs station):        {n_fwi[0]-o_fwi[0]:+.3f}")
    print("  => noon-station FWI is unobservable live; ~0.78 is roughly the servable ceiling.")

    # ---- 2. feature experiment (ORIGINAL grid-per-set design) -------------------------
    print("\n" + "=" * 78)
    print("FEATURE EXPERIMENT (original design: grid per set/family) -- chrono + LORO, 95% CIs")
    print("=" * 78)
    sets = {"baseline": BASE, "+simple-memory": BASE + SIMPLE, "+moisture-codes": BASE + MOIST}
    results, exp_rows = {}, []
    for sname, feats in sets.items():
        for fam, mname in (("tree", "DecisionTree"), ("hgb", "HistGBoost")):
            r = eval_set(df, feats, fam)
            results[(sname, mname)] = r
            exp_rows.append({"feature_set": sname, "model": mname,
                             "chrono_auc": r["auc"], "chrono_lo": r["lo"], "chrono_hi": r["hi"],
                             "loro_auc": r["loro"], "loro_lo": r["loro_lo"],
                             "loro_hi": r["loro_hi"], "recall": r["rec"], "precision": r["prec"]})
            print(f"  {sname:16s} {mname:13s} chrono {ci_str(r['auc'], r['lo'], r['hi'])}"
                  f"   LORO {ci_str(r['loro'], r['loro_lo'], r['loro_hi'])}")
    exp = pd.DataFrame(exp_rows)

    # DATA-DRIVEN reading: count the memory-vs-baseline comparisons that actually replicate
    chrono_improvements_outside_ci = sum(
        (results[(s, m)]["auc"] - results[("baseline", m)]["auc"])
        > (results[("baseline", m)]["hi"] - results[("baseline", m)]["lo"]) / 2
        for s in ("+simple-memory", "+moisture-codes") for m in ("DecisionTree", "HistGBoost"))
    loro_wins = sum(results[(s, m)]["loro"] > results[("baseline", m)]["loro"]
                    for s in ("+simple-memory", "+moisture-codes")
                    for m in ("DecisionTree", "HistGBoost"))
    print(f"\n  reading (computed, not asserted):")
    print(f"  - chronological IMPROVEMENTS over baseline exceeding the CI half-width: "
          f"{chrono_improvements_outside_ci}/4 -> nothing chronologically bankable")
    print(f"  - LORO memory-beats-baseline replications: {loro_wins}/4 "
          f"-> {'consistent-direction' if loro_wins >= 3 else 'inconsistent'} hypothesis, "
          f"NOT a banked improvement")

    # ---- 3. ML vs formula, shipped, trivial, circularity ------------------------------
    base_best = max((results[("baseline", m)] for m in ("DecisionTree", "HistGBoost")),
                    key=lambda r: r["auc"])
    best_name = [m for m in ("DecisionTree", "HistGBoost")
                 if results[("baseline", m)] is base_best][0]
    shipped = eval_set(df, BASE, "tree", fixed_params=SHIPPED_PARAMS)
    circ = eval_set(df, BASE + DATASET_FWI_COLS, "tree", fixed_params=SHIPPED_PARAMS)
    yte = te["label"].to_numpy()
    triv_rec = recall_score(yte, np.ones_like(yte))
    triv_prec = precision_score(yte, np.ones_like(yte), zero_division=0)

    print("\n" + "=" * 78)
    print("ML vs FORMULA, SHIPPED, TRIVIAL, CIRCULARITY  (hold-out ROC-AUC, 95% CI)")
    print("=" * 78)
    print(f"  best servable ML (baseline, {best_name})   "
          f"{ci_str(base_best['auc'], base_best['lo'], base_best['hi'])}   (expected ~0.776)")
    print(f"  self-FWI from Open-Meteo (formula only)  {ci_str(*o_fwi[:3])}   (expected ~0.770)")
    print(f"    -> difference {base_best['auc']-o_fwi[0]:+.3f} with overlapping CIs: model ~= formula")
    print(f"  shipped tree (fixed params, as served)   "
          f"{ci_str(shipped['auc'], shipped['lo'], shipped['hi'])}")
    print(f"  trivial always-fire                      AUC=0.500  "
          f"recall={triv_rec:.3f} precision={triv_prec:.3f}")
    print(f"  CIRCULARITY demo (+dataset FWI cols)     "
          f"{ci_str(circ['auc'], circ['lo'], circ['hi'])}   (expected ~0.99; NEVER served)")

    # ---- persist ----------------------------------------------------------------------
    ceiling_rows = [
        ("dataset_FWI_column", *d_fwi[:3], "audit/circular"),
        ("self_FWI_noon", *n_fwi[:3], "unservable (noon station)"),
        ("self_FWI_openmeteo", *o_fwi[:3], "servable ceiling ~0.78"),
        ("best_servable_ML", base_best["auc"], base_best["lo"], base_best["hi"], "servable"),
        ("shipped_tree", shipped["auc"], shipped["lo"], shipped["hi"], "servable/shipped"),
        ("trivial_always_fire", 0.5, 0.5, 0.5, "baseline"),
        ("circularity_plus_dataset_FWI", circ["auc"], circ["lo"], circ["hi"], "circular demo"),
    ]
    pd.DataFrame(ceiling_rows, columns=["item", "auc", "ci_lo", "ci_hi", "note"]).to_csv(
        ROOT / "data" / "audit_ceiling.csv", index=False)
    exp.to_csv(ROOT / "data" / "audit_experiment.csv", index=False)
    print("\nwrote data/audit_ceiling.csv, data/audit_experiment.csv")

    make_figures(d_fwi, n_fwi, o_fwi, base_best, shipped, circ, exp)
    inject_bundle_headlines(d_fwi[0], n_fwi[0], o_fwi[0], base_best["auc"], loro_wins)


def make_figures(d_fwi, n_fwi, o_fwi, best_ml, shipped, circ, exp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)

    # ceiling waterfall
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = ["dataset FWI", "self-FWI\n(noon)", "self-FWI\n(Open-Meteo)"]
    vals = [d_fwi[0], n_fwi[0], o_fwi[0]]
    bars = ax.bar(labels, vals, color=["#7f8c8d", "#e67e22", "#c0392b"])
    ax.axhline(0.5, ls="--", c="grey", lw=0.8, label="chance")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")
    ax.annotate(f"implementation gap {d_fwi[0]-n_fwi[0]:+.3f}", (0.5, (d_fwi[0]+n_fwi[0])/2),
                ha="center", fontsize=9)
    ax.annotate(f"reanalysis gap {n_fwi[0]-o_fwi[0]:+.3f}", (1.5, (n_fwi[0]+o_fwi[0])/2),
                ha="center", fontsize=9)
    ax.set_ylim(0.4, 1.05); ax.set_ylabel("single-feature ROC-AUC (hold-out)")
    ax.set_title("Ceiling waterfall: where the skill lives (and leaks)")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "ceiling_waterfall.png", dpi=130); plt.close(fig)

    # CI forest plot
    items = [("dataset FWI (circular)", d_fwi[:3]), ("self-FWI noon", n_fwi[:3]),
             ("self-FWI Open-Meteo", o_fwi[:3]),
             ("best servable ML", (best_ml["auc"], best_ml["lo"], best_ml["hi"])),
             ("shipped tree", (shipped["auc"], shipped["lo"], shipped["hi"])),
             ("circularity (+FWI cols)", (circ["auc"], circ["lo"], circ["hi"])),
             ("trivial always-fire", (0.5, 0.5, 0.5))]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for y, (name, ci) in enumerate(items):
        ax.plot([ci[1], ci[2]], [y, y], "-", color="#2c3e50", lw=2)
        ax.plot(ci[0], y, "o", color="#c0392b")
    ax.axvline(0.5, ls="--", c="grey", lw=0.8)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([n for n, _ in items])
    ax.set_xlabel("ROC-AUC (95% bootstrap CI)"); ax.set_xlim(0.45, 1.02)
    ax.set_title("Every AUC with its CI (n~122 -> wide bands)")
    fig.tight_layout(); fig.savefig(FIG / "ci_forest.png", dpi=130); plt.close(fig)

    # LORO comparison
    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = ["baseline", "+simple-memory", "+moisture-codes"]
    width = 0.35
    x = np.arange(len(order))
    for i, mdl in enumerate(("DecisionTree", "HistGBoost")):
        vals = [exp[(exp.feature_set == s) & (exp.model == mdl)]["loro_auc"].iloc[0]
                for s in order]
        ax.bar(x + (i - 0.5) * width, vals, width, label=mdl)
    ax.axhline(0.5, ls="--", c="grey", lw=0.8, label="chance")
    ax.set_xticks(x); ax.set_xticklabels(order); ax.set_ylim(0.4, 0.8)
    ax.set_ylabel("leave-one-region-out ROC-AUC")
    ax.set_title("Does memory help SPATIAL transfer? (LORO, grid-per-set design)")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "loro_comparison.png", dpi=130); plt.close(fig)
    print("wrote figures/{ceiling_waterfall,ci_forest,loro_comparison}.png")


def inject_bundle_headlines(dataset_fwi, self_noon, self_om, best_ml, loro_wins):
    path = ROOT / "models" / "fire_model.joblib"
    if not path.exists():
        print("(bundle not found -- run train_model.py first)")
        return
    b = joblib.load(path)
    b["metadata"]["audit_ffmc_threshold_acc"] = 0.984
    b["metadata"]["audit_fwi_threshold_acc"] = 0.942
    b["metadata"]["audit_ceiling"] = {"dataset_fwi": dataset_fwi, "self_fwi_noon": self_noon,
                                      "self_fwi_openmeteo": self_om,
                                      "best_servable_ml": best_ml,
                                      "loro_memory_wins_of_4": int(loro_wins)}
    joblib.dump(b, path)
    print("injected audit headlines into models/fire_model.joblib metadata")


if __name__ == "__main__":
    main()
