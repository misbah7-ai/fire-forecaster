"""
Train the servable model with a FIXED configuration -- no re-tuning.

Hyperparameters, random_state, chronological split, and the operating threshold are fixed
constants (the threshold was chosen once on cross-validation targeting fire-recall >= 0.85,
never on the test set). No GridSearchCV, no threshold re-selection, no feature search: with a
hold-out CI of ~+/-0.09, re-selecting the model on within-noise deltas is exactly the mistake
the audit exposes, so this script refuses to do it. A metrics self-check asserts the trained
model's held-out numbers match the documented values for this configuration; on mismatch it
STOPS rather than ship something different.

The shipped model is the honest servable artifact that *demonstrates the ceiling*; the audit
(audit_label.py / audit_ceiling.py) is the headline.
"""
from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_score,
                             recall_score, roc_auc_score)
from sklearn.tree import DecisionTreeClassifier

from data_prep import load_fire_labels
from features import FEATURES, REGIONS, RH_AGGREGATION, SEASON_MONTHS, engineer_features
from fwi import FWI_SEEDING
from stats_utils import bootstrap_metric_ci

ROOT = Path(__file__).parent
OUT_BUNDLE = ROOT / "models" / "fire_model.joblib"
RANDOM_STATE = 42
LABEL_NAMES = {0: "not fire", 1: "fire"}

# FIXED model configuration -- chosen once on cross-validation within the training months,
# then frozen. Never re-tuned (see module docstring).
PARAMS = {"ccp_alpha": 0.005, "max_depth": 5, "min_samples_leaf": 5}
THRESHOLD = 0.3150684931506849  # CV-chosen operating point targeting fire-recall >= 0.85

# documented held-out metrics for this configuration + tolerance for the self-check
EXPECTED = {"roc_auc": 0.710, "pr_auc": 0.756, "recall": 0.865, "precision": 0.744}
LORO_EXPECTED = {"Bejaia": 0.571, "Sidi-Bel Abbes": 0.477}
TOL = 0.02


def build_modelling_frame() -> pd.DataFrame:
    labels = load_fire_labels()[["Region", "date", "label"]]
    weather = pd.read_csv(ROOT / "data" / "openmeteo_fire_history.csv", parse_dates=["date"])
    merged = labels.merge(weather, on=["Region", "date"], how="left", indicator=True)
    miss = merged[merged["_merge"] != "both"]
    if len(miss):
        raise SystemExit(f"JOIN MISS: {len(miss)} labelled days lack weather:\n{miss}")
    merged = merged.drop(columns="_merge")
    parts = []
    for region, g in merged.groupby("Region"):
        X = engineer_features(g[["temp", "RH", "Ws", "Rain"]], region)
        X.index = g.index
        parts.append(X)
    feats = pd.concat(parts).sort_index()
    out = pd.concat([merged[["Region", "date", "label"]], feats], axis=1)
    assert list(out[FEATURES].columns) == FEATURES
    assert out[FEATURES].isna().sum().sum() == 0
    return out


def chrono_split(df):
    tr = df[df["date"].dt.month.isin([6, 7])]
    return tr.copy(), df[~df["date"].dt.month.isin([6, 7])].copy()


def main():
    params, threshold = PARAMS, THRESHOLD
    print(f"fixed configuration: params={params}  threshold={threshold:.6f}  "
          f"rh_aggregation={RH_AGGREGATION!r}")

    df = build_modelling_frame()
    train, test = chrono_split(df)
    Xtr, ytr = train[FEATURES], train["label"].to_numpy()
    Xte, yte = test[FEATURES], test["label"].to_numpy()
    print(f"\nframe: {len(df)} rows base_rate={df.label.mean():.3f} | "
          f"train Jun-Jul n={len(train)} fire={ytr.mean():.3f} | "
          f"test Aug-Sep n={len(test)} fire={yte.mean():.3f}")

    # train eval model on train months (fixed config, no tuning)
    eval_model = DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE,
                                        **params).fit(Xtr, ytr)
    proba = eval_model.predict_proba(Xte)[:, 1]
    pred = (proba >= threshold).astype(int)
    got = {"roc_auc": roc_auc_score(yte, proba), "pr_auc": average_precision_score(yte, proba),
           "recall": recall_score(yte, pred), "precision": precision_score(yte, pred)}
    cm = confusion_matrix(yte, pred)

    # LORO with the fixed params
    loro = {}
    for ho in REGIONS:
        a, b = df[df.Region != ho], df[df.Region == ho]
        m = DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE,
                                   **params).fit(a[FEATURES], a["label"])
        loro[ho] = float(roc_auc_score(b["label"], m.predict_proba(b[FEATURES])[:, 1]))

    # ---- METRICS SELF-CHECK ----
    print("\n===== METRICS SELF-CHECK (trained vs documented for this configuration) =====")
    ok = True
    for k, exp in EXPECTED.items():
        d = abs(got[k] - exp)
        flag = "OK" if d <= TOL else "MISMATCH"
        ok &= d <= TOL
        print(f"  {k:10s} got {got[k]:.3f}  exp {exp:.3f}  |d|={d:.3f}  {flag}")
    for ho, exp in LORO_EXPECTED.items():
        d = abs(loro[ho] - exp)
        flag = "OK" if d <= TOL else "MISMATCH"
        ok &= d <= TOL
        print(f"  LORO {ho:15s} got {loro[ho]:.3f}  exp {exp:.3f}  |d|={d:.3f}  {flag}")
    if not ok:
        raise SystemExit("SELF-CHECK FAILED -- trained metrics diverge from the documented "
                         "configuration. Stopping rather than shipping something different.")
    print("  -> metrics self-check PASSED")

    # bootstrap CIs on the held-out metrics
    roc_ci = bootstrap_metric_ci(yte, proba, "roc_auc")
    pr_ci = bootstrap_metric_ci(yte, proba, "pr_auc")
    print(f"\nheld-out ROC-AUC {roc_ci[0]:.3f}  95% CI [{roc_ci[1]:.3f}, {roc_ci[2]:.3f}]")
    print(f"held-out PR-AUC  {pr_ci[0]:.3f}  95% CI [{pr_ci[1]:.3f}, {pr_ci[2]:.3f}]")

    # final serve model: refit on ALL data with the same fixed params
    final = DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE,
                                   **params).fit(df[FEATURES], df["label"])
    final_proba = final.predict_proba(df[FEATURES])[:, 1]

    make_model_figures(eval_model, final, Xte, yte)

    # per-feature training envelope (for what-if slider clamping)
    feat_ranges = {c: [float(df[c].min()), float(df[c].max())] for c in FEATURES}

    # test predictions for the offline validation view
    tp = test[["Region", "date", "label"]].copy()
    tp["pred_proba"] = proba
    tp["pred_label"] = pred
    for c in FEATURES:
        tp[c] = test[c].to_numpy()
    tp.to_csv(ROOT / "data" / "test_predictions.csv", index=False)

    bundle = {
        "model": final, "features": FEATURES, "threshold": threshold, "classes": [0, 1],
        "label_names": LABEL_NAMES, "regions": REGIONS, "rh_aggregation": RH_AGGREGATION,
        "season_months": SEASON_MONTHS, "feature_ranges": feat_ranges, "fwi_seeding": FWI_SEEDING,
        "metadata": {
            "python": platform.python_version(), "sklearn": __import__("sklearn").__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__,
            "random_state": RANDOM_STATE, "split": "chronological: Jun-Jul train, Aug-Sep test",
            "hyperparameters": params,
            "hyperparameters_source": "fixed (chosen once on CV; never re-tuned)",
            "test_roc_auc": got["roc_auc"], "test_roc_auc_ci": [roc_ci[1], roc_ci[2]],
            "test_pr_auc": got["pr_auc"], "test_pr_auc_ci": [pr_ci[1], pr_ci[2]],
            "test_recall": got["recall"], "test_precision": got["precision"],
            "test_confusion_matrix": cm.tolist(),
            "leave_one_region_out_roc_auc": loro,
            "base_rate": float(df.label.mean()), "train_fire_rate": float(ytr.mean()),
            "test_fire_rate": float(yte.mean()), "n_train": int(len(train)),
            "n_test": int(len(test)), "n_total": int(len(df)),
            "weather_source": "Open-Meteo daily aggregates (ERA5 archive for training)",
            "calibrated": False,
            "calibration_note": "n~63 fires in test makes isotonic fragile; balanced tree ~calibrated -> not applied",
            # audit headline numbers (filled by the audit scripts; kept here for the app)
            "audit_ffmc_threshold_acc": None, "audit_fwi_threshold_acc": None,
            "audit_ceiling": None,
        },
    }
    OUT_BUNDLE.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, OUT_BUNDLE)
    print(f"\nsaved -> {OUT_BUNDLE} ({OUT_BUNDLE.stat().st_size/1024:.1f} KB)")

    # round-trip
    rb = joblib.load(OUT_BUNDLE)
    assert rb["features"] == FEATURES
    assert np.allclose(rb["model"].predict_proba(df[FEATURES])[:, 1], final_proba, atol=1e-10)
    assert df[FEATURES].isna().sum().sum() == 0
    print("round-trip OK: feature order matches, probabilities reproduce (atol=1e-10), no NaNs")
    (ROOT / "models" / "metrics.json").write_text(json.dumps(bundle["metadata"], indent=2,
                                                             default=str))


def make_model_figures(eval_model, final_model, Xte, yte):
    """Tree plot (final serve model, top levels) + impurity vs permutation importance
    (eval model on the Aug-Sep hold-out, n_repeats=30) -> figures/ for the app's Tab 4."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.inspection import permutation_importance
    from sklearn.tree import plot_tree

    figs = ROOT / "figures"
    figs.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(final_model, feature_names=FEATURES, class_names=["not fire", "fire"],
              filled=True, rounded=True, max_depth=3, fontsize=9, ax=ax, proportion=True)
    ax.set_title("Fire-risk decision tree (top levels)")
    fig.tight_layout(); fig.savefig(figs / "tree_structure.png", dpi=130); plt.close(fig)

    imp = pd.Series(eval_model.feature_importances_, index=FEATURES).sort_values()
    perm = permutation_importance(eval_model, Xte, yte, n_repeats=30,
                                  random_state=RANDOM_STATE, scoring="roc_auc")
    perm_s = pd.Series(perm.importances_mean, index=FEATURES).reindex(imp.index)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    imp.plot.barh(ax=ax[0], color="#c0392b"); ax[0].set_title("Impurity importance")
    perm_s.plot.barh(ax=ax[1], color="#2c7fb8")
    ax[1].set_title("Permutation importance (hold-out, ROC-AUC drop)")
    fig.tight_layout(); fig.savefig(figs / "feature_importance.png", dpi=130); plt.close(fig)
    print("saved figures/tree_structure.png, figures/feature_importance.png")


if __name__ == "__main__":
    main()
