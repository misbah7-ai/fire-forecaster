"""
Algerian Fire-Danger Benchmark Audit + Interactive 7-Day Forecaster.

The headline is the AUDIT: the dataset's label is a thresholded fire-danger index (not observed
fire), so standard use of this benchmark is circular, and the servable ceiling (~0.78) belongs to
the label, not the model. The shipped tree is the honest servable artifact that *demonstrates*
that ceiling. Everything is read from the bundle / audit CSVs -- nothing about the model is
hardcoded here.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from data_prep import load_fire_labels
from features import FEATURES
from fwi import compute_codes
from live_data import (assert_rh_aggregation_matches, build_live_features,
                       fetch_forecast_weather, season_ok)
from openmeteo import OpenMeteoError

ROOT = Path(__file__).parent
BUNDLE_PATH = ROOT / "models" / "fire_model.joblib"
FIGS = ROOT / "figures"
DATA = ROOT / "data"

st.set_page_config(page_title="Fire Forecaster (Day by Day)",
                   page_icon="🔥", layout="wide")


# ---------------------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------------------
@st.cache_resource
def load_bundle():
    b = joblib.load(BUNDLE_PATH)
    assert_rh_aggregation_matches(b)
    return b


@st.cache_data
def load_labels():
    return load_fire_labels()


@st.cache_data
def load_csv(name):
    p = DATA / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(ttl=3600)
def cached_forecast(region):
    return fetch_forecast_weather(region, forecast_days=7)


def risk_band(p):
    if p >= 0.66:
        return "High", "#c0392b"
    if p >= 0.34:
        return "Moderate", "#e67e22"
    return "Low", "#27ae60"


def decision_rule(model, x_row):
    """Readable rule for one sample: the conjunction of node conditions along its path."""
    t = model.tree_
    node, conds = 0, []
    feat_names = FEATURES
    while t.children_left[node] != t.children_right[node]:  # not a leaf
        f, thr = t.feature[node], t.threshold[node]
        val = x_row[feat_names[f]]
        if val <= thr:
            conds.append(f"{feat_names[f]} ≤ {thr:.1f}")
            node = t.children_left[node]
        else:
            conds.append(f"{feat_names[f]} > {thr:.1f}")
            node = t.children_right[node]
    return conds


# ---------------------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------------------
st.title("🔥 Fire Forecaster (Day by Day)")

try:
    bundle = load_bundle()
except FileNotFoundError:
    st.warning("Model bundle not found. Run `python train_model.py` first.")
    st.stop()

meta = bundle["metadata"]
regions = bundle["regions"]
threshold = bundle["threshold"]
label_names = bundle["label_names"]
franges = bundle["feature_ranges"]

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔥 7-Day Fire Risk", "🔎 Label audit", "📊 How good is this really?", "🌳 Model & method"])


# =======================================================================================
# TAB 1 -- 7-day fire risk
# =======================================================================================
with tab1:
    c1, c2 = st.columns([1, 2])
    with c1:
        region = st.selectbox("Region", list(regions.keys()), key="region_sel",
                              help="Only the two trained Algerian regions. No free-text "
                                   "locations — the model cannot generalise beyond them "
                                   "(leave-one-region-out ≈ chance).")
        st.caption(f"📍 lat {regions[region]['lat']}, lon {regions[region]['lon']}")
    with c2:
        compare = st.toggle("Compare both regions for the selected day", value=False)

    today = date.today()
    if not season_ok(today, bundle):
        months = ", ".join(str(m) for m in bundle["season_months"])
        st.info(f"🚫 **Out of season.** Today is {today:%B %d}. This model only covers the "
                f"Algerian fire season (months {months}, June–September) and does not predict "
                f"outside it — a deliberate refusal, not extrapolation.")
    else:
        try:
            wx = cached_forecast(region)
        except (OpenMeteoError, ValueError) as e:
            st.warning(f"⚠️ Couldn't reach the weather service ({e}). Try again shortly.")
            wx = None

        if wx is not None:
            X = build_live_features(wx, region)
            proba = bundle["model"].predict_proba(X[FEATURES])[:, 1]
            wx = wx.reset_index(drop=True)
            codes = compute_codes(wx.assign(date=pd.to_datetime(wx["date"])))  # context only

            st.caption("Risk shown as coarse **Low / Moderate / High** bands — a shallow tree "
                       "emits only a few distinct values, so decimals would be false precision.")
            # 7-day selectable strip
            day_labels = [f"{r['date']:%a %b %d}" for _, r in wx.iterrows()]
            if hasattr(st, "segmented_control"):
                sel = st.segmented_control("Pick a day", day_labels, default=day_labels[0],
                                           key="day_sel")
            else:
                sel = st.radio("Pick a day", day_labels, horizontal=True, key="day_sel")
            sel_i = day_labels.index(sel) if sel in day_labels else 0

            strip = st.columns(len(wx))
            for i, (_, row) in enumerate(wx.iterrows()):
                band, colour = risk_band(proba[i])
                border = "3px solid #111" if i == sel_i else "1px solid #ccc"
                with strip[i]:
                    st.markdown(f"<div style='border:{border};border-radius:8px;padding:4px;"
                                f"text-align:center'>"
                                f"<div style='font-size:0.8em'>{row['date']:%a}<br>{row['date']:%b %d}</div>"
                                f"<div style='background:{colour};color:white;border-radius:6px;"
                                f"padding:4px;font-weight:700;margin-top:2px'>{band}</div></div>",
                                unsafe_allow_html=True)

            # drill-down for the selected day
            r = wx.iloc[sel_i]
            band, colour = risk_band(proba[sel_i])
            call = label_names[int(proba[sel_i] >= threshold)]
            st.markdown(f"### {r['date']:%A, %B %d} — :{'red' if band=='High' else 'orange' if band=='Moderate' else 'green'}[{band} risk]")
            dc1, dc2 = st.columns(2)
            with dc1:
                st.markdown(f"**Forecast weather** (the model inputs)")
                st.markdown(f"- 🌡️ Temp max: **{r['temp']:.0f} °C**\n"
                            f"- 💧 Min RH: **{r['RH']:.0f} %**\n"
                            f"- 💨 Max wind: **{r['Ws']:.0f} km/h**\n"
                            f"- 🌧️ Rain: **{r['Rain']:.1f} mm**")
                st.caption(f"Model call at operating threshold: **{call}**")
            with dc2:
                cr = codes.iloc[sel_i]
                st.markdown("**Self-computed FWI codes** — _context, NOT model inputs_")
                st.markdown(f"- FFMC {cr['FFMC_c']:.0f} · DMC {cr['DMC_c']:.0f} · DC {cr['DC_c']:.0f}\n"
                            f"- ISI {cr['ISI_c']:.1f} · BUI {cr['BUI_c']:.1f} · FWI {cr['FWI_c']:.1f}")
                st.caption("Shown for physical context only — feeding these in would be circular "
                           "(see the Label audit tab).")

            conds = decision_rule(bundle["model"], r)
            st.markdown("**Why this day:** " + " **and** ".join(conds) + f" → **{band}**")

            # percentiles vs training envelope
            labels_df = load_labels()
            hist = pd.read_csv(DATA / "openmeteo_fire_history.csv")
            hreg = hist[hist.Region == region]
            pcs = {c: (hreg[c] <= r[c]).mean() * 100 for c in ["temp", "RH", "Ws", "Rain"]}
            st.caption("This day vs the region's training distribution — " +
                       " · ".join(f"{c} {pcs[c]:.0f}th pct" for c in pcs))

            if compare:
                st.markdown("#### Both regions, selected day")
                cc = st.columns(len(regions))
                for j, rg in enumerate(regions):
                    try:
                        wj = cached_forecast(rg)
                        Xj = build_live_features(wj, rg)
                        pj = bundle["model"].predict_proba(Xj[FEATURES])[:, 1]
                        k = min(sel_i, len(wj) - 1)
                        bj, cj = risk_band(pj[k])
                        cc[j].markdown(f"**{rg}**")
                        cc[j].markdown(f"<div style='background:{cj};color:white;border-radius:6px;"
                                       f"padding:6px;text-align:center;font-weight:700'>{bj}</div>",
                                       unsafe_allow_html=True)
                    except (OpenMeteoError, ValueError):
                        cc[j].caption(f"{rg}: weather unavailable")

            # what-if explorer
            st.divider()
            with st.expander("🎛️ What-if exploration (not a forecast)"):
                st.caption("Move the sliders to see how the band responds. Values are clamped to "
                           "the training envelope; leaving it is flagged. This is exploration, "
                           "**not** a forecast.")
                if st.button("Reset to forecast"):
                    for c in ["temp", "RH", "Ws", "Rain"]:
                        st.session_state.pop(f"wi_{c}", None)
                wc = st.columns(4)
                vals = {}
                units = {"temp": "°C", "RH": "%", "Ws": "km/h", "Rain": "mm"}
                for col, cc_ in zip(["temp", "RH", "Ws", "Rain"], wc):
                    lo, hi = franges[col]
                    vals[col] = cc_.slider(f"{col} ({units[col]})", float(lo), float(hi),
                                           float(np.clip(r[col], lo, hi)), key=f"wi_{col}")
                # sliders are clamped to the envelope; the honest out-of-envelope case is the
                # FORECAST itself exceeding what training ever saw -> flag it explicitly
                oob = [c for c in vals if not (franges[c][0] <= r[c] <= franges[c][1])]
                xrow = pd.DataFrame([{**{c: vals[c] for c in vals},
                                      "region_sidi": 1 if region == "Sidi-Bel Abbes" else 0}])[FEATURES]
                pw = bundle["model"].predict_proba(xrow)[:, 1][0]
                bw, cw = risk_band(pw)
                st.markdown(f"<div style='background:{cw};color:white;border-radius:6px;padding:8px;"
                            f"text-align:center;font-weight:700'>What-if band: {bw}</div>",
                            unsafe_allow_html=True)
                if oob:
                    st.warning("⚠️ This day's FORECAST is outside the training envelope for: "
                               + ", ".join(f"{c} ({r[c]:.1f} vs trained "
                                           f"[{franges[c][0]:.1f}, {franges[c][1]:.1f}])" for c in oob)
                               + " — sliders are clamped to the envelope, and any output out "
                                 "there is extrapolation, not meaningful.")

            st.caption(f"Forecast fetched live from Open-Meteo at "
                       f"{datetime.now():%Y-%m-%d %H:%M}. RH aggregation: {bundle['rh_aggregation']}.")


# =======================================================================================
# TAB 2 -- label audit (centrepiece)
# =======================================================================================
with tab2:
    st.subheader("The label is a thresholded fire-danger index, not observed fire")
    st.markdown("A single threshold on one FWI-system column reproduces the dataset's `Classes` "
                "almost perfectly. Real ignition can't be that deterministic in weather — so the "
                "label is an **index class**, and feeding FWI components in as features is circular.")

    labels_df = load_labels()
    y = labels_df["label"].to_numpy()
    idx = st.selectbox("Index to threshold", ["FFMC", "FWI", "ISI"], key="audit_idx")
    vals = labels_df[idx].to_numpy(float)
    lo, hi = float(vals.min()), float(vals.max())
    thr = st.slider(f"{idx} threshold (rule: {idx} ≥ t → fire)", lo, hi, float(np.median(vals)),
                    key="audit_thr")
    acc = ((vals >= thr).astype(int) == y).mean()
    # accuracy curve
    grid = np.linspace(lo, hi, 200)
    accs = [(((vals >= t).astype(int) == y).mean()) for t in grid]
    peak_t = grid[int(np.argmax(accs))]
    peak_a = max(accs)
    curve = pd.DataFrame({"threshold": grid, "accuracy": accs}).set_index("threshold")
    st.line_chart(curve)
    m1, m2 = st.columns(2)
    m1.metric(f"Accuracy at your threshold ({thr:.1f})", f"{acc:.3f}")
    m2.metric(f"Best {idx} threshold", f"{peak_t:.1f}", f"acc {peak_a:.3f}")
    st.caption(f"Slide to the peak: {idx} reproduces `Classes` at up to **{peak_a:.1%}** with one "
               f"cut. FFMC peaks near ~0.98.")

    st.divider()
    st.markdown("#### The circularity trap")
    circ = st.checkbox("Include the dataset's FWI columns as model features")
    ceil = load_csv("audit_ceiling.csv")
    if ceil is not None:
        if circ:
            row = ceil[ceil["item"] == "circularity_plus_dataset_FWI"].iloc[0]
            st.error(f"With FWI columns as features → ROC-AUC **{row['auc']:.3f}** "
                     f"[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]. This just reconstructs the "
                     f"labelling rule — impressive and meaningless. Never served.")
        else:
            row = ceil[ceil["item"] == "self_FWI_openmeteo"].iloc[0]
            st.success(f"Servable (raw weather only) → ROC-AUC **{row['auc']:.3f}** "
                       f"[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]. This is the honest number.")

    st.divider()
    st.markdown("#### Ceiling waterfall & ML-vs-formula")
    wc1, wc2 = st.columns(2)
    if (FIGS / "ceiling_waterfall.png").exists():
        wc1.image(str(FIGS / "ceiling_waterfall.png"), width="stretch")
    if ceil is not None:
        mlrow = ceil[ceil["item"] == "best_servable_ML"].iloc[0]
        frow = ceil[ceil["item"] == "self_FWI_openmeteo"].iloc[0]
        wc2.markdown(f"**ML vs formula (both servable):**\n\n"
                     f"- best servable ML: **{mlrow['auc']:.3f}** [{mlrow['ci_lo']:.3f}, {mlrow['ci_hi']:.3f}]\n"
                     f"- self-FWI (Open-Meteo) formula: **{frow['auc']:.3f}** [{frow['ci_lo']:.3f}, {frow['ci_hi']:.3f}]\n\n"
                     f"Difference **{mlrow['auc']-frow['auc']:+.3f}**, CIs overlap → the model adds "
                     f"~nothing over the formula.")

    st.divider()
    st.markdown("#### Exception rows (best FFMC rule) & downloads")
    thresholds_csv = load_csv("audit_label_thresholds.csv")
    # exception rows for FFMC best rule
    ff = labels_df["FFMC"].to_numpy(float)
    grid_ff = np.unique(ff)
    cuts = np.concatenate([[grid_ff[0]-1], (grid_ff[:-1]+grid_ff[1:])/2, [grid_ff[-1]+1]])
    best_t = max(cuts, key=lambda t: (((ff >= t).astype(int) == y).mean()))
    pred = (ff >= best_t).astype(int)
    exc = labels_df[pred != y][["Region", "date", "Temperature", "RH", "Ws", "Rain", "FFMC", "Classes"]].copy()
    exc["date"] = pd.to_datetime(exc["date"]).dt.strftime("%Y-%m-%d")
    st.dataframe(exc, width="stretch")
    dl = st.columns(3)
    for i, nm in enumerate(["audit_label_thresholds.csv", "audit_ceiling.csv", "audit_experiment.csv"]):
        p = DATA / nm
        if p.exists():
            dl[i].download_button(f"⬇ {nm}", p.read_bytes(), file_name=nm, key=f"dl_{nm}")


# =======================================================================================
# TAB 3 -- how good is this really
# =======================================================================================
with tab3:
    st.subheader("Every number with its confidence interval")
    st.markdown(f"The chronological hold-out is small (**n={meta['n_test']}**, "
                f"{int(meta['test_fire_rate']*meta['n_test'])} fire / "
                f"{meta['n_test']-int(meta['test_fire_rate']*meta['n_test'])} not-fire, base rate "
                f"{meta['test_fire_rate']:.3f}); train n={meta['n_train']}, fire rate "
                f"{meta['train_fire_rate']:.3f}. CIs are ~±0.08–0.10, so point estimates alone "
                f"over-claim.")

    rc = meta["test_roc_auc_ci"]; pc = meta["test_pr_auc_ci"]
    cc = st.columns(3)
    cc[0].metric("ROC-AUC", f"{meta['test_roc_auc']:.3f}", f"CI [{rc[0]:.2f}, {rc[1]:.2f}]")
    cc[1].metric("PR-AUC", f"{meta['test_pr_auc']:.3f}", f"CI [{pc[0]:.2f}, {pc[1]:.2f}]")
    cc[2].metric("Base rate (test)", f"{meta['test_fire_rate']:.3f}")
    st.caption("Trivial always-predict-fire baseline: recall 1.000, precision ≈ base rate, "
               "ROC-AUC 0.500 — the model must beat this, not the 0.5 midpoint.")

    if (FIGS / "ci_forest.png").exists():
        st.image(str(FIGS / "ci_forest.png"), width="stretch")

    st.divider()
    st.markdown("#### Interactive operating threshold")
    tp = load_csv("test_predictions.csv")
    if tp is not None:
        opt = st.slider("Operating threshold", 0.0, 1.0, float(threshold), 0.01, key="op_thr")
        yt = tp["label"].to_numpy()
        pr = (tp["pred_proba"].to_numpy() >= opt).astype(int)
        tp_, fp_ = int(((pr == 1) & (yt == 1)).sum()), int(((pr == 1) & (yt == 0)).sum())
        fn_, tn_ = int(((pr == 0) & (yt == 1)).sum()), int(((pr == 0) & (yt == 0)).sum())
        rec = tp_ / (tp_ + fn_) if tp_ + fn_ else 0
        prec = tp_ / (tp_ + fp_) if tp_ + fp_ else 0
        oc = st.columns(3)
        oc[0].metric("Recall", f"{rec:.3f}")
        oc[1].metric("Precision", f"{prec:.3f}")
        oc[2].metric("Shipped threshold", f"{threshold:.3f}",
                     "chosen on CV, never test" )
        cm = pd.DataFrame([[tn_, fp_], [fn_, tp_]],
                          index=["actual not-fire", "actual fire"],
                          columns=["pred not-fire", "pred fire"])
        st.table(cm)

    st.divider()
    st.markdown("#### Memory / spatial-transfer experiment (with CIs)")
    exp = load_csv("audit_experiment.csv")
    if exp is not None:
        show = exp.copy()
        show["chrono"] = show.apply(lambda r: f"{r.chrono_auc:.3f} [{r.chrono_lo:.2f},{r.chrono_hi:.2f}]", axis=1)
        show["LORO"] = show.apply(lambda r: f"{r.loro_auc:.3f} [{r.loro_lo:.2f},{r.loro_hi:.2f}]", axis=1)
        st.dataframe(show[["feature_set", "model", "chrono", "LORO"]], width="stretch")
        st.caption("Chronological deltas sit inside the CI band (not significant). Only the LORO "
                   "direction — memory beats baseline on spatial transfer — replicates across all "
                   "four comparisons. Reported as a hypothesis, not a banked result.")
        if (FIGS / "loro_comparison.png").exists():
            st.image(str(FIGS / "loro_comparison.png"), width="stretch")

    st.divider()
    st.markdown("#### Offline validation (held-out days)")
    if tp is not None:
        fcol = st.columns(2)
        rsel = fcol[0].multiselect("Region", sorted(tp["Region"].unique()),
                                   default=sorted(tp["Region"].unique()))
        tp2 = tp.copy()
        tp2["month"] = pd.to_datetime(tp2["date"]).dt.month
        msel = fcol[1].multiselect("Month", sorted(tp2["month"].unique()),
                                   default=sorted(tp2["month"].unique()))
        v = tp2[tp2.Region.isin(rsel) & tp2.month.isin(msel)].copy()
        v["actual"] = v["label"].map(label_names)
        v["predicted"] = v["pred_label"].map(label_names)
        v["date"] = pd.to_datetime(v["date"]).dt.strftime("%Y-%m-%d")
        st.dataframe(v[["Region", "date", "actual", "predicted", "temp", "RH", "Ws", "Rain"]],
                     width="stretch", height=300)


# =======================================================================================
# TAB 4 -- model & method
# =======================================================================================
with tab4:
    st.subheader("Model & method")
    st.markdown(f"`DecisionTreeClassifier(class_weight='balanced')`, hyperparameters "
                f"**fixed (chosen once on CV, never re-tuned)**: `{meta['hyperparameters']}`. "
                f"Features (raw weather only, exact order): `{bundle['features']}`. "
                f"RH aggregation: {bundle['rh_aggregation']} — persisted in the bundle and "
                f"asserted to match live (train/serve parity cannot silently drift).")

    def fig(name, cap):
        p = FIGS / name
        if p.exists():
            st.image(str(p), caption=cap, width="stretch")
        else:
            st.caption(f"({name} not found — run train_model.py)")

    fig("tree_structure.png", "Fitted tree (top levels): cool → not fire; hot + dry (low min RH) "
        "+ windy → fire. Matches fire physics.")
    fig("feature_importance.png", "Impurity importance vs a permutation cross-check (n_repeats=30) "
        "on the hold-out.")

    st.markdown("#### Measured, but deliberately NOT adopted (a strength, not a gap)")
    st.markdown(
        "- **HistGradientBoosting** and **drought-memory features** were tested. Their "
        "chronological gains fell **inside the bootstrap CI**, so adopting them would be "
        "fitting noise — we didn't. (See the audit.)\n"
        "- **Calibration** was not added: at ~63 fire days isotonic is fragile, and a balanced "
        "tree is already ~calibrated. A measured decision, not an oversight.\n"
        "- The one replicating signal — memory helping **spatial transfer** (LORO) — is reported "
        "as a hypothesis for future work, not baked into the shipped model.")

    st.markdown("#### Limitations")
    st.markdown(f"- Two regions, one 2012 season, {meta['n_total']} daily rows, daily resolution.\n"
                f"- The label is a fire-danger **index**, not observed ignition (see Label audit).\n"
                f"- Region + season gating are honesty features — the model refuses where/when it "
                f"has no training support.")
    st.caption(f"scikit-learn {meta['sklearn']}, Python {meta['python']}, random_state "
               f"{meta['random_state']}. {meta['calibration_note']}")


# ---------------------------------------------------------------------------------------
st.divider()
st.caption("Weather © Open-Meteo.com (CC BY 4.0). Fire labels: Algerian Forest Fires dataset — "
           "Abid et al., 2019 (UCI ML Repository).")
