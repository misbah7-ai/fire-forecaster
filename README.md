# 🔥 Fire Forecaster (Day by Day)

**The audit is the headline.** This project shows that the widely-used Algerian Forest Fires
benchmark's label is **not observed fire — it is a thresholded fire-danger index** (a single
FFMC threshold reproduces it at ~98%), that standard published use of the benchmark is
therefore **circular**, and that the **servable ceiling (~0.78 ROC-AUC)** belongs to the label,
not to any model. A small decision tree with a fixed, never re-tuned configuration is served as
the honest artifact that *demonstrates* that ceiling — wrapped in an interactive Streamlit
explorer with a live Open-Meteo 7-day forecast.

## The audit in five numbers (all recomputed by the audit scripts, all with bootstrap CIs — see [FINDINGS.md](FINDINGS.md))

| finding | number |
|---|---|
| one FFMC threshold reproduces the label | **0.984** accuracy (239/243) |
| dataset's own FWI column → label (circular) | ROC-AUC **0.987** [0.971, 0.998] |
| self-computed FWI from the same noon weather | **0.866** [0.800, 0.926] |
| self-computed FWI from Open-Meteo (servable) | **0.770** [0.687, 0.850] |
| best servable ML vs that formula | **0.776 vs 0.770** — indistinguishable |

With a test set of n≈122 the CI on any single ROC-AUC is ±0.08–0.10, so **no difference inside
that band is treated as real** — model-family swaps and drought-memory features both land inside
it. The one consistent direction (memory features improving **leave-one-region-out** transfer,
3 of 4 comparisons, best ≈ 0.60 → 0.71) is reported strictly as a hypothesis.

## The served model (demonstrates the ceiling)

`DecisionTreeClassifier(class_weight="balanced")` with a **fixed configuration** — hyperparameters
and operating threshold chosen once on cross-validation (fire-recall ≥ 0.85 target), then frozen;
a metrics self-check in `train_model.py` stops the build if the trained model diverges from the
documented numbers: ROC-AUC **0.710 [0.620, 0.798]**, PR-AUC 0.756, recall 0.865, precision
0.744, leave-one-region-out 0.571/0.477. Raw servable weather only (`temp` = daily max
temperature, `RH` = daily-min hourly relative humidity, `Ws` = daily max wind, `Rain` = daily
total, region flag); FWI components are **audit/context only, never features**. Region-gated to
Bejaia + Sidi-Bel Abbes and season-gated to June–September — deliberate refusals to predict
outside the training envelope.

## The app (4 tabs)

1. **7-Day Fire Risk** — live Open-Meteo forecast → Low/Moderate/High bands (no decimal
   theatre), selectable day strip, per-day drill-down with the tree's decision path ("why this
   day"), FWI codes as *context*, what-if sliders clamped to the training envelope, region
   comparison, season gate.
2. **Label audit** *(centrepiece)* — interactive threshold explorer (find the ~98% FFMC peak
   yourself), circularity toggle (servable AUC ↔ ~0.99), ceiling waterfall, exception rows,
   audit CSV downloads.
3. **How good is this really?** — every metric with its CI, trivial always-fire baseline,
   interactive operating-threshold slider, the memory/spatial-transfer experiment with CIs,
   filterable offline validation.
4. **Model & method** — tree plot, importances, the *measured-but-not-adopted* section
   (gradient boosting, memory features, calibration — evaluated, inside the noise band,
   deliberately not shipped).

## Layout

```
features.py      servable feature contract + RH aggregation (ONE place; bundle-asserted parity)
data_prep.py     parse the stacked two-region CSV -> 243 labelled rows
openmeteo.py     ERA5 archive (train) + forecast (live) fetchers -- one RH aggregation
fwi.py           Canadian FWI system (Van Wagner), lean numpy, cffdrs-validated -- AUDIT ONLY
train_model.py   fixed-configuration training + metrics self-check + bundle + figures
audit_label.py   is the label an index threshold?  -> data/audit_label_thresholds.csv
audit_ceiling.py ceiling chain, ML-vs-formula, circularity, CIs, LORO -> audit CSVs + figures
live_data.py     live fetch + build_live_features self-test + season gate
app.py           the 4-tab Streamlit explorer
```

## Run locally

```bash
python -m venv venv && venv\Scripts\activate
pip install -r requirements-dev.txt
python train_model.py      # train + metrics self-check
python audit_label.py      # label audit
python audit_ceiling.py    # ceiling audit (writes CSVs/figures)
python live_data.py        # live-parity self-test
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)

Push to a public GitHub repo (the `.joblib` **must** be committed), share.streamlit.io →
Create app → `app.py` → **Advanced settings → Python 3.12** → Deploy.

## Attribution

Weather © [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0). Fire labels: Algerian Forest
Fires dataset — Abid et al., 2019 (UCI ML Repository).
