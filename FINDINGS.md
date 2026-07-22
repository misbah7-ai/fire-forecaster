# Findings — the benchmark audit is the headline

This project audits the widely-used Algerian Forest Fires benchmark and ships the honest
servable forecaster that its label actually supports. Every number below is recomputed by
`audit_label.py` / `audit_ceiling.py` (nothing hardcoded) and every ROC-AUC carries a
≥2000-resample bootstrap CI.

## 1. The label is a thresholded fire-danger index, not observed fire

A **single threshold on the dataset's own FFMC reproduces `Classes` at 98.4%**
(`FFMC ≥ 80.15 → fire`, 239/243, 4 exceptions — all Bejaia days with FFMC barely above the
cut). One ISI threshold: 97.9%. One FWI threshold: 94.2% (229/243). Class-conditional FWI
ranges barely overlap (fire 1.7–31.1 vs not-fire 0.0–6.1).

Real Mediterranean fire occurrence is stochastic and overwhelmingly human-caused — it cannot
be ~98% deterministic in same-day weather. **`Classes` is an index class, not an event
record.** Full sweep in `data/audit_label_thresholds.csv`.

## 2. Standard use of this benchmark is therefore circular

Published work feeds FFMC/DMC/DC/ISI/BUI/FWI in as *features* to predict this label and
reports near-perfect scores (some F1 = 1.0). Those models reconstruct the labelling rule.
Our circularity demonstration: adding the dataset's FWI columns to the features lifts hold-out
ROC-AUC to ~0.99 — impressive and meaningless. It is never served.

## 3. The servable ceiling belongs to the label, not the model

Single-feature hold-out ROC-AUC (Aug–Sep), with 95% bootstrap CIs:

| step | ROC-AUC | gap |
|---|---|---|
| dataset's own FWI column | 0.987 [0.971, 0.998] | — (circular reference) |
| self-computed FWI from the same **noon** weather | 0.866 [0.800, 0.926] | −0.121 implementation/startup gap |
| self-computed FWI from **Open-Meteo ERA5** | 0.770 [0.687, 0.850] | −0.096 reanalysis-vs-station gap |

Noon-station weather is unobservable live (no free API serves it), so **~0.78 is roughly the
servable ceiling**. (Our Van Wagner implementation matches the canonical cffdrs reference
vector to 2 dp, so the first gap is the dataset's non-standard startup/spin-up, not our code.)

## 4. The ML model adds ~nothing over the formula

Best servable ML (raw weather, either model family) ≈ **0.776** vs self-FWI-from-Open-Meteo
alone ≈ **0.770** — overlapping CIs, indistinguishable. On this label, learning ≈ recomputing
the index.

## 5. Single-split comparisons on this data are underpowered — so none are banked

The chronological test set is n≈122 (≈74 fire / 48 not-fire, base rate ≈0.607; train n≈121,
fire rate ≈0.521). The 95% bootstrap CI on a single ROC-AUC is ≈ ±0.08–0.10. Model-family
swaps (gradient boosting vs the tree) and drought-memory feature sets all land **inside that
band**, so none is adopted. Rule applied repo-wide: no ROC-AUC without its CI; no within-CI
difference called an improvement or a regression. The trivial always-predict-fire baseline
(recall 1.000, precision ≈0.607, AUC 0.500) is reported alongside.

## 6. Memory may help *spatial transfer* (hypothesis, not banked)

In the feature experiment (grid per feature set, LORO predictions pooled across regions before
bootstrapping), drought-memory variants beat baseline on **leave-one-region-out** in **3 of 4**
comparisons (best: baseline ≈ 0.60 → +simple-memory HGB ≈ 0.71 [0.64, 0.77]). A consistent
direction, but not a universal one — reported strictly as a hypothesis, and itself an
illustration of how much these small-sample comparisons move between reasonable measurement
designs. Either way, the practical conclusion stands: same-day weather does not transfer
spatially here, which is *why* the app hard-gates to the two trained regions.

## The served artifact

A decision tree with a **fixed configuration** — hyperparameters and operating threshold chosen
once on cross-validation (fire-recall ≥ 0.85 target), then frozen; re-selecting on within-noise
deltas is the mistake this audit exposes, so `train_model.py` refuses to tune and self-checks
its metrics against the documented values: ROC-AUC 0.710 [0.620, 0.798], PR-AUC 0.756, recall
0.865 (64/74 fire days), precision 0.744, leave-one-region-out 0.571 / 0.477. Not calibrated
(n≈63 fires makes isotonic fragile; the balanced tree is already ~calibrated) — a measured
decision. Raw servable weather only; region- and season-gated.

## What would actually raise the ceiling (future work, not built here)

Not a better model — a better *label*: a dataset with **observed ignitions plus fuel and
human-activity drivers** (e.g. the Mesogeos Mediterranean datacube, whose extent covers both
study regions). Until the label is real fire, every gain on this benchmark is a gain at
reconstructing a weather index.
