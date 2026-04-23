# ehr-pancytopenia-detection

End-to-end machine learning pipeline for detecting pancytopenia from hospital EHR data. Built on structured lab observations, medication records, and clinical notes extracted from Epic. Implements a two-stage multimodal architecture — fine-tuned ClinicalBERT on admission notes feeding into a two-pass XGBoost classifier.

---

## What is Pancytopenia

Pancytopenia is a condition defined by simultaneous reduction in red blood cells, white blood cells, and platelets. It is identified at discharge using ICD-10 codes `D61.818`, `D61.810`, and `D61.811`. The condition affects approximately 1.8% of inpatient visits in this cohort. Early detection enables timely hematology referral and reduces risk of serious complications.

---

## Project Structure

```
ehr-pancytopenia-detection/
│
├── notebooks/
│   ├── pancytopenia_junior_analyst.ipynb       # Full ML pipeline — readable, detailed markdown
│   ├── pancytopenia_classification.ipynb       # Senior analyst version — compact code
│   ├── clinicalbert_xgboost_pipeline.ipynb     # Stage 1 ClinicalBERT + Stage 2 XGBoost
│   └── phi_screening_pipeline.ipynb            # PHI de-identification data pipeline
│
├── data/
│   ├── notes_raw.csv                           # Synthetic clinical notes (visit_id, note_text)
│   └── notes_clean.csv                         # De-identified notes (output of PHI pipeline)
│
├── scripts/
│   ├── build_features_fast.py                  # Vectorized feature engineering
│   └── pipeline_lean.py                        # XGBoost two-pass pipeline (memory-efficient)
│
└── README.md
```

---

## Dataset

Three input files from a hospital EHR system (de-identified):

| File | Rows | Description |
|------|------|-------------|
| `BasicData` | 10,261 | One row per patient visit — demographics, discharge codes |
| `Medications` | 318,194 | One row per medication administration event |
| `Observations` | 1,022,704 | One row per lab draw |

**Final cohort after exclusions:** 9,802 visits — 178 positives (1.82%), 9,624 negatives.

**Exclusion rules:**
- 319 visits with no observation records (3 positives — below-average positive rate)
- 140 visits missing all three key labs — WBC, RBC, Platelet (zero positives)

---

## Architecture

### Stage 1 — ClinicalBERT Fine-tuning

Fine-tune `emilyalsentzer/Bio_ClinicalBERT` as a binary classifier on admission H&P notes.

- Freeze bottom 10 of 12 transformer layers — preserves pre-trained clinical language knowledge
- Fine-tune top 2 layers + classification head
- 5-fold stratified OOF predictions — each training patient predicted by a model that never saw them
- Weighted BCE loss for 1:55 class imbalance
- Early stopping on validation MCC (patience=2), max 10 epochs
- Output: `bert_note_prob` — one probability per patient from notes alone

ClinicalBERT is a BERT-family encoder model (~110M parameters), not a generative LLM. It was pre-trained on MIMIC-III clinical notes and understands clinical abbreviations, medication names, and physician language natively.

### Stage 2 — XGBoost Two-Pass Pipeline

Add `bert_note_prob` as feature 140 alongside 139 tabular features. Train XGBoost with full leakage-free methodology.

```
Pass 1  →  139 tabular + bert_note_prob (140 features)
         →  Optuna 20 trials, 3-fold CV
         →  SHAP analysis (mean |SHAP| >= 0.05 threshold)

Pass 2  →  SHAP-selected features
         →  Optuna re-tuning
         →  OOF MCC-optimized threshold
         →  Test set evaluated once
```

**Why one BERT feature and not 768:** XGBoost splits on individual dimensions one at a time. BERT embedding dimensions are dense and entangled — no single dimension has independent meaning. Compressing to one probability preserves semantic signal in a form XGBoost can use correctly.

---

## Feature Engineering

**139 features total:**

| Group | Count | Description |
|-------|-------|-------------|
| Observation — base stats | 77 | min, max, mean, std, first, last, count for 11 lab types |
| Observation — extended | 6 | cv and min_minus_first for WBC, RBC, Platelet |
| Medication — drug flags | 50 | Binary flags for top 50 drugs by frequency (train-derived) |
| Medication — summary | 3 | total_admin_events, total_distinct_drugs, meds_missing |
| Demographics | 3 | age, age_squared, gender |

**Key decisions:**
- Drug list derived from training visits only — prevents leakage
- Drug category codes dropped entirely — 87% missing, SHAP confirmed zero signal
- `plateletNum_cv` (coefficient of variation) validated as rank 4 by SHAP — captures relative instability that `std` alone misses
- `age_squared` added to capture non-linear age effect peaking at 56–65

---

## Leakage-Free Methodology

Six data leakage sources identified and fixed:

| Source | Fix |
|--------|-----|
| Drug list computed on full dataset | Computed from training visits only |
| Imputer fitted before split | Fitted inside each CV fold on fold train |
| Scaler fitted before split | Fitted inside each CV fold on fold train |
| Early stopping used test set | CV validation fold used for early stopping |
| Threshold selected from test set | OOF MCC maximization on training set |
| scale_pos_weight from full train | Recomputed from each fold's training labels |

BERT OOF uses the same 5-fold splits as XGBoost (SEED=42). Same fold = same patient assignment across both stages.

---

## Results

**XGBoost Pass 2 — SHAP-selected features, OOF MCC threshold:**

| Metric | Value | Notes |
|--------|-------|-------|
| MCC | 0.661 | Primary metric — uses all 4 confusion matrix cells |
| PR-AUC | 0.723 | Honest benchmark at 1.82% positive rate |
| ROC-AUC | 0.987 | Reported for benchmarking — inflated by class imbalance |
| Recall | 0.639 | 23 of 36 test positives caught (MCC threshold) |
| Precision | 0.697 | |
| Threshold | 0.921 | OOF MCC-optimized |
| TP=23 | FP=10 | FN=13, TN=1943 |

**Note on threshold choice:** MCC threshold optimizes balance across all four confusion matrix cells. Switching to a recall floor threshold (recall >= 0.80) catches 34 of 36 positives at the cost of 51 false positives. The recall floor approach is preserved as commented-out code for clinical use cases where missing a case is the primary concern.

**Why ROC-AUC is not the headline metric:** At 1.82% positive rate, the model ranks 9,624 negatives that are easy to classify. ROC-AUC of 0.987 is expected and does not reflect genuine discriminative ability on the positive class. PR-AUC and MCC are the honest metrics here.

---

## Key Findings

1. **Lab values dominate — medications add almost nothing.** WBC nadir and Platelet nadir account for the majority of predictive power. All 26 drug category flags: SHAP = 0. Only 5 of 50 drug name flags survived SHAP selection. The model independently found the defining diagnostic labs without clinical guidance.

2. **Relative instability is more predictive than the nadir alone.** `plateletNum_cv` ranked 4th in SHAP. Two patients can have the same platelet minimum but very different trajectories — CV captures this relative volatility that standard deviation alone misses.

3. **Admission values are as important as the nadir.** `wbcNum_first` ranked 5th in SHAP. The model distinguishes patients who arrived already sick from those who deteriorated during admission.

4. **Risk peaks at age 56–65 then declines.** Positive rates: 0.7% (18–40), 2.5% (41–55), 3.1% (56–65), 2.2% (66–75), 1.9% (76–80). Non-linear — motivated `age_squared` feature.

5. **Gender carries a 2.25x rate differential.** Gender 0: 2.67% positive rate vs Gender 1: 1.19%.

6. **A 4-item bedside risk score achieves 97% sensitivity.**

| Criteria | Points |
|----------|--------|
| WBC nadir < 4.3 | +2 |
| Platelet nadir < 109 | +2 |
| RBC nadir < 3.5 | +1 |
| WBC mean < 6.7 | +1 |

Score >= 3 → Sensitivity = 97.2%, Specificity = 86.0%. No computer required.

---

## PHI and HIPAA Compliance

Clinical notes require de-identification before ML training, even for in-house models. HIPAA applies to secondary use of patient data regardless of whether the model is deployed internally. There is no in-house exception.

The `phi_screening_pipeline.ipynb` notebook implements three-layer screening:
1. **Microsoft Presidio** — standard PHI (names, dates, locations, phone, SSN, email)
2. **Custom clinical regex** — MRN, pager numbers, room/bed numbers, clinical date formats
3. **Residual audit flag** — flags notes requiring manual review

Output: `notes_clean.csv` with `visit_id`, `note_text_clean`, `phi_entities_found`, `residual_phi_flag`.

**Note on synthetic data:** The clinical notes in this repository (`notes_raw.csv`) are synthetically generated to simulate real de-identified Epic Admission H&P notes. They are for pipeline development and learning purposes only. In production, real de-identified notes from Epic would replace them.

---

## Limitations

- **EPV = 1.8** after feature reduction — below the recommended minimum of 5. Model stability would improve significantly with more positive cases.
- **No patient-level ID** — cannot link repeat visits from the same patient. If patients appear multiple times, the model may partially learn patient-level patterns.
- **Retrospective design** — all features use the full visit history. Real-time deployment requires time-windowed features (first 24–48 hours).
- **Single institution** — thresholds and drug patterns reflect one hospital system. External validation required before deployment.
- **Synthetic notes** — ClinicalBERT fine-tuned on synthetic notes may not generalize to real EHR notes the same way a model trained on actual de-identified data would.
- **Regulatory pathway** — clinical deployment requires prospective validation and FDA/CE marking as Software as a Medical Device (SaMD).

---

## Future Work

- Time-windowed features for prospective early warning (first 24/48 hours)
- Real de-identified admission notes from Epic replacing synthetic data
- External validation on held-out time period or different institution
- Bias audit across demographic subgroups
- Medication features improved with RxNorm/ATC mapping to recover the 87% uncoded drug category rows
- Temporal train/test split to prevent date-specific pattern leakage

---

## Installation

```bash
pip install xgboost optuna shap scikit-learn pandas numpy matplotlib
pip install transformers tensorflow          # for ClinicalBERT pipeline
pip install presidio-analyzer presidio-anonymizer  # for PHI pipeline
python -m spacy download en_core_web_lg     # required by Presidio
```

---

## Usage

**1. Build feature matrix:**
```bash
python scripts/build_features_fast.py
```

**2. Run PHI screening on notes:**
Open `notebooks/phi_screening_pipeline.ipynb` and set `INPUT_PATH` to your notes CSV.

**3. Run XGBoost-only pipeline:**
Open `notebooks/pancytopenia_junior_analyst.ipynb` — runs end-to-end from raw data to final metrics.

**4. Run ClinicalBERT + XGBoost pipeline:**
Open `notebooks/clinicalbert_xgboost_pipeline.ipynb` — requires GPU for reasonable runtime.

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Feature engineering | pandas, numpy, scipy |
| Hyperparameter tuning | Optuna |
| Gradient boosting | XGBoost |
| Model interpretability | SHAP |
| Clinical NLP | HuggingFace Transformers, Bio_ClinicalBERT |
| Deep learning | TensorFlow / Keras |
| PHI screening | Microsoft Presidio |
| Evaluation | scikit-learn |

---

## Disclaimer

This project is for research and educational purposes only. It is not a validated clinical decision support tool and must not be used for clinical decision-making without prospective validation, regulatory approval, and institutional oversight.
