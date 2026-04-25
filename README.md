# ehr-pancytopenia-detection

Machine learning pipeline for **medical coding and billing audit** — detecting miscoded or missed pancytopenia diagnoses in hospital EHR data. The model scores every discharged patient visit. Discrepancies between the model score and the ICD codes actually assigned flag visits for coding review, enabling revenue recovery and compliance monitoring.

---

## Business Use Case — Billing Audit

Pancytopenia (ICD-10: D61.818, D61.810, D61.811) is frequently miscoded or missed at discharge. Coders working under volume pressure assign partial codes — anemia (D64.9), thrombocytopenia (D69.6), neutropenia (D70.9) — separately instead of the unified pancytopenia code, or miss it entirely. Each missed secondary diagnosis can shift the DRG weight and reduce reimbursement by thousands of dollars per case.

**The model sits in the post-discharge, pre-submission workflow:**

```
Patient discharged
      |
Coder assigns ICD codes
      |
Model scores visit: P(pancytopenia)
      |
Audit logic compares score vs coding:

  High P + NOT coded  ->  Flag for coding review  (revenue recovery)
  High P + coded      ->  Consistent, no action
  Low P  + coded      ->  Flag for compliance review (overcoding risk)
  Low P  + NOT coded  ->  Consistent, no action
```

The model does not assign codes. It creates a prioritized worklist of visits where clinical evidence and the billing record may not align. A coder reviews each flagged chart and makes the final determination.

**Why precision matters here:** Every flagged visit costs approximately 25 minutes of coder review time. A high-precision model ensures the audit team spends time on genuine issues. The notebook includes a threshold analysis table so the audit team can choose their operating point based on weekly review capacity and expected net revenue recovery.

---

## Project Structure

```
ehr-pancytopenia-detection/
|
|-- notebooks/
|   |-- clinicalbert_xgboost_pipeline.ipynb   -- Full two-stage model (primary)
|   |-- phi_screening_pipeline.ipynb          -- PHI de-identification pipeline
|
|-- data/
|   |-- BasicData_with_ICD.csv                -- Patient visits with ICD codes
|   |-- notes_raw.csv                         -- Synthetic admission H&P notes
|   |-- notes_clean.csv                       -- De-identified notes (PHI pipeline output)
|
|-- scripts/
|   |-- build_features_fast.py                -- Vectorized feature engineering
|
|-- README.md
```

---

## Dataset

Three input files from a hospital EHR system (de-identified):

| File | Rows | Description |
|------|------|-------------|
| BasicData_with_ICD.csv | 10,261 | Patient visits with demographics, discharge codes, primary and secondary ICD codes |
| Medications | 318,194 | Medication administration records |
| Observations | 1,022,704 | Lab draw records |

**Final cohort after exclusions:** 9,802 visits, 178 positives (1.82%), 9,624 negatives.

**ICD data in BasicData_with_ICD.csv:**
- `primary_icd` -- main admission diagnosis (e.g. J18.9 pneumonia, Z51.11 chemo encounter)
- `secondary_icds` -- JSON list of all secondary diagnosis codes per visit
- 5 derived binary audit features

---

## Model Architecture

### Stage 1 -- ClinicalBERT Fine-tuning

Fine-tune `emilyalsentzer/Bio_ClinicalBERT` as a binary classifier on admission H&P notes.

- Freeze bottom 10 of 12 transformer layers -- preserves clinical language knowledge
- Fine-tune top 2 layers + classification head
- 5-fold stratified OOF predictions -- same splits as XGBoost (SEED=42)
- Weighted BCE loss for 1:55 class imbalance
- Early stopping on validation MCC, max 10 epochs
- Output: `bert_note_prob` -- one probability per patient from notes alone

ClinicalBERT is a BERT-family encoder model (~110M parameters), not a generative LLM. Pre-trained on MIMIC-III clinical notes.

### Stage 2 -- XGBoost Two-Pass Pipeline

```
Pass 1  ->  144 tabular + bert_note_prob (145 features)
         ->  Optuna 20 trials, 3-fold CV, train set only
         ->  SHAP -- features with mean |SHAP| >= 0.05 selected

Pass 2  ->  SHAP-selected features
         ->  Optuna re-tuning
         ->  OOF MCC-optimized threshold
         ->  Test set evaluated exactly once
```

---

## Feature Engineering -- 144 Features

| Group | Count | Description |
|-------|-------|-------------|
| Observation -- base stats | 77 | min, max, mean, std, first, last, count for 11 lab types |
| Observation -- extended | 6 | cv and min_minus_first for WBC, RBC, Platelet |
| Medication -- drug flags | 50 | Binary flags for top 50 drugs by frequency (train-derived) |
| Medication -- summary | 3 | total_admin_events, total_distinct_drugs, meds_missing |
| Demographics | 3 | age, age_squared, gender |
| ICD audit features | 5 | has_anemia_code, has_thrombocytopenia_code, has_neutropenia_code, has_chemo_code, has_related_blood_disorder |

**ICD audit features -- why they matter:**
These flags detect partial coding patterns -- the most common miscoding error. A coder who assigned D64.9 + D69.6 + D70.9 separately instead of D61.818 will show all three component flags as positive. This pattern is highly predictive of a missed unified pancytopenia code.

**primary_icd excluded as a feature.** The primary ICD is assigned by the same coder being audited. Using it as a model feature would create circular logic -- the model would learn coder behavior rather than clinical reality.

**Drug category codes excluded.** 87% of medication rows have no category code. SHAP confirmed all 26 category flags contributed zero signal.

---

## Leakage-Free Methodology

Six data leakage sources identified and fixed:

| Source | Fix |
|--------|-----|
| Drug list from full dataset | Computed from training visits only |
| Imputer fitted before split | Fitted inside each CV fold |
| Scaler fitted before split | Fitted inside each CV fold |
| Early stopping on test set | CV validation fold used |
| Threshold from test set | OOF MCC maximization on train set |
| scale_pos_weight from full data | Recomputed per CV fold |

BERT OOF uses the same 5-fold splits as XGBoost (SEED=42). Same patient assignment across both stages.

---

## Results

**XGBoost Pass 2 -- SHAP-selected features, OOF MCC threshold:**

| Metric | Value | Notes |
|--------|-------|-------|
| MCC | 0.661 | Primary metric |
| PR-AUC | 0.723 | Honest benchmark at 1.82% positive rate |
| ROC-AUC | 0.987 | Benchmark only -- inflated by class imbalance |
| Precision | 0.697 | Primary business metric for audit use case |
| Recall | 0.639 | 23 of 36 test positives caught |
| Threshold | 0.921 | OOF MCC-optimized |

**Precision is the primary business metric.** At 0.697 precision, approximately 7 of 10 flagged visits have a genuine coding discrepancy -- a strong hit rate for an audit worklist tool.

**Threshold flexibility:** The notebook includes a threshold analysis table showing precision, recall, flagged volume, and estimated weekly net revenue at every threshold. Audit teams select their operating point based on coder capacity.

**Recall floor alternative:** Commented in code -- recall >= 0.80 approach catches more cases at cost of more false alarms. Appropriate when completeness is prioritized over efficiency.

---

## Key Findings

1. **Lab values dominate.** WBC and Platelet nadir drive most predictions. Drug category flags: SHAP = 0. Only 5 of 50 drug flags survived SHAP selection.

2. **ICD audit features add meaningful signal.** has_chemo_code (30.9% in positives vs 0% in negatives) and has_neutropenia_code (9.4% vs 0%) are strong predictors. Partial coding patterns are detectable by the model.

3. **28% of positive visits had partial coding** -- component codes assigned separately instead of unified pancytopenia code. These are the primary revenue recovery targets.

4. **Relative instability matters.** plateletNum_cv ranked 4th in SHAP -- captures instability that standard deviation alone misses.

5. **Admission values as important as nadir.** wbcNum_first ranked 5th -- model distinguishes patients who arrived sick from those who deteriorated during admission.

6. **4-item bedside scoring rule achieves 97% sensitivity** -- WBC nadir < 4.3 (+2), Platelet nadir < 109 (+2), RBC nadir < 3.5 (+1), WBC mean < 6.7 (+1). Score >= 3 flags for review.

---

## PHI and HIPAA Compliance

De-identification is required even for in-house ML models. HIPAA applies to secondary use of patient data regardless of internal deployment. No in-house exception exists.

phi_screening_pipeline.ipynb implements three-layer screening:
1. Microsoft Presidio -- standard PHI (names, dates, locations, phone, SSN)
2. Custom clinical regex -- MRN, pager numbers, room/bed numbers
3. Residual audit flag -- flags notes requiring manual review

**Synthetic notes:** notes_raw.csv contains synthetically generated notes simulating de-identified Epic Admission H&P notes. For production, replace with real de-identified notes from Epic.

---

## Business Impact and ROI Framework

The model's value is quantifiable. Every correctly identified missed pancytopenia code recovers reimbursement through DRG weight adjustment.

**Revenue recovery estimate per corrected case:**
- Adding a pancytopenia secondary diagnosis shifts the DRG weight for many admissions
- Average revenue recovery varies by payer mix and DRG base rate
- Typical range: $1,500 to $5,000 per corrected case (Medicare DRG rates)
- Payer priority: Medicare and commercial insurance cases have the highest recovery potential. Medicaid rates are lower. Self-pay visits have no billing recovery.

**Net ROI formula:**
```
Net revenue per week = (flagged visits x precision x avg recovery)
                     - (flagged visits x review cost per chart)

Review cost per chart ~ $50 (25 min at $120/hr coder rate)
Break-even precision  ~ 0.017 (nearly any precision positive)
At precision 0.697:   ~$2,050 net recovery per flagged visit
```

**Audit prioritization — rank flagged visits by:**
1. DRG weight delta (highest revenue impact first)
2. Payer category (Medicare and commercial before Medicaid)
3. Model probability (highest confidence flags first)
4. Claim submission status (pre-submission corrections are simpler than amendments)

**Physician and coder analytics (aggregate, not individual):**
Aggregating model flags by attending physician and coding team reveals systematic documentation or coding gaps. Which service lines consistently have high-probability visits without pancytopenia codes? This guides CDI (Clinical Documentation Improvement) education and targeted audits — addressing root causes rather than just flagging individual cases.

---

## Next Steps

**Additional data to improve the model:**

- **DRG code and weight per visit** -- enables ROI-ranked audit worklist. Currently missing from the dataset. Join from billing system on visit_id.
- **Payer mix** -- Medicare, Medicaid, commercial, self-pay. Filters audit worklist to financially actionable cases. Missing from current dataset.
- **Length of stay** -- longer stays correlate with more complex cases and higher miscoding risk. Derivable from admission and discharge dates if available in BasicData.
- **Full claim submission status** -- was the claim already submitted and paid? Pre-submission corrections are operationally simpler. Post-submission require amended claims.
- **Real de-identified notes from Epic** -- replace synthetic notes. ClinicalBERT fine-tuned on real admission H&P notes will generalize better to production visits.
- **Attending physician specialty** -- oncology, hematology, hospitalist. Specialty is a strong prior for pancytopenia risk and coding complexity.

**Model improvements:**

- Temporal train/test split -- train on earlier dates, test on later dates. Prevents the model from learning date-specific patterns that do not generalize forward in time.
- More positive cases -- EPV of ~1.0 is the binding constraint. Extending the cohort to cover more discharge periods would significantly improve model stability.
- External validation -- validate on a different hospital system or held-out time period before production deployment.
- Bias audit -- check model performance separately by age group, gender, and payer. A model that performs well overall but poorly for a specific demographic is not appropriate for production use.
- RxNorm/ATC drug mapping -- recover the 87% of medication rows with no category code. Bone marrow suppressive drug flags (chemotherapy, immunosuppressants) are currently underrepresented because category codes are sparse.

**Audit workflow integration:**

- API endpoint -- wrap the model in a REST API that accepts a visit_id and returns a probability score. Integrate with the hospital's coding workflow system.
- Worklist dashboard -- build a simple interface showing flagged visits ranked by expected revenue recovery, with the top SHAP features explaining each flag.
- Feedback loop -- capture coder decisions (confirmed vs rejected flags) and use them to retrain the model quarterly. The model improves as coders provide ground truth labels on ambiguous cases.

---

## Limitations

- EPV approximately 1.0 -- 142 positive training cases / 144 features. More positive cases would improve model stability.
- Model trained on ICD codes as ground truth -- cannot detect systematic miscoding patterns. Catches inconsistent coding errors.
- Synthetic notes -- ClinicalBERT fine-tuned on synthetic notes may not generalize to real EHR notes.
- Single institution -- thresholds reflect one hospital system. External validation required.
- No patient-level ID -- cannot link repeat visits from the same patient.
- Regulatory -- deployment requires institutional oversight and compliance review.

---

## Installation

```bash
pip install xgboost optuna shap scikit-learn pandas numpy matplotlib
pip install transformers tensorflow
pip install presidio-analyzer presidio-anonymizer
python -m spacy download en_core_web_lg
```

---

## Usage

```bash
# 1. Build feature matrix
python scripts/build_features_fast.py

# 2. Run PHI screening on notes
# Open notebooks/phi_screening_pipeline.ipynb

# 3. Run full model pipeline
# Open notebooks/clinicalbert_xgboost_pipeline.ipynb
```

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Feature engineering | pandas, numpy |
| Hyperparameter tuning | Optuna |
| Gradient boosting | XGBoost |
| Interpretability | SHAP |
| Clinical NLP | HuggingFace Transformers, Bio_ClinicalBERT |
| Deep learning | TensorFlow / Keras |
| PHI screening | Microsoft Presidio |
| Evaluation | scikit-learn |

---

## Disclaimer

This project is for research and educational purposes only. It is not a validated clinical decision support tool and must not be used for autonomous coding or billing decisions without human review, institutional oversight, and compliance approval.
