# ML Learning Notes
> Personal reference document — concepts, decisions, and reasoning from hands-on project work
> Started: April 2026 | Project context: Pancytopenia Binary Classification

---

## Table of Contents
1. [Gradient Boosting — XGBoost vs LightGBM](#1-gradient-boosting--xgboost-vs-lightgbm)
2. [Class Imbalance](#2-class-imbalance)
3. [Evaluation Metrics for Imbalanced Problems](#3-evaluation-metrics-for-imbalanced-problems)
4. [Data Leakage — Sources and Fixes](#4-data-leakage--sources-and-fixes)
5. [Cross Validation](#5-cross-validation)
6. [Hyperparameter Tuning — RandomizedSearchCV vs Optuna](#6-hyperparameter-tuning--randomizedsearchcv-vs-optuna)
7. [Feature Engineering Principles](#7-feature-engineering-principles)
8. [SHAP — Model Interpretability](#8-shap--model-interpretability)
9. [EPV — Events Per Variable](#9-epv--events-per-variable)
10. [Threshold Selection](#10-threshold-selection)
11. [Model Selection Framework](#11-model-selection-framework)
12. [Feature Importance — Built-in vs SHAP](#12-feature-importance--built-in-vs-shap)
13. [Preprocessing — Imputer and Scaler](#13-preprocessing--imputer-and-scaler)
14. [Two-Pass Modeling Strategy](#14-two-pass-modeling-strategy)
15. [Clinical Translation of ML Models](#15-clinical-translation-of-ml-models)
16. [Multimodal ML — Combining Text and Tabular Data](#16-multimodal-ml--combining-text-and-tabular-data)
17. [Transfer Learning and Fine-Tuning](#17-transfer-learning-and-fine-tuning)
18. [ClinicalBERT — Architecture and Key Concepts](#18-clinicalbert--architecture-and-key-concepts)
19. [The Right Way to Combine Fine-Tuned BERT with XGBoost](#19-the-right-way-to-combine-fine-tuned-bert-with-xgboost)
20. [Prescription Data as Additional Feature Source](#20-prescription-data-as-additional-feature-source)
21. [Clinical Notes — Real-World Source, Acquisition, and Privacy](#21-clinical-notes--real-world-source-acquisition-and-privacy)
22. [PHI Screening and HIPAA Compliance Pipeline](#22-phi-screening-and-hipaa-compliance-pipeline)
22. [PHI Screening and HIPAA Compliance for Clinical Notes](#22-phi-screening-and-hipaa-compliance-for-clinical-notes)

---

## 1. Gradient Boosting — XGBoost vs LightGBM

Both are gradient boosting algorithms. They build trees sequentially — each new tree corrects the errors of the previous ones. The core idea is identical. The differences are in **how** they build those trees.

### How XGBoost builds trees — Level-wise (breadth-first)

XGBoost grows trees level by level. It splits all nodes at the same depth before going deeper.

```
Level 1:        [root]
                /     \
Level 2:      [A]     [B]
              / \     / \
Level 3:    [C][D]  [E][F]
```

Every node at level 2 is split before any node at level 3 is considered. Produces balanced, symmetric trees.

### How LightGBM builds trees — Leaf-wise (best-first)

LightGBM always splits the leaf that gives the biggest reduction in loss — regardless of which level it is on.

```
Start:          [root]
                /     \
              [A]     [B]       ← split root first
              / \
            [C] [D]             ← split A because it had bigger gain than B
            /
          [E]                   ← split C because bigger gain than D or B
```

Tree grows asymmetrically, chasing the most informative splits first.

### Comparison table

| | XGBoost | LightGBM |
|--|---------|---------|
| Tree growth | Level-wise — balanced | Leaf-wise — asymmetric |
| Speed | Slower | Faster — often 10x+ |
| Memory | More | Less |
| Overfitting risk | Lower at same depth | Higher — deep asymmetric trees overfit more easily |
| Key regularization | `max_depth` | `num_leaves` (controls tree complexity directly) |
| NaN handling | Native | Native |
| Class imbalance | `scale_pos_weight` | `scale_pos_weight` or `is_unbalance=True` |
| Small datasets | Good | Can overfit more easily — needs careful `num_leaves` tuning |

### The overfitting risk on small datasets

LightGBM's leaf-wise growth can memorize small training sets. In practice on a dataset with ~142 positive training cases, LightGBM achieved train PR-AUC = **1.000** collapsing to test PR-AUC = **0.787** — perfect training score, significant test drop.

**Fix:** Reduce `num_leaves` significantly. Default is 31. For small datasets try 10–15. This constrains tree complexity and prevents aggressive leaf-wise splitting from memorizing training data.

### Histogram binning

LightGBM bins continuous features into discrete buckets (default 255 bins) before finding splits. XGBoost does the same but LightGBM was the first to implement it aggressively. This is why LightGBM is faster — instead of evaluating every possible split point, it evaluates at most 255 per feature.

### When to use which

- **XGBoost** — safer default for small datasets. More robust out of the box.
- **LightGBM** — better for large datasets (>100k rows). Faster training. Needs careful `num_leaves` tuning on small data.
- **In practice** — run both with proper CV tuning and compare. The winner varies by dataset.

---

## 2. Class Imbalance

When one class is much rarer than the other (e.g. 1.8% positive, 98.2% negative), standard modeling approaches break down.

### Why accuracy is useless

A model that always predicts "negative" achieves 98.2% accuracy on a 1.8% positive rate dataset. This looks excellent but catches zero true cases. Accuracy is a misleading metric for imbalanced problems.

### The ratio that matters

```
scale_pos_weight = number of negatives / number of positives
```

For a 1.8% positive rate: `scale_pos_weight ≈ 54`. This tells XGBoost/LightGBM to treat each positive case as if it were 54 negative cases — directly compensating for the imbalance.

For sklearn models: `class_weight='balanced'` does the same thing automatically.

### Important: compute from training data only

`scale_pos_weight` must be computed from the training set labels, not the full dataset. If you compute from the full dataset before splitting, you are using test set information to set a training parameter — a subtle form of data leakage.

Inside each CV fold, recompute from that fold's training labels:
```python
fold_scale_pos = (y_fold_train == 0).sum() / (y_fold_train == 1).sum()
```

### SMOTE — synthetic oversampling

An alternative to class weighting. Generates synthetic positive cases by interpolating between existing ones. Apply to training fold only — never to validation or test set. Evaluate whether it improves over class weighting — it does not always help.

---

## 3. Evaluation Metrics for Imbalanced Problems

### Why ROC-AUC misleads at severe imbalance

ROC-AUC evaluates ranking across all positives and negatives. At 1.82% positive rate, the model ranks 9,624 easy negatives — getting most of them right inflates the score. A model with ROC-AUC = 0.99 can still miss many positives. Use ROC-AUC for benchmarking only.

### PR-AUC (Precision-Recall AUC)

Focuses only on the positive class. Evaluates the tradeoff between:
- **Precision** — of all flagged cases, what % are real?
- **Recall** — of all real cases, what % did we catch?

PR-AUC is the honest benchmark for imbalanced classification. A random classifier achieves PR-AUC equal to the positive rate (1.82%) — so there is no inflation from easy negatives.

### MCC (Matthews Correlation Coefficient)

```
MCC = (TP×TN − FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

Range: -1 (perfectly wrong) → 0 (random) → +1 (perfect)

The only single metric that uses all four confusion matrix cells simultaneously. Cannot be gamed:
- Predicting all positive → MCC ≈ 0
- Predicting all negative → MCC ≈ 0
- Only genuinely discriminating classifiers get high MCC

**MCC vs PR-AUC:**
- PR-AUC is threshold-independent — evaluates overall ranking ability
- MCC is point-in-time at a chosen threshold — honest summary of classifier quality at that operating point
- Use both together: PR-AUC tells you if the model's scores are meaningful, MCC tells you if the chosen threshold produces a genuinely useful classifier

### Recall (Sensitivity)

What % of actual positive cases did the model catch?

```
Recall = TP / (TP + FN)
```

The primary clinical metric when missing a case is more costly than a false alarm. A model catching 34 of 36 positives (recall = 0.944) is clinically meaningful regardless of other metrics.

### Do NOT use Accuracy

For imbalanced problems, accuracy is misleading. Always report PR-AUC, MCC, and Recall.

---

## 4. Data Leakage — Sources and Fixes

Data leakage is when information from the test set (or future data) influences model training. It makes performance metrics look better than they really are. The model appears to generalize but is actually memorizing test set patterns.

### Six leakage sources — identified and fixed in practice

| # | Leakage Risk | Fix |
|---|-------------|-----|
| 1 | Feature list (e.g. top 50 drugs) computed on full dataset | Compute on training visits only. Apply same list to test. |
| 2 | SimpleImputer fitted on full X before splitting | Fit inside each CV fold on fold train. Transform fold val separately. |
| 3 | StandardScaler fitted on full X before splitting | Same as above — fit on fold train only. |
| 4 | Early stopping used test set as `eval_set` | Use CV validation fold for early stopping. Never test set. |
| 5 | Threshold selected using test set performance | Select threshold from OOF predictions on training set. Apply fixed threshold to test. |
| 6 | `scale_pos_weight` computed from full training set | Recompute from each fold's training labels inside the CV loop. |

### Why leakage is subtle

Most practitioners catch leakages 1 and 3. Leakages 4 (early stopping) and 5 (threshold selection) are commonly missed. Leakage 5 is especially insidious — it feels like you are just "choosing a cutoff" but you are actually fitting the model to the test data.

### The honest cost of fixing leakage

When leakage was fixed in our project, PR-AUC dropped from 0.832 to 0.609 (CV estimate). This does not mean model quality degraded — it means the original number was inflated. The CV estimate of 0.609 is the genuine, reproducible measure of model skill.

---

## 5. Cross Validation

### Why cross validation

A single train/test split gives one estimate of performance. If the split was lucky or unlucky, the estimate is noisy. CV runs multiple splits and averages the results — more stable estimate.

### Stratified K-Fold — always use for imbalanced problems

Regular KFold does not guarantee each fold has the same class proportion. With 1.82% positive rate and only 142 positives in training, a regular fold could end up with very few or zero positives in the validation set — making PR-AUC meaningless.

```python
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

`shuffle=True` randomizes before splitting. `random_state=42` ensures reproducibility.

### Leave-One-Out CV (LOOCV) — when NOT to use it

LOOCV is an extreme case where K equals the number of samples. Each fold validates on exactly one patient.

**Problems for imbalanced clinical data:**
- Computationally prohibitive with XGBoost (thousands of model fits)
- Stratification is impossible — a single sample cannot be stratified
- With 1.82% positive rate, most folds see only one negative patient — PR-AUC on one sample is meaningless
- High variance estimate — worse than 5-fold CV despite appearing more thorough

**LOOCV is appropriate for:** very small balanced datasets (<100 samples) with fast models (linear regression). Not for our use case.

### Out-of-Fold (OOF) predictions

When running K-Fold CV, every training sample gets predicted exactly once — when its fold is the held-out validation fold. Collecting all these predictions gives OOF probabilities across the entire training set.

Key property: every OOF prediction was made by a model that never saw that patient during training. OOF predictions behave like test set predictions but are available on training data — useful for threshold selection without touching the test set.

---

## 6. Hyperparameter Tuning — RandomizedSearchCV vs Optuna

### RandomizedSearchCV

Randomly samples combinations from a predefined grid. Simple, no extra library, no function definition needed.

```python
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator           = model,
    param_distributions = param_grid,
    n_iter              = 30,
    scoring             = 'average_precision',
    cv                  = stratified_kfold,
    random_state        = 42,
    n_jobs              = -1
)
```

**Important:** Pass a `StratifiedKFold` object explicitly to `cv` — the default `cv=5` uses regular KFold which does not stratify.

After `fit()`, `search.best_estimator_` is already retrained on the full training set. No need to retrain manually.

### Optuna

Bayesian optimization — learns from previous trials to focus search on promising parameter regions. More efficient than random search, especially with many parameters.

```python
import optuna

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
    }
    # run CV with these params, return mean CV score
    return mean_cv_score

study = optuna.create_study(direction='maximize',
                             sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=30)
best_params = study.best_params
```

### Comparison

| | RandomizedSearchCV | Optuna |
|--|-------------------|--------|
| Code complexity | Low — no objective function | Medium — requires objective function |
| Search strategy | Random | Bayesian — learns from prior trials |
| Efficiency | Moderate | Better — focuses on promising regions |
| Practical difference (30 trials) | Small | Small — advantage grows with more trials |

**Rule of thumb:** For quick exploration, RandomizedSearchCV is fine. For production tuning with 6+ parameters and 50+ trials, Optuna is worth the extra code.

### Essential XGBoost parameters to tune

| Parameter | What it controls | Range |
|-----------|-----------------|-------|
| `max_depth` | Tree depth — primary complexity control | 3–6 |
| `learning_rate` | Step size — lower = more conservative | 0.01–0.2 |
| `n_estimators` | Number of trees | 100–500 |
| `subsample` | Fraction of rows per tree | 0.6–1.0 |
| `colsample_bytree` | Fraction of features per tree | 0.5–1.0 |
| `min_child_weight` | Minimum samples in a leaf | 3–20 |

L1/L2 regularization (`reg_alpha`, `reg_lambda`) can be skipped when the above six are tuned — they provide diminishing returns.

---

## 7. Feature Engineering Principles

### Coefficient of Variation (CV) — normalized instability

`cv = std / mean`

A dimensionless measure of relative variability. Unlike `std`, it is normalized by the mean — making it comparable across features on different scales.

**Why it matters:** Two patients can have the same `std` but very different clinical pictures.
- Patient A: mean platelet = 200, std = 40 → cv = 0.20 (stable)
- Patient B: mean platelet = 40, std = 40 → cv = 1.00 (wildly unstable relative to low baseline)

`std` cannot distinguish these. `cv` can. Useful whenever relative instability matters more than absolute instability.

**Correlation note:** `cv`, `std`, and `mean` are mathematically related. For tree models this does not matter — correlated features just share importance scores. For Logistic Regression, use L2 regularization to handle multicollinearity.

### min_minus_first — capturing trajectory to nadir

`min_minus_first = min_value - first_value`

How far a value dropped from admission to its lowest point during the stay. Captures dangerous dips even if the patient later recovered.

**Why last_minus_first alone is insufficient:** A patient whose platelet starts at 116, drops to 28, then recovers to 150 shows `last_minus_first = +34` — appears normal. `min_minus_first = -88` correctly captures the dangerous deterioration. V-shaped or U-shaped trajectories are invisible to endpoint-only features.

### Why count features matter

How many times a lab was drawn during a stay is a proxy for clinical concern. A patient with 50 platelet draws was being actively monitored for a worrying trend. 2 draws suggests a routine check. Draw frequency is clinical worry made numerical.

### age_squared — capturing non-linear age effects

When a linear model sees `age`, it can only fit a straight line. Adding `age_squared` allows it to fit a curve. Use when EDA shows a non-linear relationship between age and outcome (e.g., risk peaks at 56–65 then declines).

Tree models find non-linearity automatically through splits — `age_squared` is primarily for linear models. Harmless to include for tree models since they will ignore it if not useful.

### Train/test split before feature derivation

Any feature whose derivation involves looking at the data must be computed from training data only. Examples:
- Top N drugs by frequency → compute frequency on train visits only
- Top N drugs by lift → compute lift on train visits only
- Imputation values (median, mean) → compute on train only

Computing on the full dataset before splitting leaks test information into feature engineering.

---

## 8. SHAP — Model Interpretability

SHAP (SHapley Additive exPlanations) quantifies each feature's contribution to every individual prediction.

### What SHAP tells you

- Which features the model actually learned to use
- Direction of effect — does a high value push toward positive or negative prediction?
- Magnitude — how much does each feature move the prediction score?
- Individual explanations — why was this specific patient flagged?

### What SHAP does NOT tell you

- Feature redundancy — two correlated features can both have high SHAP
- Relationships between features themselves

### SHAP vs built-in feature importance

Built-in importance (gain, weight, cover) measures how often and how much a feature is used in tree splits. A feature used in many shallow splits ranks high on weight but may contribute little to the final probability output.

SHAP traces each feature's contribution through the entire tree structure to the final prediction value. More accurate, especially for features with subtle but consistent effects.

### Four key SHAP plots

| Plot | What it shows |
|------|--------------|
| Summary bar | Global feature importance ranked by mean \|SHAP\| |
| Beeswarm | Direction and magnitude of each feature across all predictions |
| Waterfall | Individual patient explanation — how the prediction was built step by step |
| Dependence | How feature value relates to SHAP value — reveals thresholds and non-linearity |

### Using SHAP for feature selection

Compute mean |SHAP| for each feature across all training predictions. Features with mean |SHAP| near zero contributed nothing to any prediction — the model already ignores them. Removing them formalizes what regularization was doing implicitly.

Threshold: features with mean |SHAP| < 0.05 are candidates for removal. Adjust based on how many features you want to keep and the EPV constraint.

---

## 9. EPV — Events Per Variable

Events Per Variable measures the ratio of positive cases to features.

```
EPV = number of positive cases / number of features
```

### Why EPV matters

With low EPV, a model has more degrees of freedom than positive examples to fit. It can find spurious patterns in noise features that happen to correlate with the target in the training set but do not generalize.

### Guidelines

| EPV | Interpretation |
|-----|---------------|
| ≥ 10 | Safe for logistic regression |
| 5–10 | Acceptable with regularization |
| 1–5 | Risky — strong regularization required, monitor train/test gap |
| < 1 | Very high overfitting risk |

### EPV is model-dependent

Tree models with regularization (max_depth, min_child_weight, colsample_bytree) are more robust to low EPV than logistic regression. The EPV concern is most serious for LR, moderate for Random Forest, less serious for XGBoost/LightGBM with proper tuning.

### How to improve EPV

1. Collect more positive cases (most impactful)
2. Reduce feature count via SHAP selection
3. Use strong regularization to prevent the model from using noise features
4. Two-pass strategy: train on all features, SHAP-select, retrain on reduced set

---

## 10. Threshold Selection

Model outputs are probabilities (0–1). A threshold converts probabilities into binary predictions. Choosing the right threshold is as important as model performance.

### Default threshold of 0.5 is rarely correct for imbalanced problems

At 1.82% positive rate, the model's probabilities are naturally low for all patients. A threshold of 0.5 might flag almost no one. The optimal threshold is often 0.03–0.15 for heavily imbalanced problems.

### Three approaches

**Recall floor approach:**
Find threshold where recall ≥ X (e.g. 0.80), then maximize precision within that constraint. Use when missing cases is the primary concern and you want to guarantee a minimum recall level.

```python
precisions, recalls, thresholds = precision_recall_curve(y_train, oof_probs)
valid = [(p, r, t) for p, r, t in zip(precisions, recalls, thresholds) if r >= 0.80]
_, _, threshold = max(valid, key=lambda x: x[0])
```

**MCC maximization:**
Find threshold that maximizes MCC. More balanced — no hard constraint on any single metric. Recommended when costs of FP and FN are similar.

```python
thresholds = np.linspace(0.01, 0.99, 200)
mcc_scores = [matthews_corrcoef(y_train, (oof_probs >= t).astype(int)) for t in thresholds]
threshold  = thresholds[np.argmax(mcc_scores)]
```

**Always use OOF predictions for threshold selection — not the test set.**

Finding the threshold on the test set tunes it to that specific random split. OOF predictions give an honest estimate of what threshold works in general.

### Threshold is a clinical/business decision

Different stakeholders may want different thresholds:
- Clinical screening tool (missing a case is catastrophic) → low threshold, high recall
- Resource-constrained review process → higher threshold, fewer false alarms
- Report multiple thresholds and let the end user choose based on their risk tolerance

---

## 11. Model Selection Framework

Do not pick models randomly. Evaluate each against your problem profile first.

### Five dimensions to check before choosing a model

| Dimension | Questions to ask |
|-----------|-----------------|
| Data size | How many samples? How many positive cases? |
| Feature type | Continuous, categorical, binary flags, mixed? |
| Missing data | Are NaN values present? |
| Class imbalance | How severe? Does the model handle it natively? |
| Interpretability | Do you need to explain individual predictions? |

### When NOT to use Neural Networks

- Fewer than ~10,000 positive training cases
- Structured tabular data with engineered features (gradient boosting consistently outperforms NNs here)
- Missing values present (no native NaN handling)
- Interpretability required in clinical/regulated domain
- Limited tuning time (NNs have far more hyperparameters)

Neural networks excel at unstructured data — images, text, audio, signals. For structured tabular data with engineered features, gradient boosted trees are the industry standard.

### When to use Logistic Regression

- As a baseline — always run LR first. If a complex model cannot beat it, something is wrong with the features or the complex model is overfitting.
- When interpretability via coefficients is required
- When EPV is healthy (≥ 10)
- Note: with EPV ~1.0, LR will struggle significantly — consider ElasticNet (L1+L2) which performs automatic feature selection

### Gradient boosting family comparison

| Model | Best for | Watch out for |
|-------|---------|---------------|
| XGBoost | Small-medium structured data, reliable default | Slower than LightGBM |
| LightGBM | Large datasets, speed matters | Overfits small datasets without careful `num_leaves` tuning |
| CatBoost | Many categorical features | Main advantage negated when cats are already encoded |
| Random Forest | Different architecture perspective, overfit resistance | Needs imputation, slower than boosting |

---

## 12. Feature Importance — Built-in vs SHAP

### XGBoost built-in importance types

| Type | Measures | Problem |
|------|---------|---------|
| `weight` | How many times feature used in splits | Biased toward continuous features used in many small splits |
| `gain` | Average loss reduction per split | Better but still not directional |
| `cover` | Average samples affected per split | Less commonly used |

`feature_importances_` attribute uses gain by default.

### Why built-in importance can mislead

A feature used in 50 shallow splits ranks high on weight but may contribute little to the final probability. Built-in importance measures tree structure usage, not prediction contribution.

### SHAP is more accurate

SHAP traces each feature's contribution through the entire tree to the final output. A feature with subtle but consistent directional effects (like `plateletNum_cv`) shows up strongly in SHAP but weakly in built-in importance.

### Practical guidance

- **Quick check:** built-in importance is fine and usually correctly identifies the top 2–3 features
- **Feature selection:** use SHAP — more reliable for identifying truly useless features
- **Presentation/clinical reporting:** always use SHAP — directional, individual-level, defensible

---

## 13. Preprocessing — Imputer and Scaler

### SimpleImputer

Replaces NaN values with a computed statistic (median, mean, most frequent).

**When needed:** Any sklearn model that cannot handle NaN — Logistic Regression, Random Forest, SVM.
**When NOT needed:** XGBoost and LightGBM handle NaN natively.

**Critical rule:** Fit imputer on training data only. Transform both train and test using train-derived statistics.

```python
imputer     = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)   # fit on train
X_test_imp  = imputer.transform(X_test)         # transform using train stats
```

If fitted on the full dataset first, test set values influence the median used for imputation — data leakage.

Why median over mean: median is more robust to outliers. A single extreme lab value does not skew the imputed median the way it would the mean.

### StandardScaler

Transforms each feature to have mean=0 and standard deviation=1.

```
scaled_value = (value - mean) / std
```

**When needed:** Logistic Regression, SVM, any distance-based model. Without scaling, features on large scales dominate gradient steps.
**When NOT needed:** Tree models (XGBoost, LightGBM, Random Forest). They split on thresholds — scale is irrelevant.

**Critical rule:** Fit scaler on training data only. Same leakage concern as imputer.

### Pipeline — cleanest way to chain preprocessing

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('model',   LogisticRegression(C=0.01, class_weight='balanced'))
])

pipeline.fit(X_train, y_train)   # fits imputer and scaler on X_train automatically
pipeline.predict_proba(X_test)   # applies train-fitted imputer and scaler to X_test
```

Pipeline ensures imputer and scaler are always fitted on training data and applied correctly to test data — eliminates the leakage risk automatically.

---

## 14. Two-Pass Modeling Strategy

A practical approach to address high feature count relative to positive cases (low EPV).

### Why two passes

With EPV ~1.0, the model has more parameters to fit than positive examples. Many features are noise — the model partially fits them during training but they do not generalize.

**Counterintuitive insight:** Removing noise features can improve performance because the model stops wasting capacity on spurious patterns. Fewer features with the same positive cases = higher EPV = more stable model.

### The strategy

**Pass 1 — Exploration model (all features)**
- Train on full feature set
- Use strong regularization
- Primary goal: identify which features the model actually uses
- Run SHAP to rank all features by mean |SHAP|

**Pass 2 — Production candidate (SHAP-selected features)**
- Keep features with mean |SHAP| above a threshold (e.g. 0.05)
- Retrain with new Optuna tuning on reduced feature set
- Compare Pass 1 vs Pass 2 CV scores and OOF→Test gap
- If Pass 2 achieves equivalent or better performance → prefer Pass 2

### Decision rule

If Pass 2 CV score is within 5% of Pass 1 → choose Pass 2. Simpler model, better EPV, more interpretable, easier to maintain.

### Expected outcomes

- **Performance improves or stays flat:** Most likely. Noise features removed, model focuses on real signal.
- **Performance drops slightly:** Acceptable if the reduction is small and interpretability/stability improve.
- **Performance drops significantly:** Some removed features carried real signal SHAP underestimated. Use a lower SHAP threshold and retry.

---

## 15. Clinical Translation of ML Models

A predictive model is a tool, not the end goal. The value lies in translating model findings into clinical action.

### Three deliverables beyond the model

**1. Lab threshold table**
For each key feature, find the value where risk sharply increases. Use sensitivity/specificity tradeoff curves — the inflection point where sensitivity rises steeply then flattens is the clinically meaningful threshold. The graph makes the decision visually obvious without needing statistics.

**2. Simple risk scoring rule**
Take top 4–6 SHAP features. Assign point values reflecting clinical importance. Define a score cutoff. A clinician can apply this from morning lab results without a computer. Evaluate at every cutoff level:

```
Score ≥ 2: very sensitive, many false alarms
Score ≥ 3: balanced — often the right clinical choice
Score ≥ 4: specific, misses more cases
```

**3. High-risk patient profile**
Describe the patient most likely to have the condition in plain language. Actionable for admission screening.

### Important limitations to always state

- Thresholds derived from one dataset need prospective validation before clinical adoption
- Correlation vs causation — a feature predicting a condition may be a symptom, not a cause
- Single institution data may not generalize
- Any clinical deployment requires regulatory approval (FDA/CE as Software as a Medical Device)

### Why this matters in an interview

A data scientist who delivers only a model is a technician. One who asks "how does this become clinical action?" and bridges the model to practical use is a partner to the medical team. Demonstrating this thinking is a strong differentiator.

---

*This document will be updated as new concepts are encountered.*

---

## 16. Multimodal ML — Combining Text and Tabular Data

When a prediction problem has both structured tabular data (labs, medications, demographics) and unstructured text data (clinical notes), you have a multimodal ML problem. The core challenge is that these two data types require fundamentally different model architectures and cannot simply be concatenated.

### The three architectures — and why we chose one

**Architecture 1 — Feature concatenation (frozen BERT)**
Extract embeddings from a pre-trained BERT model without fine-tuning. Apply dimensionality reduction (PCA) to reduce 768 dimensions to 50. Concatenate with tabular features. Feed everything into XGBoost.

Problem: BERT was not trained for your specific task. The embeddings may not encode the right signal. XGBoost splits on individual dimensions one at a time — this destroys the geometric structure of embeddings where meaning is distributed across all dimensions simultaneously. Also severely worsens EPV.

**Architecture 2 — Model stacking**
Train two base models independently — ClinicalBERT on notes, XGBoost on tabular features. Feed their predicted probabilities into a CatBoost meta-learner that learns the optimal combination.

Why we rejected it:
- Requires enough data for a third model (meta-learner) to generalize — 142 positive cases is insufficient
- Three models to train, tune, and maintain — high complexity
- Leakage risk: meta-learner must be trained on out-of-fold base model predictions, not in-sample predictions — easy to get wrong
- Interpretability is very hard — explaining a meta-learner on top of two black boxes
- Works well when base models make genuinely different errors — valid here, but the data constraint overrides the benefit
- A soft voting ensemble (weighted average of two probabilities) achieves most of the complementarity benefit with far less complexity

**Architecture 3 — Fine-tuned BERT + tabular features → XGBoost (chosen)**
Fine-tune ClinicalBERT as a standalone binary classifier on notes. Use it to produce one probability per patient. Add that probability as a single new feature to the tabular feature matrix. Train XGBoost on the enriched feature set.

Why this is right:
- BERT's output is compressed to one meaningful number — the model's confidence from the note alone
- XGBoost can split on this meaningfully: "is bert_note_probability > 0.6?"
- SHAP works cleanly — you see exactly how much the note probability contributes vs each lab feature
- EPV: only one feature added (139 → 140), EPV unchanged
- If SHAP shows bert_note_prob ranks near zero → notes add no signal (valid scientific finding)
- If SHAP shows bert_note_prob ranks high → notes genuinely complement labs
- Simple, interpretable, maintainable

---

## 17. Transfer Learning and Fine-Tuning

### Transfer learning — the core idea

Take knowledge learned solving one problem and apply it to a different but related problem, rather than learning from scratch.

General pattern:
```
Large dataset + general task → pre-train → general model
                                                ↓  transfer
Small dataset + specific task → fine-tune → specialized model
```

The pre-trained model does not start from zero on the new task. It brings knowledge from the general task that accelerates learning on the specific task and requires far less labeled data.

### Three well-known examples

**ImageNet → Medical imaging:** A model pre-trained on 1 million general images learns edges, shapes, textures. Fine-tuned on 500 chest X-rays, it detects pneumonia without needing 500,000 X-rays.

**BERT → Sentiment analysis:** BERT pre-trained on Wikipedia and books learns language structure. Fine-tuned on 10,000 movie reviews, it classifies sentiment. It already understands English.

**ClinicalBERT → Pancytopenia:** ClinicalBERT pre-trained on millions of clinical notes learns medical language. Fine-tuned on 9,942 labeled patient visits, it predicts pancytopenia.

### Fine-tuning IS transfer learning

Fine-tuning is the specific technique of continuing to train a pre-trained model on a new task. The pre-trained weights are not frozen — they get slightly adjusted to make them useful for the specific prediction problem.

The pre-training head (e.g. masked word prediction for BERT) is discarded. A new task-specific head (e.g. binary classifier) is added and trained from scratch. The pre-trained layers are updated with a very small learning rate so they adapt without destroying the general knowledge they already contain.

### Three-stage transfer learning — ClinicalBERT's full lineage

ClinicalBERT is itself a product of transfer learning:

```
General text (Wikipedia, books)
    ↓  pre-train BERT
General language model
    ↓  domain adaptation on clinical notes (MIMIC-III)
ClinicalBERT — clinical language model
    ↓  task fine-tuning on pancytopenia labels
Our model — pancytopenia predictor
```

Each stage transfers knowledge to the next, getting more specific each time.

### The spectrum of fine-tuning approaches

**Feature extraction (fully frozen):** Use pre-trained model as-is, extract embeddings, never update weights. Fastest, requires least data. May not capture task-specific signal well.

**Partial fine-tuning (freeze lower layers):** Freeze bottom layers, fine-tune top layers + classification head. Sweet spot for small datasets. Preserves general knowledge in lower layers, adapts upper layers to the task.

**Full fine-tuning (update everything):** Update all layers. Most powerful. Risks catastrophic forgetting on small datasets — pre-trained knowledge gets overwritten by noisy gradients from too few examples.

For our use case with 142 positive training cases — partial fine-tuning is correct. Freeze the bottom 10 of 12 BERT layers. Fine-tune the top 2 layers and the classification head. This preserves the clinical language knowledge while adapting the model to pancytopenia prediction.

### Catastrophic forgetting

When you fine-tune aggressively on a small dataset, the model's weights shift far from their pre-trained values. The general knowledge that made the pre-trained model valuable gets overwritten. The model may achieve good training accuracy but loses the ability to generalize.

Prevention:
- Use a very small learning rate for BERT layers (e.g. 2e-5) — much smaller than for the classification head (1e-3)
- Freeze most layers
- Use early stopping on validation MCC
- Fine-tune for few epochs (3–10, not 100)

---

## 18. ClinicalBERT — Architecture and Key Concepts

### What BERT is

BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based language model. It reads text in both directions simultaneously — left to right and right to left — giving it full context for every word. This bidirectionality is what made it a breakthrough over earlier sequential models.

### The [CLS] token

BERT prepends a special [CLS] (classification) token to every input sequence. After passing through all transformer layers, the [CLS] token's output embedding accumulates a representation of the entire input sequence. This 768-dimensional vector is what gets used for classification tasks.

Think of it as a compressed summary of the entire document, learned through training to be useful for downstream prediction.

### Why ClinicalBERT specifically

Standard BERT was pre-trained on Wikipedia and BookCorpus — general English text. Clinical notes use completely different language: abbreviations (SOB = shortness of breath, c/o = complains of, Hx = history), drug names, procedures, anatomical terms, and sentence fragments. A model trained on Wikipedia does not understand "Pt c/o SOB, Hx of MDS, started on AZA last month."

ClinicalBERT was fine-tuned on MIMIC-III — a large dataset of de-identified clinical notes from ICU patients. It understands clinical language natively.

### The 512 token limit

BERT has a maximum input length of 512 tokens (approximately 380 words). Clinical notes — especially discharge summaries — can be thousands of words. Strategies:

| Strategy | Description | Best for |
|----------|-------------|---------|
| Truncate first 512 | Take beginning of note | Admission H&P — context is at the start |
| Truncate last 512 | Take end of note | Discharge summary — diagnosis is at the end |
| Chunk and average | Split into 512-token chunks, embed each, average all embeddings | Full note coverage |
| Hierarchical | Embed chunks, pass chunk embeddings to another model | Research-grade, complex |

For a prospective model using admission H&P — first 512 tokens is appropriate.
For a retrospective model using discharge summaries — chunk and average captures the full clinical picture.

### The label leakage problem in clinical notes

Discharge summaries for pancytopenia patients will almost certainly contain the word "pancytopenia" or the ICD-10 codes. A model trained on these notes could learn to simply search for the diagnosis word rather than learning clinical predictors.

For a retrospective model this is somewhat acceptable — note and label are both discharge-time information. But it means the model is detecting the condition rather than predicting it.

For a prospective model — notes must be restricted to text written before the condition was diagnosed. Either use admission H&P only, or exclude any text written after the first suspicious lab result.

Always preprocess notes to remove direct mentions of the target condition before training.

### Class imbalance in neural net training

The 1:55 imbalance cannot be handled with `scale_pos_weight` in BERT fine-tuning (that is XGBoost-specific). Instead use weighted binary cross-entropy loss:

The loss function is modified so that errors on positive cases are penalized 55x more than errors on negative cases. The model learns to treat every positive case as worth 55 negative cases — equivalent to `scale_pos_weight` but implemented in the loss function rather than the model structure.

---

## 19. The Right Way to Combine Fine-Tuned BERT with XGBoost

### Why feeding 768 BERT dimensions into XGBoost is wrong

XGBoost builds trees by finding split points on individual features — threshold-based decisions on one dimension at a time. BERT embeddings are dense and entangled — no single dimension has independent meaning. The meaning is distributed across all 768 dimensions simultaneously.

Splitting on individual embedding dimensions destroys the geometric structure that makes embeddings powerful. Additionally, 768 extra features with 142 positive cases makes EPV approximately 0.18 — catastrophic.

### The correct approach — BERT probability as one feature

**Stage 1: Fine-tune ClinicalBERT as a standalone binary classifier**

Train BERT with a classification head on notes alone. The model learns to predict pancytopenia from text. The output is one number per patient — the BERT-predicted probability from the note alone. This is a compressed, semantically meaningful summary of the entire note distilled to the dimension XGBoost can use correctly.

**Stage 2: Add BERT probability as one new feature to XGBoost**

The fine-tuned BERT produces `bert_note_probability` for every patient. This becomes column 140 in the feature matrix. XGBoost can now ask "is bert_note_probability > 0.6?" — a meaningful threshold split.

Everything else in the XGBoost pipeline is identical. Same Optuna, same 5-fold stratified CV, same OOF MCC threshold, same test set used once.

### The critical OOF requirement for Stage 1

If you fine-tune BERT on the full training set and then use its predictions on that same training set as a feature for XGBoost — BERT has already seen those patients. Its predictions will be overfit. XGBoost will learn to trust an artificially inflated signal.

Correct approach: use out-of-fold BERT predictions for training patients. BERT trained on folds 2–5 predicts fold 1. BERT trained on folds 1,3,4,5 predicts fold 2. Each training patient gets a BERT prediction from a model that never saw them. Exactly the same OOF logic as the XGBoost threshold selection.

**Same fold splits must be used across both stages.** Same random seed, same stratification. This ensures no patient appears in any fold whose BERT prediction was made by a model trained on XGBoost's validation data.

### The full pipeline

```
Stage 1 — Fine-tune ClinicalBERT
─────────────────────────────────────────────────────
1. Preprocess notes — clean, redact target mentions, truncate/chunk
2. Use same 5-fold stratified split as XGBoost pipeline (same seed)
3. For each fold: fine-tune BERT on fold train, predict fold val
4. Collect OOF predictions → bert_note_prob for all training patients
5. Fine-tune final BERT on full training set → predict test patients
6. bert_note_prob is now available for all patients

Stage 2 — XGBoost with enriched features
─────────────────────────────────────────────────────
1. Add bert_note_prob as column 140 to feature matrix
2. Run existing pipeline: Optuna + 5-fold CV + OOF MCC threshold
3. SHAP: check if bert_note_prob contributes
4. Final evaluation on test set — used exactly once
```

### What SHAP tells you about the note signal

After training the enriched XGBoost:

- **bert_note_prob ranks high in SHAP** → notes genuinely complement lab values. The clinical narrative captures signal that structured data misses. Invest further in NLP pipeline quality.
- **bert_note_prob ranks near zero in SHAP** → for this condition, lab values already capture everything the notes contain. Notes add no incremental value. Valid scientific finding — do not invest further in NLP for this specific prediction task.

Either outcome is informative and guides next steps.

### Optional extension — PCA on fine-tuned embeddings

Instead of just the final probability, extract the [CLS] embedding after fine-tuning and reduce to 5–10 dimensions via PCA. The fine-tuned [CLS] embedding is far more task-relevant than the pre-trained one — even a small PCA reduction captures the most important variation for pancytopenia.

This gives XGBoost slightly more to work with from the note without the dimensionality explosion. Whether this beats the single probability depends on data volume. Start with the single probability — clean, interpretable, and likely captures most of the signal.

---

## 20. Prescription Data as Additional Feature Source

### What prescription data adds that administration records do not

Administration records capture what was actually given to the patient — downstream of clinical decisions. Prescription data captures clinical intent — what the physician decided to order and why.

| Data source | Captures | Limitation |
|-------------|---------|------------|
| Administration records (current) | Drugs given, timing, amounts | No clinical context, no intent |
| Prescription data | Clinical intent, prescribing specialty, indication, duration | Still structured — no narrative |
| Visit notes | Clinical reasoning, history, symptoms | Unstructured — requires NLP |

### High-value fields in prescription data for pancytopenia

**Prescribing physician specialty** — If an oncologist or hematologist is involved, that is a strong prior for conditions causing pancytopenia. A single specialty flag could outperform all 50 drug name flags combined.

**Indication / diagnosis codes on prescriptions** — If methotrexate is prescribed with ICD-10 code M05 (rheumatoid arthritis), the clinical context is clear. This links drug to condition in a way administration records cannot.

**Chemotherapy / immunosuppressant flag** — The primary drug-induced causes of pancytopenia. Our current model found near-zero medication signal partly because we could not identify bone marrow suppressive drugs. Prescription data with indication codes would fix this directly.

**Duration of treatment** — A patient on methotrexate for 3 months is fundamentally different from one prescribed it for 3 days. Chronic exposure to bone marrow suppressive drugs accumulates risk — invisible in administration records.

**Scheduled vs PRN** — Scheduled medications reflect ongoing treatment. PRN (as needed) reflects acute symptom management. This distinction is lost in our current data.

### EPV implication

Adding prescription features increases feature count. Introduce carefully:
1. Start with high-signal fields only — specialty flag, chemo flag, duration (5–10 features)
2. Run SHAP on enriched model
3. Only expand if SHAP confirms signal

---

*Document updated: April 2026*

---

## 21. Clinical Notes — Real-World Source, Acquisition, and Privacy

### Source — Epic EHR

In practice, clinical notes for this project would be extracted from Epic — one of the most widely used EHR systems in the United States. Epic stores all clinical documentation tied to the patient encounter through an encounter ID, which maps directly to our `visit_id`.

The relevant note types in Epic for pancytopenia prediction:

| Note Type | Written by | Timing | Value for our model |
|-----------|-----------|--------|---------------------|
| Admission H&P | Admitting physician | Within hours of arrival | Highest — chief complaint, past history, medications, exam |
| Progress Notes | Attending + residents | Daily throughout stay | High — evolving clinical picture, lab discussions |
| Discharge Summary | Attending physician | At discharge | Highest information — but highest leakage risk |
| Consultation Notes | Specialist physician | When called in | High — hematology consult is itself a signal |

**For our pipeline: Admission H&P** is the right note type. It is available at admission (enabling prospective use), contains the richest clinical context, and has the lowest leakage risk compared to discharge summaries.

### How notes are stored in Epic

Notes are stored in a one-to-many relationship with visits — one visit can have many notes. The raw extract from Epic looks like:

```
note_id | visit_id | patient_id | note_type     | author_specialty | note_datetime | note_text
--------|----------|------------|---------------|-----------------|---------------|----------
N001    | 642556   | P10023     | Admission H&P | Hospitalist     | 2022-01-01    | "Patient is a..."
N002    | 642556   | P10023     | Progress Note | Hospitalist     | 2022-01-02    | "Day 2..."
N004    | 1025760  | P20045     | Admission H&P | Oncology        | 2022-07-06    | "Patient is a..."
```

To get to visit-level for our pipeline — select one note per visit (the Admission H&P). This gives a clean one-to-one join on `visit_id`. No aggregation logic needed.

### De-identification before ML training

**HIPAA applies even for in-house models.**

Using clinical notes to train a machine learning model is a secondary use of Protected Health Information (PHI). HIPAA governs secondary use regardless of whether the model is built and deployed entirely within the same health system. There is no in-house exception.

PHI includes: patient names, dates (admission, discharge, DOB, procedures), geographic identifiers smaller than state level, phone numbers, email, SSN, MRN, device identifiers, photographs, and biometric identifiers.

**What a health system like Kaiser would do:**

Apply Safe Harbor de-identification — remove all 18 HIPAA-defined PHI identifiers before the data reaches the ML team. Replace with standardized placeholders:

```
Before: "Patient John Smith, DOB 03/14/1972, admitted 01/15/2023 
         to Kaiser Permanente Oakland..."

After:  "Patient [NAME], DOB [DATE], admitted [DATE] 
         to [HOSPITAL]..."
```

Tools used: Epic's built-in de-identification module, Amazon Comprehend Medical, Microsoft Presidio, or custom NLP pipelines.

**Why de-identification is required even in-house:**

- **Legal:** OCR (HIPAA enforcement) fines for unauthorized secondary use of PHI. No carve-out for internal ML.
- **Model risk:** Neural networks — especially large language models — can memorize training data. A fine-tuned BERT that reproduces PHI from training in its outputs constitutes a breach, even if unintentional.
- **Audit risk:** Healthcare organizations are regularly audited. Undocumented PHI use in ML pipelines creates compliance exposure.
- **Ethics:** Patients consented to their data being used for treatment — not for training ML models. De-identification or IRB waiver is how secondary use is authorized.

**The four de-identification routes:**

| Option | Description | Common use |
|--------|-------------|-----------|
| Safe Harbor | Remove all 18 PHI identifiers, replace with placeholders | Standard for ML training data |
| Expert Determination | Statistician certifies re-identification risk is very small | Less common, requires documentation |
| Limited Dataset + DUA | Keep some dates/geography, remove direct identifiers | When temporal data is critical |
| Federated Learning | Model trains locally, gradients shared — PHI never extracted | Emerging, privacy-preserving |

For pancytopenia modeling — Safe Harbor de-identification is the standard path.

**Important nuance:** De-identified data under HIPAA Safe Harbor still carries some re-identification risk when combined with external data sources. Additional controls typically applied: access limited to approved team members, data on secure servers only, audit logs of all access, model outputs reviewed before publication.

### Format for our pipeline

Notes received from Epic (post de-identification) become a CSV:

```
visit_id  | note_text
----------|----------
642556    | "Patient [NAME] is a 50yo [GENDER] admitted on [DATE]..."
1025760   | "Patient [NAME] is a 72yo [GENDER] with history of..."
```

One row per visit (Admission H&P selected). Joined to feature matrix on `visit_id`. This is the format our synthetic notes follow.

### Synthetic notes — what they are and their limitation

For pipeline development and learning purposes, we generate synthetic notes that simulate what real de-identified Epic Admission H&P notes look like. These are generated to be:

- Realistic in language and structure
- Varied across physician writing styles
- Calibrated so notes are a supporting signal — not overriding lab values
- Inclusive of realistic noise — abbreviations, typos, boilerplate, incidental findings

**Limitation that must be stated when presenting:** Synthetic notes do not capture the full complexity and distribution of real clinical language. A ClinicalBERT model fine-tuned on synthetic notes may not generalize to real EHR notes the same way a model trained on actual de-identified data would. In production, synthetic notes would be replaced with real de-identified Admission H&P notes from Epic.

---

---

## 22. PHI Screening and HIPAA Compliance for Clinical Notes

### Why a dedicated screening pipeline is necessary

Even when notes are extracted from Epic with de-identification applied, a secondary screening step is best practice. De-identification tools have known gaps. A missed patient name or medical record number in the training data can constitute a HIPAA breach if the model memorizes it. The screening pipeline is the last line of defense before data reaches the ML team.

### Microsoft Presidio

Presidio is an open-source data protection SDK from Microsoft, available under the MIT license on GitHub. Free to use, no licensing cost.

It does two things:
- **Analysis** — detects PHI entities in text using named entity recognition, regex patterns, and rule-based detectors
- **Anonymization** — replaces detected entities with standardized placeholders

PHI types Presidio detects natively: PERSON, DATE_TIME, LOCATION, PHONE_NUMBER, EMAIL_ADDRESS, US_SSN, US_BANK_NUMBER, CREDIT_CARD, IP_ADDRESS, URL, NRP (nationality/religion/political group), MEDICAL_LICENSE

Installation:
```bash
pip install presidio-analyzer presidio-anonymizer
python -m spacy download en_core_web_lg
```

### Known gaps in Presidio for clinical text

Presidio is strong on standard PHI but has gaps in clinical-specific identifiers:

| Gap | Example | Fix |
|-----|---------|-----|
| Medical record numbers in non-standard formats | "MRN: 1234567" | Custom regex recognizer |
| Physician names mid-sentence in clinical style | "per Dr. Williams" | Presidio catches most but not all |
| Pager numbers | "pager 4521" | Custom regex |
| Room/bed identifiers with patient context | "room 412B" | Custom regex |
| Date shifting not built in | Dates replaced with placeholder but not shifted | Separate date shifting step if needed |

### Three-layer screening approach

**Layer 1 — Presidio** for standard PHI (names, dates, locations, phone numbers, SSN, email, URLs, IP addresses)

**Layer 2 — Custom regex patterns** for clinical-specific identifiers Presidio misses:
- Medical record numbers
- SSN alternative formats
- Pager numbers
- Room/bed numbers

**Layer 3 — Audit flag** — after anonymization, flag any note containing patterns that may be residual PHI. Manual review of flagged notes before training.

### The pipeline output

Input: `visit_id, note_text` CSV from Epic

Output: `visit_id, note_text_clean, phi_entities_found, residual_phi_flag` CSV

- `note_text_clean` — de-identified text ready for ML training
- `phi_entities_found` — count of entities detected and replaced (audit trail)
- `residual_phi_flag` — 1 if suspicious patterns remain after cleaning (needs manual review)

### For synthetic notes

Synthetic notes use placeholders by construction — already de-identified. Running them through the screening pipeline anyway is best practice because it:
- Simulates the real production pipeline end-to-end
- Catches anything accidentally made too realistic
- Validates the pipeline before real data is processed
- Documents compliance due diligence

### HIPAA Safe Harbor — 18 PHI identifiers to remove

The complete list of PHI identifiers that must be removed under HIPAA Safe Harbor:

1. Names
2. Geographic subdivisions smaller than state
3. All dates except year (for patients over 89, ages and dates must be aggregated)
4. Phone numbers
5. Fax numbers
6. Email addresses
7. Social security numbers
8. Medical record numbers
9. Health plan beneficiary numbers
10. Account numbers
11. Certificate/license numbers
12. Vehicle identifiers and serial numbers
13. Device identifiers and serial numbers
14. Web URLs
15. IP addresses
16. Biometric identifiers (finger and voice prints)
17. Full-face photographs and comparable images
18. Any other unique identifying number or code

Presidio covers most of these. Custom regex covers the clinical-specific ones (8, 11 for medical licenses, 12–13 for device serials in procedure notes).

---

---

## 22. PHI Screening and HIPAA Compliance Pipeline

### Why a dedicated pipeline

Even when notes arrive from Epic already de-identified, a secondary screening pass is best practice before ML training. De-identification tools are imperfect. A layered approach catches what any single tool misses.

### Microsoft Presidio

Open source (MIT license), available on GitHub. One of the best tools for PHI detection and anonymization in text. Used by healthcare organizations and ML teams worldwide.

Two core components:
- **presidio-analyzer** — detects PHI entities using NER models, regex patterns, and rule-based detectors
- **presidio-anonymizer** — replaces detected entities with placeholders or synthetic values

Detects out of the box: names, dates, locations, phone numbers, email addresses, SSN, credit card numbers, IP addresses, URLs, medical license numbers.

Installation:
```bash
pip install presidio-analyzer presidio-anonymizer
python -m spacy download en_core_web_lg
```

### Three-layer screening approach

**Layer 1 — Presidio**
Handles standard PHI categories. Covers names, dates, locations, phone numbers, SSN, email, URLs.

**Layer 2 — Custom clinical regex**
Presidio misses clinical-specific identifiers. Add custom patterns for:
- Medical record numbers (MRN): `\bMRN[:\s#]*\d{4,10}\b`
- Pager numbers: `\b(pager|pg)\s*\d{4,7}\b`
- Room and bed numbers: `\b(room|bed|rm)\s*\d{1,4}[A-Z]?\b`
- SSN alternative formats: `\b\d{3}[-\s]\d{2}[-\s]\d{4}\b`

**Layer 3 — Residual audit flag**
After anonymization, scan output for patterns that could still be PHI. Flag any note with residual risk for manual review. A flagged note is not automatically excluded — it gets human eyes before training.

### Output format

The pipeline outputs a CSV with:
- `visit_id` — unchanged
- `note_text_clean` — de-identified note text
- `phi_entities_found` — list of PHI types detected and removed
- `residual_phi_flag` — 1 if manual review recommended, 0 if clean

Notes with `residual_phi_flag = 1` should be manually reviewed before including in training data.

### For synthetic notes

Synthetic notes generated with `[NAME]`, `[DATE]`, `[HOSPITAL]`, `[PHYSICIAN]` placeholders are already de-identified by construction. Running them through the PHI pipeline anyway simulates the real production process and validates the pipeline works correctly before applying it to real data.

### HIPAA Safe Harbor — 18 identifier categories

For reference, the 18 PHI identifiers that must be removed under HIPAA Safe Harbor:

Names, geographic subdivisions smaller than state, dates (except year) related to an individual, phone numbers, fax numbers, email addresses, SSN, medical record numbers, health plan beneficiary numbers, account numbers, certificate/license numbers, vehicle identifiers, device identifiers, URLs, IP addresses, biometric identifiers (fingerprints, voiceprints), full-face photographs, any other unique identifying number or code.

---
