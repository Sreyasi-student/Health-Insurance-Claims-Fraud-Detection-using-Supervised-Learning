# Healthcare Claims Fraud Detection: 12-Model Comparison

Supervised classification of health insurance claims as fraudulent or legitimate. The notebook compares **12 model configurations** across five algorithms, crossing two design choices:

- **Missing-value handling:** impute (median / mode) vs. leave NaN in place (native handling)
- **Class-imbalance handling:** SMOTENC oversampling vs. class weights (`class_weight` / `scale_pos_weight`)

Models are ranked on **AUC-PR** (area under the precision-recall curve), which suits a dataset where only ~8% of claims are fraud.

> **Intended use:** a post-adjudication audit tool. `Claim_Status` and `Approved_Amount` are described in the notebook as unavailable at prediction time, which is what makes this an audit setting rather than a pre-payment screen. See [Known Issues](#known-issues) before relying on any results.

---

## Dataset

- **Source:** [Healthcare Fraud Detection Dataset](https://www.kaggle.com/datasets/nudratabbas/healthcare-fraud-detection-dataset) (Kaggle, `nudratabbas`), downloaded automatically via `kagglehub`
- **Size:** 10,000 claims, 20 raw columns, 300 unique providers
- **Target:** `Is_Fraud` (0 = legitimate, 1 = fraud), about 8.3% positive
- **Missing values:** 3.5% (350 rows each) in `Insurance_Type`, `Provider_Specialty`, `Prior_Visits_12m`

## Feature Engineering

| Step | Detail |
|---|---|
| Dropped | `Claim_ID`, `Provider_ID` (high-cardinality identifiers; would let models memorise providers), `Claim_Submission_Date` |
| Derived | `Claim_Submission_Month` extracted from the submission date |
| Recast | `Procedure_Code` cast from numeric to categorical (discrete billing codes), with NaN preserved |
| Categorical (9) | `Patient_Gender`, `Diagnosis_Code`, `Procedure_Code`, `Insurance_Type`, `Provider_Specialty`, `Patient_State`, `Visit_Type`, `Chronic_Condition_Flag`, `Claim_Submission_Month` |
| Numeric (6) | `Patient_Age`, `Claim_Amount`, `Approved_Amount`, `Days_Between_Service_and_Claim`, `Length_of_Stay`, `Prior_Visits_12m` |

## Methodology

- **Split:** 80/20 stratified train/test (8,000 / 2,000), `random_state=42`, performed before any preprocessing
- **No leakage by construction:** every step (imputation, scaling, encoding, SMOTENC) lives inside a pipeline, so it is fit on training data only and re-fit on the training portion of each CV fold
- **Tuning:** `RandomizedSearchCV`, 16 iterations, 3-fold stratified CV, scored on AUC-PR. One hyper-parameter grid per algorithm, shared by all its variants
- **Evaluation:** all models are scored on the same untouched test set. Precision, recall and F1 use a 0.5 threshold, which is not comparable between SMOTE and class-weight models, so **AUC-PR is the comparison metric**

### The 12 configurations

| Algorithm | impute + SMOTE | impute + weight | no-impute + weight |
|---|:-:|:-:|:-:|
| Logistic Regression (untuned) | yes | yes | n/a |
| KNN (tuned) | yes | n/a (no class weights) | n/a |
| Decision Tree (tuned) | yes | yes | yes |
| Random Forest (tuned) | yes | yes | yes |
| XGBoost (tuned) | yes | yes | yes |

No-impute + SMOTE is impossible (SMOTENC cannot take NaN). No-impute is also not possible for LR and KNN, which need complete data.

## Results

Random-guess AUC-PR baseline on the test set: **0.083**.

| Model | CV AUC-PR | Test AUC-PR | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| XGB, impute + SMOTE | 0.9830 ± 0.0024 | **0.9866** | 0.9984 | 0.908 | 0.952 | 0.929 |
| XGB, no-impute + weight | 0.9809 ± 0.0030 | 0.9802 | 0.9975 | 0.928 | 0.934 | 0.931 |
| XGB, impute + weight | 0.9806 ± 0.0032 | 0.9800 | 0.9975 | 0.912 | 0.934 | 0.923 |
| LR, impute + SMOTE | 0.9784 ± 0.0058 | 0.9785 | 0.9976 | 0.837 | 0.958 | 0.893 |
| LR, impute + weight | 0.9748 ± 0.0054 | 0.9764 | 0.9972 | 0.744 | 0.982 | 0.847 |
| RF, no-impute + weight | 0.9736 ± 0.0041 | 0.9792 | 0.9975 | 0.943 | 0.904 | 0.923 |
| RF, impute + weight | 0.9733 ± 0.0048 | 0.9794 | 0.9974 | 0.955 | 0.904 | 0.929 |
| RF, impute + SMOTE | 0.9684 ± 0.0002 | 0.9761 | 0.9974 | 0.862 | 0.940 | 0.899 |
| DT, impute + weight | 0.9194 ± 0.0073 | 0.9166 | 0.9457 | 0.861 | 0.898 | 0.879 |
| DT, impute + SMOTE | 0.9171 ± 0.0194 | 0.9402 | 0.9778 | 0.838 | 0.964 | 0.896 |
| DT, no-impute + weight | 0.9141 ± 0.0084 | 0.8902 | 0.9295 | 0.857 | 0.868 | 0.862 |
| KNN, impute + SMOTE | 0.5795 ± 0.0256 | 0.6187 | 0.9387 | 0.296 | 0.946 | 0.451 |

### What the comparison supports

Paired by CV fold, so rows of the same algorithm that differ in one factor isolate that factor:

- **SMOTE vs. class weights:** gains of 0.002 to 0.004 for LR and XGBoost, and no clear difference for DT or RF. The effects are tiny, and with only 3 folds the spread is comparable to the gain, so treat "SMOTE better" as weak evidence at best.
- **Impute vs. native NaN:** no meaningful difference for RF and XGBoost. Imputation is slightly better for Decision Trees (-0.005).
- **Algorithm choice matters far more than either preprocessing choice.** The gap between XGBoost / LR / RF (about 0.97 to 0.98) and DT (about 0.92) and KNN (about 0.58) dwarfs every SMOTE or imputation effect.
- **Top tier is statistically indistinguishable.** XGB, LR and RF all sit within roughly 0.01 AUC-PR of each other, and an untuned Logistic Regression is competitive with tuned XGBoost.

## Known Issues

Read these before treating the numbers as meaningful.

1. **`Approved_Amount` is used as a feature despite the notebook saying it should be excluded.** The markdown states `Claim_Status` and `Approved_Amount` "should be excluded because the information might not be available while prediction", but `Approved_Amount` is in `num_features1` and therefore in the model inputs. `Claim_Status` is correctly excluded. Either remove `Approved_Amount` from the features or drop the claim that it is excluded, then re-run. This could inflate results, and it undermines the "no information unavailable at prediction time" framing.
2. **The scores are suspiciously high.** An untuned Logistic Regression reaching ~0.98 AUC-PR on real claims fraud would be extraordinary. This pattern is typical of a synthetic dataset with fraud injected by simple rules. Results here probably reflect how the data was generated, not real-world difficulty, and should not be expected to transfer to production claims data.
3. **Small positive class.** About 166 fraud cases in the test set. A handful of claims moves precision and recall noticeably, and no confidence intervals are reported on test metrics.
4. **3-fold CV and a 16-iteration random search** give noisy comparisons. The paired-fold "verdicts" rest on three data points per comparison.
5. **Thresholds are fixed at 0.5** for precision, recall and F1, and no threshold selection or cost-based analysis is done. For an audit workflow, the cost of a missed fraud vs. a wasted investigation should drive the threshold.
6. **Provider-level signal is discarded.** `Provider_ID` is dropped, and `Number_of_Claims_Per_Provider_Monthly` is excluded from the numeric features without explanation. Provider behaviour is usually the strongest fraud signal in real claims data.
7. **KNN performs poorly** (AUC-PR 0.58). It is likely hurt by one-hot / ordinal-encoded categoricals in a distance metric and is included mainly as a baseline.

## Requirements

- Python 3.9+
- `scikit-learn >= 1.4` (required for `make_scorer(response_method=...)`)
- `imbalanced-learn`, `xgboost`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `kagglehub`

```bash
pip install scikit-learn imbalanced-learn xgboost pandas numpy matplotlib seaborn kagglehub jupyter
```

## Running

```bash
jupyter notebook Healthcare_Fraud_detection_12models.ipynb
```

Run all cells top to bottom. The dataset downloads automatically through `kagglehub` (a Kaggle account or API credentials may be needed depending on your setup). The full tuning loop runs searches in parallel (`n_jobs=-1`), so expect it to take a while on a laptop.

## Repository Structure

```
.
├── Healthcare_Fraud_detection_12models.ipynb   # EDA, preprocessing, 12-model training and comparison
└── README.md
```

## Suggested Next Steps

- Resolve the `Approved_Amount` inconsistency and re-run
- Validate on a real or independently generated claims dataset
- Add provider-level aggregate features computed without leakage (for example, time-windowed history)
- Use repeated stratified CV and bootstrap confidence intervals for test metrics
- Choose an operating threshold from audit cost assumptions, and report precision at a fixed review budget (for example, top 5% of claims)
- Add calibration plots and feature importance (SHAP) for the top model
