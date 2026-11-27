# Healthcare Insurance Claims Fraud Detection Using Supervised Learning

## Project Overview

This project explores supervised machine learning approaches for
identifying potentially fraudulent healthcare insurance claims. The
objective is to compare multiple classification models and evaluate
different strategies for handling class imbalance and missing values.

The analysis uses a labelled healthcare claims dataset and treats fraud
detection as a binary classification problem, where `Is_Fraud` indicates
whether a claim is fraudulent.

## Dataset

-   **Source:** [Healthcare Fraud Detection Dataset --
    Kaggle](https://www.kaggle.com/datasets/nudratabbas/healthcare-fraud-detection-dataset)
-   **Observations:** 10,000
-   **Original columns:** 20
-   **Target variable:** `Is_Fraud`
-   **Fraud prevalence:** Approximately 8.29%

The dataset contains claim, patient, provider, diagnosis, procedure,
insurance and visit-related information. Missing values occur in
`Insurance_Type`, `Provider_Specialty` and `Prior_Visits_12m` (350
records in each column).

## Methodology

### 1. Data preparation and feature engineering

-   Created `Claim_Submission_Month` from the claim submission date.
-   Treated `Procedure_Code` as a categorical feature.
-   Removed `Claim_ID` and `Provider_ID` from the modelling features.
-   Separated numerical and categorical predictors.
-   Used an 80:20 stratified train-test split to preserve the fraud
    proportion in both sets.

### 2. Preprocessing

The preprocessing approach depended on the model and experiment:

-   **Missing-value imputation:** Median imputation for numerical
    variables and mode imputation for categorical variables.
-   **Categorical encoding:** Ordinal encoding was used where required
    before resampling; one-hot encoding was used in the modelling
    pipeline.
-   **Scaling:** Numerical features were scaled for models that are
    sensitive to feature magnitude.
-   **Native missing-value handling:** Additional tree-based model
    configurations were evaluated without explicit imputation where
    supported.

### 3. Class imbalance

Fraudulent claims represent approximately 8.29% of the observations,
making the target imbalanced.

Two approaches were compared:

-   **SMOTENC:** Oversampling the minority class while accounting for
    both numerical and categorical features.
-   **Class weighting:** Assigning greater weight to the minority class
    during model fitting.

### 4. Models compared

Twelve model configurations were evaluated:

  Model                  Missing-value handling          Imbalance strategy
  ---------------------- ------------------------------- --------------------
  Logistic Regression    Imputation                      SMOTENC
  Logistic Regression    Imputation                      Class weighting
  K-Nearest Neighbours   Imputation                      SMOTENC
  Decision Tree          Imputation                      SMOTENC
  Decision Tree          Imputation                      Class weighting
  Decision Tree          Native missing-value handling   Class weighting
  Random Forest          Imputation                      SMOTENC
  Random Forest          Imputation                      Class weighting
  Random Forest          Native missing-value handling   Class weighting
  XGBoost                Imputation                      SMOTENC
  XGBoost                Imputation                      Class weighting
  XGBoost                Native missing-value handling   Class weighting

### 5. Model selection and evaluation

-   Used **3-fold stratified cross-validation**.
-   Applied `RandomizedSearchCV` for hyperparameter tuning of KNN,
    Decision Tree, Random Forest and XGBoost configurations.
-   Used **AUC-PR (Area Under the Precision-Recall Curve)** as the
    primary model-selection metric because the dataset is imbalanced.
-   Evaluated the selected configurations on a held-out test set.

Additional evaluation metrics included ROC-AUC, precision, recall,
F1-score and accuracy.

## Results

The following table summarises the principal test-set results recorded
in the notebook.

  -----------------------------------------------------------------------------------------
  Model           CV AUC-PR      Test      Test   Precision    Recall   F1-score   Accuracy
  configuration      (mean)    AUC-PR   ROC-AUC                                  
  --------------- --------- --------- --------- ----------- --------- ---------- ----------
  XGBoost --         0.9830    0.9866    0.9984      0.9080    0.9518     0.9294     0.9880
  Imputation +                                                                   
  SMOTENC                                                                        

  XGBoost --         0.9809    0.9802    0.9975      0.9281    0.9337     0.9309     0.9885
  Native missing                                                                 
  values + Class                                                                 
  weighting                                                                      

  XGBoost --         0.9806    0.9800    0.9975      0.9118    0.9337     0.9226     0.9870
  Imputation +                                                                   
  Class weighting                                                                

  Logistic           0.9784    0.9785    0.9976      0.8368    0.9578     0.8933     0.9810
  Regression --                                                                  
  Imputation +                                                                   
  SMOTENC                                                                        

  Logistic           0.9748    0.9764    0.9972      0.7443    0.9819     0.8468     0.9705
  Regression --                                                                  
  Imputation +                                                                   
  Class weighting                                                                

  Random Forest      0.9736    0.9792    0.9975      0.9434    0.9036     0.9231     0.9875
  -- Native                                                                      
  missing                                                                        
  values + Class                                                                 
  weighting                                                                      

  Random Forest      0.9733    0.9794    0.9974      0.9554    0.9036     0.9288     0.9885
  -- Imputation +                                                                
  Class weighting                                                                

  Random Forest      0.9684    0.9761    0.9974      0.8619    0.9398     0.8991     0.9825
  -- Imputation +                                                                
  SMOTENC                                                                        

  Decision Tree      0.9194    0.9166    0.9457      0.8613    0.8976     0.8791     0.9795
  -- Imputation +                                                                
  Class weighting                                                                

  Decision Tree      0.9171    0.9402    0.9778      0.8377    0.9639     0.8964     0.9815
  -- Imputation +                                                                
  SMOTENC                                                                        

  Decision Tree      0.9141    0.8902    0.9295      0.8571    0.8675     0.8623     0.9770
  -- Native                                                                      
  missing                                                                        
  values + Class                                                                 
  weighting                                                                      

  KNN --             0.5795    0.6187    0.9387      0.2962    0.9458     0.4511     0.8090
  Imputation +                                                                   
  SMOTENC                                                                        
  -----------------------------------------------------------------------------------------

The reported precision, recall and F1-scores use the classification
threshold applied in the notebook.

## Key Observations

-   XGBoost with imputation and SMOTENC achieved the highest test AUC-PR
    (0.9866) among the evaluated configurations.
-   XGBoost configurations performed strongly across the reported
    evaluation metrics.
-   SMOTENC and class weighting showed different effects across model
    families; neither strategy was uniformly better.
-   KNN had substantially lower AUC-PR and F1-score than the other
    evaluated models.
-   Comparing imputation and native missing-value handling helped assess
    whether explicit imputation improved tree-based model performance.

## Tools and Libraries

-   Python
-   Pandas and NumPy
-   Matplotlib and Seaborn
-   Scikit-learn
-   imbalanced-learn (SMOTENC)
-   XGBoost
-   Jupyter Notebook

## Running the Project

1.  Clone or download this repository.
2.  Install the required libraries.
3.  Download the dataset from the Kaggle source linked above.
4.  Update the dataset path in the notebook, if required.
5.  Run the notebook cells in sequence.

Example installation command:

``` bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost jupyter
```

## Limitations and Future Scope

-   **Feature availability and potential leakage:** The notebook's
    modelling feature list includes `Approved_Amount`, while its
    discussion notes that this field may not be available at the
    intended prediction time. Its use must be checked against the exact
    point at which a fraud prediction is meant to be made. If it is
    unavailable at that point, it should be excluded and the experiments
    rerun.
-   **Very high reported performance:** The strong test metrics warrant
    further validation for leakage, dataset-specific patterns and
    generalisation to independent data.
-   **Decision threshold:** The reported classification metrics use the
    notebook's applied threshold. A real deployment should select a
    threshold based on the relative costs of missed fraud and false
    alerts.
-   **Generalisation:** Evaluate the model on an independent or
    later-period dataset before drawing conclusions about real-world
    performance.
-   **Explainability:** Add feature-importance analysis and model
    explanation methods to support investigation of flagged claims.
-   **Operational evaluation:** Assess alert volume, false-positive
    burden and fraud captured at different thresholds.

## Disclaimer

This project is an exploratory machine learning analysis on a labelled
dataset. The reported results should not be interpreted as evidence of
production readiness or real-world fraud-detection performance without
further validation.
