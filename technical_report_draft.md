# Technical Report Draft
## Assignment 1 - Part 2: Software Effort Estimation

**Course:** Project Management  
**Author:** [Your name]  
**Date:** [Submission date]

---

## 1. Introduction

Software effort estimation remains one of the most critical and uncertain tasks in project management. Inaccurate estimates can lead to budget overruns, unrealistic schedules, resource conflicts, and reduced project quality. This work evaluates data-driven effort estimation models using two publicly available benchmark datasets and a rigorous validation framework to provide defensible comparisons.

The main objective is to build, compare, and critically analyze effort estimation models under reliable resampling conditions, emphasizing both predictive performance and stability across validation splits.

The study addresses the following research questions:
1. Which model family provides better predictive performance for each dataset?
2. Are model rankings consistent across modern error metrics?
3. How stable are models across repeated validation splits?
4. How do dataset characteristics influence estimation behavior?

---

## 2. Dataset Description

Two software effort estimation datasets were selected from a fallback candidate pool from PROMISE/Zenodo sources.

### 2.1 NASA93

- Source: https://zenodo.org/records/268419
- Format: ARFF
- Number of projects: 93
- Target variable: `act_effort`
- Relevant characteristics:
  - Mixed numeric and categorical features
  - Several COCOMO-style ordinal cost drivers (`vl`, `l`, `n`, `h`, `vh`, `xh`)
  - Potential target skewness typical of real project effort data

### 2.2 China

- Source: https://zenodo.org/records/268446
- Format: ARFF
- Number of projects: 499
- Target variable: `Effort`
- Relevant characteristics:
  - Larger sample size and mostly numeric feature set
  - Includes project-size and functional-point style metrics
  - Supports lower-variance estimates in repeated validation

### 2.3 Optional Fallback Candidate

- COC81 DEM (https://zenodo.org/records/268424) is kept as an additional candidate dataset.
- If one primary candidate fails parsing or validation, the notebook can skip it and continue with the next available candidate.

### 2.4 Limitations and Data Quality Notes

- NASA93 is relatively small, while China is moderate-sized for this domain.
- Dataset age and historical context may limit direct transferability to current software processes.
- Feature distributions may be skewed and differ substantially across datasets.
- Small samples increase variance in estimated performance and justify repeated resampling.
- Fallback policy: if a candidate dataset fails loading/validation, the workflow skips it and tries the next candidate.

---

## 3. Methodology

### 3.1 Preprocessing Strategy

A structured preprocessing pipeline was applied to avoid leakage and preserve reproducibility:

- Numeric features: median imputation.
- Ordinal categorical features (COCOMO rating scale): ordinal encoding with order `vl < l < n < h < vh < xh`.
- Nominal categorical features: one-hot encoding with unknown-category handling.
- ID-like or leakage-prone columns were removed from predictors (`recordnumber`, `ID`, and normalized-effort derivatives).

### 3.2 Outlier and Skewness Handling

Instead of deleting observations (risky with small datasets), the target was transformed using `log1p(y)` during training. Predictions were back-transformed with `expm1` before metric computation. This approach reduces the influence of extreme target values while preserving all projects.

### 3.3 Models

Three estimators were compared:

1. **ElasticNet** (regression-based, interpretable baseline with regularization).
2. **RandomForestRegressor** (non-linear ensemble, robust to interactions and heterogeneity).
3. **GradientBoostingRegressor** (boosted trees, often strong in tabular settings).

Hyperparameters were tuned inside inner cross-validation loops using grid search.

---

## 4. Validation Strategy

To reduce optimistic bias and improve comparison fairness, a nested design was used:

- **Outer loop:** Repeated K-Fold cross-validation (`5` folds, `10` repeats).
- **Inner loop:** K-Fold (`3` folds) for model selection and hyperparameter tuning.

Why this design:
- Repeated splitting reduces dependence on one random partition.
- Nested CV separates tuning from final evaluation.
- The strategy is suitable for small datasets where a single train/test split is unstable.

All random processes were controlled with fixed seeds for reproducibility.

---

## 5. Evaluation Metrics

Model performance was evaluated using:

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **MdAE** (Median Absolute Error)
- **MASE** (Mean Absolute Scaled Error)
- **MdASE** (Median Absolute Scaled Error)

For scaled errors (MASE, MdASE), a naive baseline predictor was defined as the **training-fold median effort**. This provides a robust, comparable scale across folds and datasets.

---

## 6. Results

> Replace this section with actual values generated by notebook outputs (`results_summary.csv`, `model_ranking.csv`, and figures).

### 6.1 Aggregated Performance by Dataset and Model

Insert the summary table from `summary_table.md`.

### 6.2 Fold-Level Stability

Use boxplots and coefficient-of-variation charts from `figures/` to discuss dispersion and ranking robustness.

### 6.3 Best Model per Dataset

Report the best model according to MAE/RMSE and verify consistency using MdAE, MASE, and MdASE.

---

## 7. Analysis and Discussion

This section should interpret model behavior in terms of both data structure and estimator bias/variance tradeoffs.

Recommended discussion points:

1. **Performance differences between datasets**
- Explain whether the same model wins on both NASA93 and China.
- Relate differences to sample size and feature composition.

2. **Metric-dependent ranking effects**
- Compare rankings under MAE vs RMSE (outlier sensitivity).
- Compare absolute vs scaled metrics (MASE/MdASE).

3. **Stability across splits**
- Use spread and coefficient of variation to justify whether a model is consistently reliable.

4. **Practical implications for project management**
- Discuss how uncertainty ranges should be communicated to stakeholders.
- Explain the risk of relying on a single estimate without variability analysis.

5. **Threats to validity**
- Small dataset size and historical context.
- Potential mismatch between legacy project environments and current software practices.
- Hyperparameter search bounds may influence observed rankings.

---

## 8. Conclusion

This study presents a reproducible and statistically rigorous workflow for software effort estimation model comparison. By combining nested repeated cross-validation with modern error metrics, the analysis supports more defensible model selection than single-split evaluations.

Final conclusions should include:
- The best-performing model(s) per dataset.
- Whether performance gains are stable and practically meaningful.
- Recommendations for estimation practice in project management contexts.

---

## 9. Appendix

### A. Reproducibility Information

- Python version: [fill from environment]
- Key package versions: [fill from `pip freeze`]
- Random seed: 42
- Notebook: `effort_estimation.ipynb`

### B. Artifact List

- `results_by_fold.csv`
- `results_summary.csv`
- `model_ranking.csv`
- `stability_metrics.csv`
- `summary_table.md`
- `figures/*.png`

### C. How to Re-run

```bash
cd assignment1_part2_effort_estimation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace effort_estimation.ipynb
```
