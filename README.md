# Responsible Machine Learning Individual Assignment 4: COMPAS End-to-End Audit

## The Purpose of the Analysis

This project presents a unified, end-to-end responsible machine learning audit of the COMPAS risk assessment system across Assignments 1–4. The analysis evaluates the model from multiple perspectives including **fairness, explainability, predictive performance, generalization, distribution drift, and robustness under stress testing**.

The goal is to assess not only whether the model is fair and accurate, but also whether it is **interpretable, stable over time, resistant to distribution shift, and robust to perturbations in key features**.

Across the full pipeline, the analysis progresses from fairness diagnostics and explainability (initial assignments) to model reliability and stability (final assignment).

---

## Methods Used Across the Full Pipeline (Assignments 1–4)

The following methods were used throughout the entire analysis:

### Predictive Modeling & Performance Evaluation
- **Logistic Regression**: Used to model the probability of receiving a high COMPAS risk score while controlling for demographic and criminal history features.
- **Gradient Boosted Trees (GBT)**: Used as a nonlinear benchmark model to compare predictive performance and generalization behavior.
- **Evaluation Metrics**:
  - AUC (Area Under the Curve)
  - Accuracy
  - Log Loss
- Train vs Test comparisons were used to diagnose **overfitting and generalization gaps**.

---

### Fairness, Disparity, and Harm Analysis
- **Adverse Impact Ratio (AIR)**: Measured disparities in favorable outcomes across demographic groups.
- **Error Rate Parity**:
  - False Positive Rate (FPR)
  - False Negative Rate (FNR)
  - Group-wise comparisons to identify uneven model error burdens
- **Intersectional Analysis**: Evaluated compounded disparities across overlapping demographic attributes.
- **Standardized Mean Difference (SMD)** and **Mean Difference (ME)**: Quantified magnitude of disparities across groups.

---

### Explainability and Interpretability
- **SHAP (Shapley Additive Explanations)**:
  - Global feature importance
  - Local prediction explanations
  - Feature contribution magnitude analysis
- **LIME (Local Interpretable Model-Agnostic Explanations)**:
  - Individual-level explanations of model predictions
- **DiCE (Diverse Counterfactual Explanations)**:
  - Identified minimal feature changes required to alter predictions
  - Highlighted potential issues with immutable or sensitive features

---

### Survival and Time-to-Event Analysis
- **Cox Proportional Hazards Model**:
  - Modeled relationship between COMPAS scores and time-to-recidivism
- **Kaplan-Meier Survival Curves**:
  - Estimated probability of non-recidivism over time across risk groups
  - Compared survival distributions across demographic subgroups

---

### Distribution Drift & Data Shift Detection (New in Assignment 4)
- **Population Stability Index (PSI)**:
  - Measured shifts in feature distributions (e.g., `priors_count`)
- **Kolmogorov–Smirnov (KS) Test**:
  - Statistical test for univariate distribution differences
- **Maximum Mean Discrepancy (MMD²)**:
  - Multivariate drift detection in encoded feature space

---

### Robustness, Sensitivity, and Stress Testing (New in Assignment 4)
- **Counterfactual Attribute Swaps**:
  - Tested sensitivity of predictions to changes in race and gender
- **Stress Testing on `priors_count`**:
  - Evaluated how prediction probabilities change under systematic feature perturbation
- **Individual Conditional Expectation (ICE) Curves**:
  - Visualized individual-level prediction sensitivity to feature changes

---

### Slice-Based Evaluation
- Performance evaluated across:
  - Race groups
  - Gender groups
  - Age groups
- Used to identify subgroup-specific degradation in:
  - Accuracy
  - Error rates
  - Predictive stability

---

## Python Libraries Used

- `pandas` – Data cleaning, feature engineering, subgroup construction, and summary table generation  
- `numpy` – Numerical transformations and array operations  
- `matplotlib` – Visualization of survival curves, ICE plots, and fairness comparisons  
- `scikit-learn` – Model training (Logistic Regression, GBT) and evaluation metrics (AUC, Accuracy, Log Loss)  
- `lifelines` – Survival analysis (Kaplan-Meier, Cox proportional hazards models)  
- `shap` – Global and local feature importance analysis  
- `lime` – Local model interpretability  
- `dice-ml` – Counterfactual explanation generation and robustness testing  
- `solas-ai` – Fairness metrics including AIR, SMD, and ME  
- `statsmodels & scipy.stats` – Statistical testing (e.g., KS tests, z-tests for proportion differences)

---

## Instructions for Reproducing the Results

Install required dependencies, then run notebook file given in repo:
```python
!pip install pandas numpy matplotlib scikit-learn lifelines lime shap dice-ml solas-ai statsmodels scipy
```
Open Nick_Botti_RML_Assignment_4_Generalization.ipynb in Google Colab or a local Jupyter environment and run all cells from top to bottom.

---

## A Statement on AI Usage
Google Gemini (embedded in Google Colab) was utilized to debug runtime error messages and to assist in the logic for the Population Stability Index (PSI) calculations. Additionally, Gemini helped refine the interpretation of the ICE curves to ensure the findings aligned with the "audit-level reasoning" standards required for this assignment. This usage is consistent with the instructional guidance provided in previous live coding sessions and lectures.
