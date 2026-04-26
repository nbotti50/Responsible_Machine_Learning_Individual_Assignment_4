# Responsible Machine Learning Individual Assignment 4: Generalization, Drift, and Robustness

## Project Objective
The objective of this assignment is to reproduce and extend the lecture pipeline for the COMPAS risk assessment dataset using Python. This analysis moves beyond initial fairness audits to evaluate the **long-term reliability and stability** of the predictive models. The goal is to determine if the model maintains its integrity when faced with new data and to identify potential vulnerabilities through stress testing.

---

## Technical Methodologies & Evaluation
This audit focuses on four critical dimensions of model health:

1. **Distribution Drift (Part A)**: 
   - Computed **Population Stability Index (PSI)** and **Kolmogorov-Smirnov (KS)** tests for numeric features (e.g., `priors_count`) to identify shifts between training and testing populations.
   - Implemented **Maximum Mean Discrepancy (MMD²)** in the encoded feature space to detect multivariate drift.
2. **Generalization (Part B)**: 
   - Evaluated **AUC, Accuracy, and Log Loss** across both training and testing sets for Logistic Regression and Gradient-Boosted Trees.
   - Diagnosed overfitting by calculating the performance gaps (e.g., `AUC_gap`) to ensure the model's predictive power is stable.
3. **Spurious-Correlation Probe (Part C)**: 
   - Utilized **Counterfactual Swaps** on protected attributes (race, gender) to measure changes in predicted probabilities, ensuring the model does not rely on sensitive or immutable proxies.
4. **Robustness & Sensitivity (Part D)**: 
   - Performed stress tests on `priors_count` and generated **Individual Conditional Expectation (ICE) curves** to visualize how individual predictions respond to specific feature perturbations.
5. **Slice-Based Evaluation (Part E)**: 
   - Conducted a performance audit across demographic slices (race, gender, age) to monitor for subgroup performance disparities.

---

## Python Libraries Used
- `pandas` – Data manipulation and creation of the final Summary Table.
- `numpy` – Numerical transformations and array handling.
- `matplotlib` – Visualizing score distributions and ICE curves.
- `scikit-learn` – Model training (Logistic Regression/GBT) and performance metrics (AUC, Log Loss).
- `scipy.stats` – Statistical testing for distribution drift (KS test).
- `dice-ml` – Generating diverse counterfactual explanations for the robustness probe.

---

## Instructions for Reproducing the Results

1. **Environment Note**: This notebook was developed and tested on **macOS**. 
2. **Installation**: Ensure the necessary dependencies are installed by running:
   ```python
   !pip install pandas numpy matplotlib scikit-learn dice-ml scipy
3. Execution: Open Nick_Botti_RML_Assignment_4_Generalization (1).ipynb in Google Colab or a local Jupyter environment and run all cells from top to bottom

---

## A Statement on AI usage
Google Gemini (embedded in Google Colab) was utilized to debug runtime error messages and to assist in the logic for the Population Stability Index (PSI) calculations. Additionally, Gemini helped refine the interpretation of the ICE curves to ensure the findings aligned with the "audit-level reasoning" standards required for this assignment. This usage is consistent with the instructional guidance provided in previous live coding sessions and lectures.
