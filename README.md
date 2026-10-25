# CalibraShift: When Confidence Lies

## Research question
When a classifier assigns a high probability to its prediction, does that probability match observed reliability—and how does that relationship change under distribution shift?

## Overview
This project compares Logistic Regression and Random Forest, with and without sigmoid probability calibration, on an in-distribution test set and a controlled synthetic feature-shift test set. It uses scikit-learn's built-in Breast Cancer Wisconsin Diagnostic dataset, so no large external dataset download is required. This is a methods demonstration, not a clinical tool.

## Methods
- Stratified 75/25 train/test split with fixed random seed.
- Logistic Regression with median imputation and standardization.
- Random Forest with median imputation.
- Sigmoid calibration fitted using cross-validation on training data only.
- Synthetic test-only feature shift: add 0.35 training-standard-deviation units to each test feature.
- Metrics: accuracy, balanced accuracy, ROC-AUC, Brier score, log loss, and 10-bin Expected Calibration Error (ECE).
- Reliability diagrams and probability-quality comparison plots.

## Run in Google Colab
1. Upload `CalibraShift.ipynb` to Google Colab.
2. Run the cells in order.
3. The dataset is built into scikit-learn; no large external dataset download is needed.
4. The final cell packages the real generated CSVs and figures into `CalibraShift_results_and_figures.zip`.

## Outputs
- `results/calibration_shift_metrics.csv`
- `results/best_brier_by_condition.csv`
- `results/experiment_config.json`
- `figures/reliability_diagrams.png`
- `figures/probability_quality_under_shift.png`

## Interpretation
Do not assume calibration improves every metric. Interpret the observed results after running the notebook. Lower Brier score, log loss, and ECE generally indicate better probability quality; accuracy and ROC-AUC measure different aspects of performance.

## Limitations
- The feature shift is synthetic and simplified.
- A single split is subject to sampling variability.
- ECE depends on binning.
- This medical dataset is used only for a methods demonstration and is not suitable for clinical claims.
- Model selection and calibration must not use the test set.
