# Predicting the Heat: A Comparative Analysis of Machine Learning Architectures for ENSO Forecasting

Repository: https://github.com/macoconut09/ML_FinalProject

## Project Overview

This project studies short-term ENSO forecasting as a supervised regression problem. Using the previous 12 months of standardized Nino 3.4 sea surface temperature (SST) values, the task is to predict the next month's SST value.

The practical motivation is that ENSO affects global climate variability, disaster risk, agriculture, and economic planning. Better short-term forecasts can support earlier preparation for El Nino and La Nina impacts.

The technical goal is to compare an interpretable OLS baseline against more complex non-linear models and a 1D CNN used as the study's SOTA deep learning representative. The comparison tests whether the SOTA model and other non-linear methods can capture temporal patterns that a simple linear model might miss.

In the final results, OLS achieved the best test score with an $R^2$ of 0.9461. This suggests that, for this dataset and one-month-ahead setup, short-term SST persistence was captured very well by the previous 12 months.

## Dataset & Preprocessing

The project uses the NOAA ERSSTv5 Nino 3.4 index dataset, which contains monthly SST values for the ENSO monitoring region.

Preprocessing steps:

- Selected the Nino 3.4 SST time series as the forecasting target.
- Converted the problem into a supervised regression task.
- Engineered features using a 12-month sliding window, where months 1-12 are used to predict month 13.
- Split the data chronologically using an 80/20 train/test split.
- Used `shuffle=False` to prevent temporal data leakage.
- Applied Z-score normalization using statistics fitted only on the training set.
- Used the same processed train/test files across all model notebooks for a fair comparison.

## Model Architectures

Models used in the project:

- **OLS Linear Regression**: Interpretable baseline model for checking how much predictive value comes from recent SST persistence.
- **CART Decision Tree**: Single decision-tree model for non-linear regression.
- **Random Forest Regressor**: Ensemble of decision trees trained with bagging.
- **XGBoost Regressor**: Gradient boosting model for tabular regression.
- **1D CNN**: SOTA deep learning representative for learning patterns from the 12-month sequence.

## Key Results

OLS had the best test-set performance. The tree-based models performed very well on the training set but dropped on the test set, which points to overfitting. The 1D CNN served as the SOTA representative and gave reasonable results, but it did not outperform the linear baseline.

| Model | Test RMSE | Test MAE | Test $R^2$ |
|---|---:|---:|---:|
| OLS Linear Regression | 0.2367 | 0.1810 | 0.9461 |
| Random Forest Regressor | 0.2922 | 0.2333 | 0.9179 |
| XGBoost Regressor | 0.3011 | 0.2433 | 0.9128 |
| 1D CNN | 0.3130 | 0.2484 | 0.9057 |
| CART Decision Tree | 0.3613 | 0.2816 | 0.8744 |

Best model:

**OLS Linear Regression** achieved RMSE = 0.2367, MAE = 0.1810, and $R^2$ = 0.9461 on the unseen chronological test set.

## Setup & Reproducibility

Clone the repository:

```bash
git clone https://github.com/macoconut09/ML_FinalProject.git
cd ML_FinalProject
```

Create and activate a Python environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the required data science libraries:

```bash
pip install pandas numpy matplotlib scikit-learn xgboost tensorflow notebook
```

To reproduce the processed data, model metrics, prediction CSVs, and plots, run:

```text
notebooks/ENSO_Forecasting_All_Models.ipynb
```

This notebook contains the full workflow from data preparation to final model comparison. Tree-based models use `random_state=42`. The CNN sets NumPy and TensorFlow seeds, but neural-network results may still vary slightly across hardware or TensorFlow versions.

## Repository Structure

```text
ML_FinalProject/
|-- README.md
|-- data/
|   |-- X_train.csv
|   |-- X_test.csv
|   |-- y_train.csv
|   |-- y_test.csv
|   `-- model_comparison_metrics.csv
`-- notebooks/
    `-- ENSO_Forecasting_All_Models.ipynb
```

## AI Use Disclosure

This project used AI assistance as a programming and review aid, not as a replacement for project understanding or final authorship.

AI-assisted portions:

- Drafting notebook scaffolds for data loading, model fitting, metric calculation, plotting, and CSV export.
- Debugging environment issues involving TensorFlow and XGBoost installation.
- Standardizing notebook structure, metric names, and result-table formatting across models.
- Reviewing the manuscript and README for consistency with the implemented experiments.

Human-directed portions:

- Selection of the ENSO forecasting problem, NOAA ERSSTv5 Nino 3.4 dataset, 12-month sliding-window setup, and supervised regression framing.
- Final decisions on model families, evaluation metrics, chronological train/test splitting, and interpretation of results.
- Verification of generated outputs, comparison of model behavior, and final inclusion of results in the manuscript.
