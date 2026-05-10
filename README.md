# Predicting the Heat: A Comparative Analysis of Machine Learning Architectures for ENSO Forecasting

Repository: https://github.com/macoconut09/ML_FinalProject

## Project Overview

This project compares several machine learning models for ENSO forecasting. The task is treated as a supervised regression problem, where recent Sea Surface Temperature (SST) values from the Nino 3.4 region are used to predict the next monthly SST value.

The main goal is to compare a simple baseline against more complex models. Ordinary Least Squares (OLS) Linear Regression is used as the baseline, while CART, Random Forest, XGBoost, and a 1D Convolutional Neural Network (CNN) are tested as non-linear and deep learning approaches.

The best test result came from the OLS model, with an $R^2$ score of 0.9461. In this dataset, the short-term SST pattern appears to be captured well by the previous 12 months, so the more complex models did not perform better here.

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

- **OLS Linear Regression**: Baseline model for checking how much predictive value comes from recent SST persistence.
- **CART Decision Tree**: Single decision-tree model for non-linear regression.
- **Random Forest Regressor**: Ensemble of decision trees trained with bagging.
- **XGBoost Regressor**: Gradient boosting model for tabular regression.
- **1D CNN**: Neural network model for learning patterns from the 12-month sequence.

## Key Results

OLS had the best test-set performance. The tree-based models performed very well on the training set but dropped on the test set, which points to overfitting. The 1D CNN gave reasonable results, but its predictions were smoother during stronger ENSO phases.

| Model | Test RMSE | Test MAE | Test $R^2$ |
|---|---:|---:|---:|
| OLS Linear Regression | 0.2367 | 0.1810 | 0.9461 |
| Random Forest Regressor | 0.2922 | 0.2333 | 0.9179 |
| XGBoost Regressor | 0.3011 | 0.2433 | 0.9128 |
| 1D CNN | 0.3130 | 0.2484 | 0.9057 |
| CART Decision Tree | 0.3613 | 0.2816 | 0.8744 |

Best model:

**OLS Linear Regression** achieved RMSE = 0.2367, MAE = 0.1810, and $R^2$ = 0.9461 on the unseen chronological test set.

## Setup & Installation

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

Run the notebooks in this order:

1. `notebooks/Data_Preparation.ipynb`
2. `notebooks/OLS_Linear_Regression.ipynb`
3. `notebooks/CART_Decision_Tree.ipynb`
4. `notebooks/Ensemble_Models.ipynb`
5. `notebooks/CNN_1D_Model.ipynb`

## Repository Structure

```text
ML_FinalProject/
|-- README.md
|-- data/
|   |-- X_train.csv
|   |-- X_test.csv
|   |-- y_train.csv
|   |-- y_test.csv
|   |-- model_comparison_metrics.csv
|   |-- *_metrics.csv
|   |-- *_test_predictions.csv
|   `-- *.png
`-- notebooks/
    |-- Data_Preparation.ipynb
    |-- OLS_Linear_Regression.ipynb
    |-- CART_Decision_Tree.ipynb
    |-- Ensemble_Models.ipynb
    `-- CNN_1D_Model.ipynb
```

## AI Use Disclosure

AI-assisted: Initial code structure and implementation were drafted with Codex.
Human edits: Adapted for the ENSO dataset, selected features and targets, verified outputs, and tuned evaluation.
