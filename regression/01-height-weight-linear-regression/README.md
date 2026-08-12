# Height–Weight Simple Linear Regression

A foundational regression project predicting **Height from Weight** using
Simple Linear Regression — built to practice the core regression workflow
(load → explore → split → scale → train → evaluate → predict) before moving
to more complex, regularized regression problems.

## Problem Statement

Given a person's weight, predict their height using a simple linear
relationship. This is a classic teaching dataset — small and clean by
design — used to build intuition for how linear regression fits a line to
data, rather than to model real-world complexity.

**Type of problem:** Supervised regression (continuous target)
**Independent variable (X):** Weight
**Dependent variable (y):** Height

## Workflow

1. **Load and inspect the dataset** — `df.head()`, checked shape and structure
2. **EDA** — scatter plot of Weight vs Height to confirm a roughly linear trend
3. **Train/test split** — held out a test set to evaluate generalization
4. **Feature standardization** — scaled Weight using `StandardScaler`
   (fit only on training data, applied to test data)
5. **Model training** — fit a `LinearRegression` model on the scaled training data
6. **Evaluation** — measured prediction quality on the held-out test set
7. **Prediction on new input** — tested the trained model on a custom weight value

## Results

| Metric | Value |
|---|---|
| R² Score | 0.777 |
| MAE | 9.82 |
| MSE | 109.78 |
| RMSE | 10.48 |

The model explains about 78% of the variance in Height using Weight alone.
Since RMSE and MAE are close in value, prediction errors are fairly
consistent across the test set, without being thrown off by extreme
outliers.

## Tech Stack

`Python` · `pandas` · `numpy` · `matplotlib` · `scikit-learn`

## Author

Arsh Tramboo — [GitHub](https://github.com/ArshTramboo)