# California Housing — Multiple Linear Regression

Predicting median house value across California districts using Multiple
Linear Regression, on scikit-learn's California Housing dataset.

## Problem

Predict median house value from 8 features describing a California block
group (income, housing age, room counts, population, location).

**Target (y):** Median House Value
**Features (X):** MedInc, HouseAge, AveRooms, AveBedrms, Population,
AveOccup, Latitude, Longitude

## Dataset

20,640 instances, 8 features, no missing values. Loaded via
`sklearn.datasets.fetch_california_housing()`.

## Workflow

1. Loaded dataset, converted to DataFrame
2. EDA — distributions, correlation heatmap
3. Train-test split (test size 0.33, random_state=10)
4. Standardized features
5. Trained `LinearRegression`
6. Evaluated with MAE, MSE, RMSE, R²
7. Checked residuals (scatter + KDE distribution)

## Results

| Metric | Value |
|---|---|
| R² Score | 0.59 |
| MAE | 0.552 |
| MSE | 0.537 |
| RMSE | 0.743 |

## Notes

R² of 0.59 means the model explains about 59% of the variance in house
value — reasonable but not strong for a real-world dataset like this.

Two features, Latitude and Longitude, relate to price through geography
(coastal areas, city clusters) rather than a straight-line relationship, so
plain linear regression structurally can't capture that part of the signal.
The residual plot backs this up: a funnel-shaped spread instead of random
scatter, and a right-skewed residual distribution instead of a symmetric
one — both point to the same non-linearity, not a coding issue.

Regularized regression (Ridge/Lasso/ElasticNet) won't fix the non-linearity
itself, but a tree-based model (Random Forest, Gradient Boosting) would
likely do meaningfully better on this dataset.

## Tech Stack

`Python` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn`

## Author

Arsh Tramboo — [GitHub](https://github.com/ArshTramboo)
