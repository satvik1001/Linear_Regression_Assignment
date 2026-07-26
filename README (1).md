# Medical Insurance Cost Prediction using Linear Regression

Machine Learning assignment (4th Year B.Tech) predicting individual medical insurance charges from demographic and lifestyle attributes, using Linear Regression as the primary model with Ridge, Lasso, and Polynomial Regression evaluated as improvements.

## Business Problem

Health insurance companies need to estimate a customer's expected medical expenses before issuing a policy, so premiums can be priced appropriately, risk can be managed, and plans can be personalized. This project builds a regression model that predicts `charges` from `age`, `sex`, `bmi`, `children`, `smoker`, and `region`.

## Dataset

- **Source:** [Kaggle - Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **Rows / Columns:** 1,338 rows x 7 columns
- **Target:** `charges` (continuous, USD)
- **Features:** `age`, `sex`, `bmi`, `children`, `smoker`, `region`
- No missing values in any column.

## Repository Contents

| File | Description |
|---|---|
| `Assignment_Linear_Regression.ipynb` | Full Jupyter Notebook: EDA, preprocessing, model training, evaluation, and tuning, with all outputs and plots executed inline. |
| `Insurance_Charges_Prediction_Report.pdf` | 10-page written report covering all required sections (Introduction through Conclusion). |
| `README.md` | This file. |
| `insurance.csv` | Dataset used (Medical Cost Personal Dataset). |
| `images/` | All plots generated during EDA and modeling (11 PNG files), embedded in the report. |

## Pipeline Summary

1. **Data Understanding** — loaded data, checked shape/types/missing values, identified numerical vs. categorical features, computed descriptive statistics.
2. **EDA** — target distribution, correlation heatmap, histograms, boxplots by category, scatter plots, pairplot, and IQR-based outlier analysis.
3. **Preprocessing** — label encoding for `sex`/`smoker`, one-hot encoding for `region` (drop-first), `StandardScaler` feature scaling, 80:20 train-test split.
4. **Model Training** — baseline `LinearRegression` (scikit-learn), with coefficients and feature importance analyzed.
5. **Evaluation** — MAE, MSE, RMSE, R², Adjusted R², Predicted-vs-Actual plot, residual plot.
6. **Improvement** — Ridge and Lasso (tuned via `GridSearchCV`), and degree-2 Polynomial Regression compared against the baseline.
7. **Business Insights** — translated model findings into pricing and risk-management recommendations.

## Key Results

| Model | MAE | RMSE | R² | Adjusted R² |
|---|---|---|---|---|
| Linear Regression (baseline) | 4,181.19 | 5,796.28 | 0.7836 | 0.7769 |
| Ridge (alpha=10) | 4,197.66 | 5,803.95 | 0.7830 | 0.7763 |
| Lasso (alpha=100) | 4,210.61 | 5,835.80 | 0.7806 | 0.7739 |
| **Polynomial Regression (deg=2)** | **2,729.50** | **4,551.13** | **0.8666** | **0.8403** |

**Headline finding:** `smoker` is by far the strongest predictor of insurance charges (correlation ≈ 0.79 with charges; largest model coefficient, +9,558), followed by `age` and `bmi`. Smokers pay ~3.8x more on average than non-smokers (~$32,050 vs. ~$8,434), and this effect compounds with BMI — the highest-cost segment is smokers with BMI ≥ 30. Because of this non-linear smoker×age/BMI interaction, degree-2 polynomial regression outperforms the plain linear model, raising R² from ~0.78 to ~0.87.

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook Assignment_Linear_Regression.ipynb
```

Run all cells top to bottom; `insurance.csv` must be in the same directory as the notebook.

## Author

Satvik Singh — B.Tech CSE (AI & ML), GL Bajaj Institute of Technology and Management (Batch 2023-2027)
