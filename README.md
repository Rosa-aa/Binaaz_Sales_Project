# 🏠 Bina.az Real Estate Market Analysis & Price Prediction Model

## 1. Project Overview

This project analyzes the Baku real estate market using real listing data scraped from **bina.az**, and builds a machine learning model to predict property prices. The project covers the full data science lifecycle: raw data cleaning, exploratory data analysis (EDA), feature engineering, comparison of multiple regression models, and hyperparameter tuning of the best-performing model.

The final result is a tuned **XGBoost** model that predicts property prices with **R² = 0.823** on the held-out test set.

---

## 2. Dataset

| Attribute | Value |
|---|---|
| Source | bina.az (collected via web scraping) |
| File | `house_sale.csv` |
| Raw size | 5,902 rows × 51 columns |
| After cleaning (used for modeling) | 4,721 rows × 28 columns |
| Geographic coverage | Baku and surrounding areas |

**Note:** The dataset file is not included in this repository due to its size. To run the notebook, you'll need to obtain your own `house_sale.csv` file.

---

## 3. Methodology

The project follows this sequential pipeline:

```
Raw Data (5,902 × 51)
     │
     ▼
Removal of Duplicate/Redundant Columns
     │
     ▼
Parsing Structured Text Fields (Area, Room Count, Floor)
     │
     ▼
Renaming Columns to Azerbaijani
     │
     ▼
Missing Value Analysis
     │
     ▼
Exploratory Data Analysis (EDA) — 10+ visualizations
     │
     ▼
Outlier Removal (IQR Method)
     │
     ▼
Feature Engineering (4 new variables)
     │
     ▼
Encoding (Label / Target / One-Hot)
     │
     ▼
Train/Test Split (80/20) + Preprocessing Pipeline
     │
     ▼
Comparison of 5 Models
     │
     ▼
Hyperparameter Tuning of the Best Model (XGBoost)
     │
     ▼
Cross-Validation and Feature Importance Analysis
```

---

## 4. Data Cleaning

### 4.1 Detecting Duplicate and Redundant Columns

Column pairs with more than 95% similarity were automatically detected (e.g., `id_x` and `estate_id` matched 99.98% of the time — the same information stored under two different column names). This analysis led to the removal of 11 duplicate/redundant columns (`estate_id`, `currency_y`, `total_price`, `id_y`, etc.) and 5 low-quality columns (`Binanın növü` — 99.8% missing, `featured`, `vip`, `Torpaq sahəsi`, `hour_y`).

**Result:** Reduced from 51 columns to 35 columns.

### 4.2 Parsing Structured Text Fields

Several columns stored multiple pieces of information as combined text, which were split into separate numeric fields using regex:

| Original column | Example value | Extracted column(s) |
|---|---|---|
| `Sahə` (Area) | `"145 m²"` | `area` (145.0) |
| `Otaq sayı` (Room count) | `"4"` | `rooms` (4.0) |
| `Mərtəbə` (Floor) | `"7/9"` | `floor` (7), `total_floors` (9) |
| `unit_price` | `"2500 AZN/m²"` | `kvadrat_metr_qiyməti` (2500.0) |

### 4.3 Renaming Columns to Azerbaijani

All column names were standardized from a mixed English/Azerbaijani format into consistent Azerbaijani (`price` → `qiymət`, `location` → `yer`, `area` → `sahə`, etc.) — this improves readability for anyone reviewing the code and aligns with the terminology used in the dashboard.

### 4.4 Missing Value Analysis

Columns with the highest proportion of missing values:

| Column | Missing % |
|---|---|
| `ipoteka` (mortgage) | 67.1% |
| `agentlik_adı` / `agentlik_tipi` (agency name/type) | 32.16% |
| `elan_etiketi` (listing tag) | 31.31% |
| `ümumi_mərtəbə` / `kvadrat_metr_qiyməti` / `mərtəbə` (total floors / price per m² / floor) | 23.92% |
| `alqı_sənədi` (purchase document) | 21.5% |
| `təmir_statusu` (renovation status) | 19.45% |

These gaps were filled during the modeling stage using `SimpleImputer` (median for numeric columns, most-frequent value for categorical columns).

---

## 5. Exploratory Data Analysis (EDA)

Over 10 visualizations were produced (`matplotlib`, `seaborn`, `plotly`), including:

- **Price distribution** — both raw and log-scaled (a right-skewed distribution was detected, prompting a log-transformation experiment during modeling)
- **Boxplot analysis** — visual confirmation of outliers in price, area, view count, and price-per-m²
- **Correlation heatmap** — relationships between numeric variables
- **Category distribution** — donut chart
- **Median price by room count** — bar chart
- **Area vs. Price** — scatter plot with trend line (on a 600-listing sample)
- **Median price by district** — horizontal bar chart, with the most expensive district highlighted
- **Median price by floor** — line chart, with the peak point marked
- **Geographic map** — built with `folium`, showing each property's distance to the nearest metro station

---

## 6. Outlier Removal

Statistical thresholds were calculated using the IQR (Interquartile Range) method:

| Column | Outliers detected |
|---|---|
| `qiymət` (price) | 369 |
| `sahə` (area) | 287 |
| `baxış_sayı` (view count) | 545 |
| `kvadrat_metr_qiyməti` (price per m²) | 249 |

The price column was filtered to stay within the IQR bounds, and view count was filtered to below its 99th percentile.

**After cleaning:** 4,721 rows remained, with price stabilizing between 3,500 AZN and 600,000 AZN (median: 200,000 AZN).

---

## 7. Feature Engineering

Four new variables were derived from existing columns to improve the model's predictive power:

| New feature | Formula / Logic |
|---|---|
| `qiymət_per_m2` (price per m²) | `price / area` |
| `mərtəbə_nisbəti` (floor ratio) | `floor / total_floors` (the property's relative position in the building) |
| `zemin_mərtəbə` (ground floor) | 1 if on the first floor, else 0 |
| `son_mərtəbə` (top floor) | 1 if on the top floor, else 0 |

**Important note — preventing data leakage:** `qiymət_per_m2` and `kvadrat_metr_qiyməti` (price per m²) are directly derived from the target variable `qiymət` (price), so these columns — along with `qiymət_qrupu` (price bucket) — were **excluded** from the final feature set used for modeling. Leaving them in would let the model effectively "predict price from price," producing meaningless results in a real-world scenario.

---

## 8. Encoding Strategy

Categorical variables were converted to numeric form using three different techniques, each chosen for a specific reason:

| Method | Applied to | Reason |
|---|---|---|
| **Label Encoding** | `təmir_statusu`, `elan_etiketi`, `alqı_sənədi`, `mülkiyyətçi_tipi`, `agentlik_tipi`, `kateqoriya`, `çıxarış`, `ipoteka` | Low cardinality (few unique values) |
| **Target Encoding** | `yer` (location), `agentlik_adı` (agency name) | High cardinality — one-hot encoding would have created hundreds of columns |
| **One-Hot Encoding** | `şəhər` (city) | A small number of categories with no inherent order |

In addition, a full preprocessing pipeline was built using `ColumnTransformer`:
- **Numeric columns:** median imputation → `StandardScaler` scaling
- **Categorical columns:** most-frequent-value imputation → `OneHotEncoder`

---

## 9. Model Building and Comparison

An 80/20 train/test split was used (`random_state=42` for reproducibility). Five different regression models were tested under identical conditions:

| Model | MAE (AZN) | RMSE (AZN) | R² |
|---|---|---|---|
| **XGBoost** | **31,666** | **49,871** | **0.818** |
| Random Forest | 33,272 | 52,115 | 0.801 |
| Gradient Boosting | 35,652 | 55,121 | 0.778 |
| Decision Tree | 41,267 | 65,839 | 0.683 |
| Linear Regression | 50,056 | 206,924 | **−2.135** |

**Note:** Linear Regression's negative R² clearly demonstrates that property price has a **non-linear** relationship with area, location, and the other predictors — explaining why tree-based models (Random Forest, XGBoost) are much better suited to this problem.

---

## 10. Best Model — Tuned XGBoost

The winning model from the initial comparison, XGBoost, was further tuned:

```python
XGBRegressor(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

**Final test results:**

| Metric | Value |
|---|---|
| MAE | 30,568.83 AZN |
| RMSE | 49,147.53 AZN |
| **R²** | **0.8231** |

**5-Fold Cross-Validation:** Mean R² = 0.8214, Standard Deviation = 0.0127 — the low variance across folds indicates the model performs reliably regardless of how the data happens to be split.

### Overfitting Check

| Set | R² |
|---|---|
| Train | 0.977 |
| Test | 0.818 |

The gap between train and test R² (0.977 → 0.818) indicates **moderate overfitting** — the model has a tendency to memorize training data to some degree. This is flagged as an open area for improvement (see Section 13).

---

## 11. Feature Importance

According to XGBoost's `feature_importances_`, the top 10 variables driving price predictions are:

| Rank | Feature | Importance |
|---|---|---|
| 1 | `sahə` (area, m²) | 0.211 |
| 2 | `kateqoriya` (category) | 0.136 |
| 3 | `yer` (location/district) | 0.134 |
| 4 | `otaq_sayı` (room count) | 0.113 |
| 5 | `çıxarış` (property extract/deed) | 0.065 |
| 6 | `enlik` (latitude) | 0.043 |
| 7 | `alqı_sənədi` (purchase document) | 0.040 |
| 8 | `uzunluq` (longitude) | 0.037 |
| 9 | `təmir_statusu` (renovation status) | 0.028 |
| 10 | `son_mərtəbə` (top floor) | 0.025 |

**Interpretation:** Area (`sahə`) is by far the strongest predictor, followed by category and location — an intuitive result for a real estate market and confirmation that the model is learning patterns consistent with real-world logic.

---

## 12. Key Insights

- **Price does not scale linearly with area** — the scatter plot and model results show a complex relationship, which is why plain Linear Regression failed outright (negative R²)
- **Some districts command significantly higher median prices than others** — location is among the top-3 most important features
- **Floor level has a non-linear effect on price** — neither the ground floor nor the top floor is optimal; mid-range floors tend to be valued higher

---

## 13. Dashboard

The project includes an interactive EDA dashboard:

```bash
open house_sale_eda_dashboard.html
```

Opens directly in any browser — no server required.

---

## 14. Repository Structure

```
Binaaz_Sales_Project/
├── README.md
├── .gitignore
├── Binaaz_Sale_Project.ipynb      ← Full analysis and modeling notebook
└── house_sale_eda_dashboard.html  ← Interactive EDA dashboard
```

**Note:** `house_sale.csv` (the raw dataset) is not included in the repository due to file size.

---

## 15. Setup & Installation

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn plotly folium
jupyter notebook Binaaz_Sale_Project.ipynb
```

The `house_sale.csv` file must be placed in the same directory as the notebook before running it.

---

## 16. Tech Stack

**Data processing:** Python, pandas, NumPy
**Visualization:** Matplotlib, Seaborn, Plotly, Folium
**Machine Learning:** scikit-learn (Linear Regression, Decision Tree, Random Forest, Gradient Boosting), XGBoost
**Environment:** Google Colab

---

## 17. Limitations & Future Improvements

- **Overfitting:** The train/test R² gap (0.977 vs. 0.818) suggests the model could be improved further by reducing `max_depth` or applying stronger L1/L2 regularization
- **Dataset size:** At 4,721 rows, the modeling dataset is relatively small — collecting data over a longer time window could improve the model's ability to generalize
- **Geographic features:** Currently only latitude/longitude are used as raw coordinates — distance to the nearest metro station (already computed for the folium map) has not yet been integrated as a model feature, which is a promising direction for improvement
- **Log transformation:** A log-transformation of the target variable was experimented with but not used in the final model — a systematic comparison (log vs. raw target) would be a worthwhile follow-up
