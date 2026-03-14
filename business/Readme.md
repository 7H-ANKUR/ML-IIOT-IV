# 📊 BizForecast — Business Sales Analysis & Forecasting Module

> **Author:** Ankur Pratap Singh  
> **Project Type:** B.Tech Major Project — Machine Learning  
> **Dataset:** BigMart Sales Dataset (8,523 records)  
> **Status:** Evaluation Ready ✅

---

## 📌 Table of Contents

1. [What This Project Is](#what-this-project-is)
2. [What Problem It Solves](#what-problem-it-solves)
3. [Technologies & Libraries Used](#technologies--libraries-used)
4. [Project Structure](#project-structure)
5. [Dataset Description](#dataset-description)
6. [What Each Module Does](#what-each-module-does)
7. [Machine Learning Models](#machine-learning-models)
8. [Flow Diagram](#flow-diagram)
9. [Feature Selection](#feature-selection)
10. [Model Performance](#model-performance)
11. [Threat Analysis](#threat-analysis)
12. [How to Run Locally](#how-to-run-locally)
13. [API Endpoints](#api-endpoints)
14. [Work Completed So Far](#work-completed-so-far)

---

## What This Project Is

BizForecast is a **Machine Learning-powered Business Sales Forecasting System** that analyses retail sales data, identifies key drivers of sales performance, predicts future sales figures, and classifies products into sales categories (Low / Medium / High).

The system encompasses five integrated modules:

| Module | Purpose |
|---|---|
| EDA | Exploratory Data Analysis — understand data distribution, patterns, and anomalies |
| Feature Selection | Identify which variables most significantly influence sales |
| Forecasting Engine | Train ML models to predict sales values |
| Threat Analysis | Detect model risks — outliers, bias, instability |
| Dashboard | Visual frontend interface for live prediction and result display |

---

## What Problem It Solves

Retail businesses generate enormous volumes of transactional data but lack systematic mechanisms to:

- Predict how much a product will sell at a given outlet
- Identify which factors (price, location, outlet type) drive or suppress sales
- Classify products into actionable sales tiers for inventory and pricing decisions

This project addresses all three objectives using supervised machine learning on the BigMart Sales dataset — a real-world retail dataset comprising 8,523 product-outlet combinations across multiple store types and locations.

---

## Technologies & Libraries Used

### Backend & Machine Learning

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.x | Primary programming language |
| scikit-learn | Latest | ML models, preprocessing, evaluation |
| pandas | Latest | Data loading, manipulation, cleaning |
| numpy | Latest | Numerical computations |
| matplotlib | Latest | Visualisation and charts |
| seaborn | Latest | Statistical plots |
| Flask | Latest | Web server / API backend |
| pickle | Built-in | Model serialisation (.pkl files) |

### Frontend

| Technology | Purpose |
|---|---|
| HTML5 | Page structure, forms, tabs |
| CSS3 | Styling — grid layout, animations, variables, responsive design |
| Vanilla JavaScript | Tab switching, fetch API calls to Flask backend |

### No external JS frameworks (React, Vue, jQuery) were used. The frontend is pure HTML + CSS + JavaScript.

---

## Project Structure

```
BizForecast/
│
├── Train.csv                    # Raw dataset (BigMart Sales)
│
├── sales_forecast_pipeline.py   # Full ML pipeline script
│   ├── Section 1: Preprocessing
│   ├── Section 2: Feature Selection
│   ├── Section 3: Forecasting Engine
│   ├── Section 4: Threat Analysis
│   └── Section 5: Dashboard charts
│
├── app.py                       # Flask backend server
│   ├── GET  /                   # Serves frontend HTML
│   ├── POST /predict            # Returns model predictions
│   └── GET  /model-info         # Returns model metrics as JSON
│
├── sales_dashboard_ui.html      # Frontend UI (5 tabs)
│
├── all_models.pkl               # Trained model bundle
│   ├── svr                     # SVR model
│   ├── linear_regression        # Linear Regression model
│   ├── logistic_regression      # Logistic Regression model
│   ├── scaler                   # StandardScaler (mandatory for inference)
│   ├── features                 # List of 6 selected feature names
│   └── metrics                  # Stored evaluation metrics
│
├── feature_selection.png        # Feature importance chart
├── sales_dashboard.png          # Model performance dashboard chart
└── README.md                    # This file
```

---

## Dataset Description

**Source:** BigMart Sales Dataset  
**Records:** 8,523 rows × 12 columns  
**Target Variable:** `Item_Outlet_Sales` (continuous — retail sales in ₹)

| Column | Type | Description |
|---|---|---|
| Item_Identifier | Categorical | Unique product ID |
| Item_Weight | Numerical | Weight of the product (kg) |
| Item_Fat_Content | Categorical | Low Fat / Regular |
| Item_Visibility | Numerical | % of display area allocated to the product |
| Item_Type | Categorical | Product category (Dairy, Meat, Snacks, etc.) |
| Item_MRP | Numerical | Maximum Retail Price (₹) |
| Outlet_Identifier | Categorical | Unique store ID |
| Outlet_Establishment_Year | Numerical | Year the outlet was established |
| Outlet_Size | Categorical | Small / Medium / High |
| Outlet_Location_Type | Categorical | Tier 1 / Tier 2 / Tier 3 city |
| Outlet_Type | Categorical | Grocery Store / Supermarket Type 1/2/3 |
| **Item_Outlet_Sales** | **Numerical** | **Target — sales value (₹)** |

### Missing Values Handled

| Column | Missing Count | Treatment |
|---|---|---|
| Item_Weight | 1,463 | Imputed with median |
| Outlet_Size | 2,410 | Imputed with mode |

### Engineered Feature

| Feature | Formula | Rationale |
|---|---|---|
| Outlet_Age | 2024 − Outlet_Establishment_Year | Older outlets may have larger customer bases |

---

## What Each Module Does

### Module 1 — EDA (Exploratory Data Analysis)
Investigates the raw dataset to understand:
- Distribution of sales (right-skewed, range ₹33 – ₹13,087)
- Correlation between numerical features and target
- Missing value patterns
- Categorical variable distributions (outlet types, fat content inconsistencies)
- Key observation: `Item_MRP` has the strongest positive correlation (0.57) with sales

### Module 2 — Feature Selection
Three independent methods applied to identify the most predictive variables:

- **Method A — Pearson Correlation:** Measures linear relationship strength between each feature and target
- **Method B — F-Regression (SelectKBest):** Statistical F-test assessing feature significance
- **Method C — Mutual Information:** Captures both linear and non-linear dependencies

Features selected by consensus across all three methods:
`Item_MRP`, `Outlet_Type`, `Outlet_Age`, `Item_Visibility`, `Item_Weight`, `Item_Fat_Content`

### Module 3 — Forecasting Engine
Three ML models trained on 80% of data, evaluated on 20%:
- **Linear Regression** — baseline regression
- **Logistic Regression** — sales category classification (Low / Medium / High)
- **SVR with RBF Kernel** — non-linear regression, best performer

All inputs scaled via `StandardScaler` before model training.  
5-fold cross-validation applied to assess generalisation stability.

### Module 4 — Threat Analysis
Automated detection of four model risks:
1. High-error outliers (predictions with |residual| > 2σ)
2. Underfitting (R² < 0.60 threshold)
3. Cross-validation instability (CV std > 0.05)
4. Systematic bias (mean residual significantly non-zero)

### Module 5 — Dashboard
Interactive frontend with 5 tabs:
1. **Overview** — Model comparison + Sales category donut chart
2. **Feature Selection** — Correlation and Mutual Information bar charts
3. **Live Predictor** — Real-time prediction form connected to Flask backend
4. **Algorithms** — Explanations of all 3 algorithms with metrics
5. **Threat Analysis** — Visual display of all risk flags and summary

---

## Machine Learning Models

### 1. Linear Regression
**Type:** Supervised — Regression  
**Objective:** Predict the continuous `Item_Outlet_Sales` value

**How it works:**  
Fits a straight line (hyperplane in multiple dimensions) through the data by minimising the sum of squared differences between predicted and actual values (Ordinary Least Squares).

**Equation:**
```
ŷ = β₀ + β₁(Item_MRP) + β₂(Outlet_Type) + β₃(Outlet_Age) + ...
```

**When it works well:** When the relationship between features and target is approximately linear.  
**Limitation in this project:** Sales does not scale linearly with MRP across all outlet types — hence the modest R² of 0.503.

---

### 2. Logistic Regression
**Type:** Supervised — Classification  
**Objective:** Classify sales into Low / Medium / High categories

**How it works:**  
Applies the sigmoid (logistic) function to model the probability of a sample belonging to each class. The class with the highest probability is assigned as the prediction.

**Target variable transformation:**
```
Sales < ₹1,000        →  Class 0 (Low)    — 2,504 samples (29.4%)
₹1,000 ≤ Sales < ₹3,000 →  Class 1 (Medium) — 3,753 samples (44.0%)
Sales ≥ ₹3,000        →  Class 2 (High)   — 2,266 samples (26.6%)
```

**Why transformation was necessary:** Logistic Regression is fundamentally a classification algorithm. Since `Item_Outlet_Sales` is continuous, it was discretised into three ordinal categories to make Logistic Regression applicable.

**Result:** 67.1% accuracy across 3 classes (1705 test samples).

---

### 3. SVR — Support Vector Regression (RBF Kernel)
**Type:** Supervised — Regression  
**Objective:** Predict the continuous `Item_Outlet_Sales` value

**How it works:**  
SVR finds an optimal regression function within an ε-insensitive tube (ε = 0.1). Predictions within ±ε of the actual value incur zero penalty. Only data points outside this tube (called **support vectors**) influence the model.

The **RBF (Radial Basis Function) kernel** projects data into a higher-dimensional space, enabling SVR to capture non-linear relationships that linear models cannot.

**Parameters used:**
```
kernel  = 'rbf'   → Non-linear kernel for complex patterns
C       = 100     → High regularisation — penalises large errors strictly
epsilon = 0.1     → Tolerance margin — predictions within ±0.1 are acceptable
```

**Why SVR outperformed Linear Regression:**  
Sales data contains non-linear interactions (e.g., MRP × Outlet_Type combinations). The RBF kernel captured these relationships, improving R² from 0.503 (Linear) to 0.603 (SVR).

---

## Flow Diagram

```
╔══════════════════════════════════════════════════════════════════╗
║                     RAW DATASET (Train.csv)                      ║
║                   8,523 rows × 12 columns                        ║
╚══════════════════════════════════════════════════════════════════╝
                              │
                              ▼
╔══════════════════════════════════════════════════════════════════╗
║                    PREPROCESSING                                 ║
║  • Impute Item_Weight (median)                                   ║
║  • Impute Outlet_Size (mode)                                     ║
║  • Standardise Fat Content labels                                ║
║  • Engineer Outlet_Age = 2024 − Establishment_Year              ║
║  • Label Encode all categorical columns                          ║
║  • Drop non-informative IDs                                      ║
╚══════════════════════════════════════════════════════════════════╝
                              │
                              ▼
╔══════════════════════════════════════════════════════════════════╗
║                    FEATURE SELECTION                             ║
║  Method A: Pearson Correlation                                   ║
║  Method B: F-Regression (SelectKBest)                           ║
║  Method C: Mutual Information                                    ║
║                                                                  ║
║  Selected: Item_MRP, Outlet_Type, Outlet_Age,                   ║
║            Item_Visibility, Item_Weight, Item_Fat_Content        ║
╚══════════════════════════════════════════════════════════════════╝
                              │
                              ▼
╔══════════════════════════════════════════════════════════════════╗
║               TRAIN / TEST SPLIT  (80% / 20%)                   ║
║            Train: 6,818 samples │ Test: 1,705 samples           ║
╚══════════════════════════════════════════════════════════════════╝
                              │
                              ▼
╔══════════════════════════════════════════════════════════════════╗
║                   STANDARD SCALING                               ║
║         Fit on Train → Transform Train & Test                    ║
╚══════════════════════════════════════════════════════════════════╝
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
     ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
     │   LINEAR    │ │  LOGISTIC   │ │     SVR     │
     │ REGRESSION  │ │ REGRESSION  │ │ (RBF Kernel)│
     │             │ │             │ │             │
     │ R² = 0.503  │ │ Acc = 67.1% │ │ R² = 0.603  │
     └─────────────┘ └─────────────┘ └─────────────┘
              │               │               │
              └───────────────┼───────────────┘
                              ▼
╔══════════════════════════════════════════════════════════════════╗
║                    THREAT ANALYSIS                               ║
║  • Outlier detection (|residual| > 2σ)                          ║
║  • Underfitting check (R² < 0.60)                               ║
║  • CV stability check (std > 0.05)                              ║
║  • Bias detection (mean residual)                               ║
╚══════════════════════════════════════════════════════════════════╝
                              │
                              ▼
╔══════════════════════════════════════════════════════════════════╗
║                  MODEL SAVED (.pkl)                              ║
║         all_models.pkl — SVR + LR + Logistic + Scaler           ║
╚══════════════════════════════════════════════════════════════════╝
                              │
                              ▼
╔══════════════════════════════════════════════════════════════════╗
║                   FLASK BACKEND (app.py)                        ║
║         Loads .pkl → Accepts POST /predict requests             ║
╚══════════════════════════════════════════════════════════════════╝
                              │
                              ▼
╔══════════════════════════════════════════════════════════════════╗
║              FRONTEND DASHBOARD (HTML/CSS/JS)                   ║
║      5 Tabs: Overview │ Features │ Predictor │ Algo │ Threats   ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## Feature Selection

Three methods were applied and results cross-validated:

| Feature | Correlation | F-Score | Mutual Info | Selected |
|---|---|---|---|---|
| Item_MRP | 0.568 | 4049.5 | 0.763 | ✅ Yes |
| Outlet_Type | 0.402 | 1637.8 | 0.271 | ✅ Yes |
| Outlet_Size | 0.173 | 261.3 | 0.040 | ❌ No |
| Item_Visibility | 0.129 | 143.4 | 0.084 | ✅ Yes |
| Outlet_Location_Type | 0.089 | 68.6 | 0.038 | ❌ No |
| Outlet_Age | 0.049 | 20.6 | 0.160 | ✅ Yes |
| Item_Fat_Content | 0.019 | 3.0 | 0.022 | ✅ Yes |
| Item_Weight | 0.014 | 0.8 | 0.075 | ✅ Yes |

`Outlet_Size` and `Outlet_Location_Type` were excluded — inconsistent rankings across methods and low mutual information scores indicate limited predictive contribution.

---

## Model Performance

| Model | Type | MAE | RMSE | R² / Accuracy | CV R² Mean | CV Std |
|---|---|---|---|---|---|---|
| Linear Regression | Regression | 876.34 | 1162.04 | 0.503 | 0.482 | 0.011 |
| Ridge Regression | Regression | 876.24 | 1162.04 | 0.503 | 0.482 | 0.011 |
| Lasso Regression | Regression | 876.18 | 1161.92 | 0.503 | 0.482 | 0.011 |
| **SVR (RBF Kernel)** | **Regression** | **724.60** | **1039.15** | **0.603** | **0.568** | **0.009** |
| Logistic Regression | Classification | — | — | **67.1%** | — | — |

**Best Model: SVR (RBF Kernel)** — lowest MAE, lowest RMSE, highest R²

### Logistic Regression — Class-wise Performance

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Low | 0.74 | 0.72 | 0.73 | 521 |
| Medium | 0.63 | 0.70 | 0.67 | 761 |
| High | 0.66 | 0.56 | 0.60 | 423 |
| **Overall** | — | — | **0.67** | **1705** |

---

## Threat Analysis

| Flag | Status | Detail |
|---|---|---|
| High-Error Outliers | ⚠ Warning | 94 samples (5.5%) have \|residual\| > 2σ |
| Model Fit | ✅ Pass | SVR R² = 0.603 — above 0.60 threshold |
| CV Stability | ✅ Pass | CV std = 0.009 — well below 0.05 threshold |
| Systematic Bias | ⚠ Warning | Mean residual = +101.72 — slight underestimation |

**Interpretation of Bias:** The model marginally underestimates actual sales. A contributing factor is the absence of a promotional pricing or seasonality feature in the dataset.

---

## How to Run Locally

### Prerequisites

```bash
pip install flask scikit-learn pandas numpy matplotlib seaborn
```

### Step 1 — Place all files in one folder

```
your_folder/
├── app.py
├── all_models.pkl
└── sales_dashboard_ui.html
```

### Step 2 — Start the Flask server

```bash
python app.py
```

You will see:
```
==================================================
  BizForecast Server Starting...
  Open: http://localhost:5000
==================================================
✓ Models loaded successfully
```

### Step 3 — Open the dashboard

Open your browser and navigate to:
```
http://localhost:5000
```

The frontend will load. Navigate to the **Live Predictor** tab and enter product/outlet details to receive a real prediction from the trained SVR model.

---

## API Endpoints

### POST /predict

**Request body (JSON):**
```json
{
  "Item_MRP": 249.80,
  "Outlet_Type": 3,
  "Outlet_Age": 25,
  "Item_Visibility": 0.016,
  "Item_Weight": 9.30,
  "Item_Fat_Content": 0
}
```

**Response:**
```json
{
  "success": true,
  "svr_prediction": 3521.45,
  "lr_prediction": 3104.22,
  "log_class": "High",
  "log_proba": {
    "Low": 0.041,
    "Medium": 0.198,
    "High": 0.761
  },
  "category": "High",
  "features_used": ["Item_MRP", "Outlet_Type", "Outlet_Age", "Item_Visibility", "Item_Weight", "Item_Fat_Content"]
}
```

### GET /model-info

Returns all model metrics and configuration as JSON. Useful for live demonstration.

```
http://localhost:5000/model-info
```

---

## Work Completed So Far

| # | Module | Status | Output |
|---|---|---|---|
| 1 | Data Collection | ✅ Complete | Train.csv — 8,523 records |
| 2 | EDA | ✅ Complete | Distribution analysis, correlation heatmap, missing value treatment |
| 3 | Preprocessing | ✅ Complete | Imputation, encoding, feature engineering (Outlet_Age) |
| 4 | Feature Selection | ✅ Complete | 3 methods applied, 6 features selected |
| 5 | Model Training — Linear Regression | ✅ Complete | R² = 0.503, saved in .pkl |
| 6 | Model Training — Logistic Regression | ✅ Complete | Accuracy = 67.1%, saved in .pkl |
| 7 | Model Training — SVR (RBF) | ✅ Complete | R² = 0.603, saved in .pkl |
| 8 | Cross Validation | ✅ Complete | 5-fold CV on all models |
| 9 | Threat Analysis | ✅ Complete | 4 risk flags evaluated |
| 10 | Model Serialisation | ✅ Complete | all_models.pkl — bundle of models + scaler |
| 11 | Flask Backend | ✅ Complete | /predict and /model-info endpoints active |
| 12 | Frontend UI | ✅ Complete | 5-tab HTML/CSS dashboard, connected to Flask |
| 13 | Documentation | ✅ Complete | This README |

### Pending / Future Scope

| Item | Description |
|---|---|
| Hyperparameter Tuning | GridSearchCV on SVR (C, epsilon, gamma) for improved R² |
| Additional Features | Incorporate seasonality, promotional data if available |
| Deployment | Host on cloud platform (AWS / Heroku / Render) |
| Report / PPT | Formal project report and Gamma AI presentation |

---

## Key Takeaways

- `Item_MRP` is the single most dominant predictor of sales (correlation = 0.57, MI = 0.76)
- SVR with RBF kernel outperforms Linear Regression by 20% in R² due to its capacity to model non-linear relationships
- Logistic Regression, applied after discretising the target into 3 categories, achieves 67.1% classification accuracy
- The model exhibits mild underestimation bias (+₹101 mean residual) — a known limitation of the feature set
- The complete system — from raw CSV to live web-based prediction — is operational and evaluation-ready

---

*BizForecast · Ankur Pratap Singh · B.Tech Major Project*
