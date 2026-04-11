# Ryan_Data_Cleaning.py
**CW2 Group Project — Data Preprocessing & Transformation**
Authored by: Ryan Lin

---

## Overview

This script handles all data preprocessing for the CW2 credit card default prediction project. It prepares cleaned, transformed, and tokenised data for both the **Transformer model** and the **Random Forest benchmark**.

The processed outputs are saved to the `processed/` folder so that model-building teammates can load them directly without re-running this script.

---

## What This Script Does

### 1. Load Data
Loads `clients.csv` (UCI Credit Card Default dataset — 30,000 records, 24 features). Handles the double-header row, drops the ID column, and renames the target column.

### 2. Exploratory Data Analysis (EDA)
Generates 5 charts saved to `figures/`:

| File | Content |
|------|---------|
| `01_class_distribution.png` | Default vs non-default count |
| `02_feature_distributions_before.png` | Feature distributions before log transform |
| `03_correlation_heatmap.png` | Feature correlation with default |
| `04_pay0_vs_default.png` | Repayment status vs default rate |
| `05_feature_distributions_after.png` | Feature distributions after log transform |

### 3. Data Cleaning
- Remaps undocumented `EDUCATION` values (0, 5, 6) → 4 ("other")
- Remaps undocumented `MARRIAGE` value (0) → 3 ("other")

### 4. Outlier Handling
- Caps all financial columns at the 1st–99th percentile
- Creates 6 binary `BILL_NEG_*` indicator columns for negative bill amounts (overpayment signal)

### 5. Feature Engineering
Creates 14 new features from existing columns:

| Feature | Formula | Meaning |
|---------|---------|---------|
| `UTIL_RATE_1~6` | Bill ÷ Credit limit | Monthly financial pressure |
| `PAY_RATE_1~6` | Payment ÷ Bill amount | Monthly repayment ability |
| `BILL_TREND` | Latest bill − Oldest bill | Is debt growing? |
| `PAY_AMT_TREND` | Latest payment − Oldest payment | Is repayment declining? |
| `DELAY_COUNT` | Months with delayed payment | Habitual lateness |
| `MAX_DELAY` | Worst single-month delay | Severity of worst behaviour |
| `AVG_UTIL_RATE` | Mean utilisation across 6 months | Overall financial strain |

### 6. Log Transformation
Applies `sign * log1p(abs(x))` to all `BILL_AMT` and `PAY_AMT` columns to correct heavy right skew, making distributions more symmetric and easier to learn from.

### 7. Train / Validation / Test Split
Splits data **before** any scaling or oversampling to prevent data leakage:
- **70%** Training
- **15%** Validation
- **15%** Test

All splits are stratified to preserve the 22% default rate.

### 8. Class Imbalance — SMOTE
Uses SMOTE (Synthetic Minority Over-sampling Technique) to balance the training set from 22% → 50% default rate. Applied to the **training set only**. Falls back to random oversampling if `imbalanced-learn` is not installed.

### 9. Normalisation
Applies `StandardScaler` (mean=0, std=1) to all continuous features. The scaler is **fit on training data only**, then applied to validation and test sets.

### 10. Categorical Re-indexing
Shifts all categorical values to 0-based indexing for PyTorch Embedding layers:
- `SEX`: 1,2 → 0,1
- `EDUCATION`: 1,2,3,4 → 0,1,2,3
- `MARRIAGE`: 1,2,3 → 0,1,2
- `PAY` columns: −2..8 → 0..10

### 11. Tokenisation — Temporal 7-Token Scheme
Each customer record is structured as a **sequence of 7 tokens** for the Transformer:

```
Token 0     : Static profile (demographics + engineered features)
Token 1     : Month 6 behaviour (oldest)
Token 2     : Month 5 behaviour
Token 3     : Month 4 behaviour
Token 4     : Month 3 behaviour
Token 5     : Month 2 behaviour
Token 6     : Month 1 behaviour (most recent)
```

Each monthly token contains 4 numeric values (`BILL_AMT`, `PAY_AMT`, `UTIL_RATE`, `PAY_RATE`) and 1 repayment status category. This lets the Transformer learn temporal patterns across months.

### 12. Random Forest Pipeline
Uses `OneHotEncoder` on categorical features and passes continuous features through unchanged. Tree-based models do not require feature scaling.

---

## Output Files

All outputs are saved to `processed/`:

| File | Description | Used By |
|------|-------------|---------|
| `train_dataset.pt` | PyTorch Dataset (training) | Transformer teammate |
| `val_dataset.pt` | PyTorch Dataset (validation) | Transformer teammate |
| `test_dataset.pt` | PyTorch Dataset (test) | Transformer teammate |
| `X_train_rf.npy` | RF training features | Random Forest teammate |
| `X_val_rf.npy` | RF validation features | Random Forest teammate |
| `X_test_rf.npy` | RF test features | Random Forest teammate |
| `y_train.npy` | Training labels | Both |
| `y_val.npy` | Validation labels | Both |
| `y_test.npy` | Test labels | Both |
| `metadata.pkl` | Feature lists, vocab sizes, scalers | Both |

### Key Metadata for Model Teammates

```python
import pickle
with open("processed/metadata.pkl", "rb") as f:
    meta = pickle.load(f)

# Transformer architecture hints
meta["seq_len"]        # 7  — sequence length (1 static + 6 monthly tokens)
meta["n_static_num"]   # 13 — number of static numerical features
meta["n_static_cat"]   # 3  — SEX, EDUCATION, MARRIAGE
meta["n_months"]       # 6  — number of monthly tokens
meta["n_monthly_num"]  # 4  — features per monthly token
meta["vocab_sizes"]    # dict of embedding vocab sizes per categorical feature
```

---

## Folder Structure

```
Ryan_Data_Preprocessing/
├── Ryan_Data_Cleaning.py    # Main preprocessing script (this file)
├── check_results.py         # Script to verify processed outputs
├── README.md                # This file
├── clients.csv              # Raw dataset (add this yourself — see below)
├── figures/                 # Auto-generated EDA charts
│   ├── 01_class_distribution.png
│   ├── 02_feature_distributions_before.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_pay0_vs_default.png
│   └── 05_feature_distributions_after.png
└── processed/               # Auto-generated model-ready outputs
    ├── train_dataset.pt
    ├── val_dataset.pt
    ├── test_dataset.pt
    ├── X_train_rf.npy
    ├── X_val_rf.npy
    ├── X_test_rf.npy
    ├── y_train.npy
    ├── y_val.npy
    ├── y_test.npy
    └── metadata.pkl
```

---

## Setup & Usage

### Step 1 — Get the Dataset
Download `clients.csv` from either source and place it in this folder:
- Kaggle: https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
- UCI: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

### Step 2 — Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### Step 3 — Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch imbalanced-learn scipy
```

### Step 4 — Run the Script
```bash
python Ryan_Data_Cleaning.py
```

### Step 5 — Verify Outputs (Optional)
```bash
python check_results.py
```

> **Note:** Run `source venv/bin/activate` every time you open a new terminal window before running the scripts.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Chart generation |
| `seaborn` | Statistical visualisations |
| `scikit-learn` | StandardScaler, OneHotEncoder, train/test split |
| `torch` | PyTorch Dataset and DataLoader |
| `imbalanced-learn` | SMOTE oversampling |
| `scipy` | Sparse matrix handling |

---

## Dataset

**UCI Default of Credit Card Clients**
- 30,000 customer records from a Taiwanese bank (2005)
- 23 features covering credit limit, demographics, repayment history, and bill/payment amounts over 6 months
- Target: whether the customer defaulted the following month (1 = default, 0 = no default)
- Class imbalance: ~22% default, ~78% no default
