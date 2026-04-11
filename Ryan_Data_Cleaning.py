# =============================================================================
# Ryan_Data_Cleaning.py
# CW2 – Credit Card Default Prediction
# Responsibility: Data Cleansing, Transformation, Tokenisation, Normalisation
#
# Combines the best of both group approaches:
#   - Rigorous data cleaning & leakage-free pipeline (from 01_data_preprocessing)
#   - Temporal 7-token tokenisation scheme           (from Ifte0002)
#   - New: EDA visualisations
#   - New: Log transformation for skewed columns
#   - New: Feature engineering (utilisation rate, payment rate, trend, delay count)
#   - New: Outlier detection & capping
#   - New: SMOTE oversampling (upgrade from random oversampling)
# =============================================================================

# ── 1. IMPORTS ────────────────────────────────────────────────────────────────
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import resample
from scipy.sparse import issparse

# SMOTE for advanced oversampling (install with: pip install imbalanced-learn)
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("[Warning] imbalanced-learn not found. Falling back to random oversampling.")
    print("          Install with: pip install imbalanced-learn")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# Create output directories
os.makedirs("processed", exist_ok=True)
os.makedirs("figures",   exist_ok=True)

print("=" * 60)
print("  Ryan_Data_Cleaning.py — CW2 Data Pipeline")
print("=" * 60)


# ── 2. LOAD & INSPECT DATA ───────────────────────────────────────────────────
print("\n[Step 1] Loading data...")

# The CSV has two header rows — row 0 is X1/X2..., row 1 is the real names
df = pd.read_csv("clients.csv", header=1)
df = df.drop(columns=["ID"])
df = df.rename(columns={"default payment next month": "target"})

print(f"  Shape      : {df.shape}")
print(f"  Columns    : {list(df.columns)}")
print(f"  Null values: {df.isnull().sum().sum()} (total)")
print(f"  Default rate: {df['target'].mean():.2%}")


# ── 3. EXPLORATORY DATA ANALYSIS (EDA) ───────────────────────────────────────
print("\n[Step 2] Running EDA & saving figures...")

# 3a. Class distribution
fig, ax = plt.subplots(figsize=(5, 4))
counts = df["target"].value_counts()
ax.bar(["No Default (0)", "Default (1)"], counts.values,
       color=["steelblue", "tomato"])
ax.set_ylabel("Count")
ax.set_title("Class Distribution")
for i, v in enumerate(counts.values):
    ax.text(i, v + 200, f"{v:,}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("figures/01_class_distribution.png", dpi=150)
plt.close()

# 3b. Distribution of key continuous features (check for skewness)
cont_sample = ["LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1"]
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, col in zip(axes, cont_sample):
    df[col].hist(bins=60, ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(col)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
plt.suptitle("Feature Distributions (Before Log Transform)", fontsize=13)
plt.tight_layout()
plt.savefig("figures/02_feature_distributions_before.png", dpi=150)
plt.close()

# 3c. Correlation heatmap of continuous features vs target
corr_cols = ["LIMIT_BAL", "AGE", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
             "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_0", "target"]
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f",
            cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation Heatmap (Selected Features)")
plt.tight_layout()
plt.savefig("figures/03_correlation_heatmap.png", dpi=150)
plt.close()

# 3d. PAY_0 distribution by default status
fig, ax = plt.subplots(figsize=(8, 4))
df.groupby(["PAY_0", "target"]).size().unstack().plot(
    kind="bar", ax=ax, color=["steelblue", "tomato"])
ax.set_title("PAY_0 (Repayment Status) vs Default")
ax.set_xlabel("PAY_0 Value")
ax.set_ylabel("Count")
ax.legend(["No Default", "Default"])
plt.tight_layout()
plt.savefig("figures/04_pay0_vs_default.png", dpi=150)
plt.close()

print("  EDA figures saved to /figures/")


# ── 4. DATA CLEANING ─────────────────────────────────────────────────────────
print("\n[Step 3] Cleaning noisy categorical values...")

# EDUCATION: documented values 1=graduate, 2=university, 3=high school, 4=other
# Values 0, 5, 6 are undocumented → map to 4 (other)
before_edu = df["EDUCATION"].value_counts().to_dict()
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
print(f"  EDUCATION cleaned: {before_edu} → {df['EDUCATION'].value_counts().to_dict()}")

# MARRIAGE: documented values 1=married, 2=single, 3=other
# Value 0 is undocumented → map to 3 (other)
before_mar = df["MARRIAGE"].value_counts().to_dict()
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
print(f"  MARRIAGE  cleaned: {before_mar} → {df['MARRIAGE'].value_counts().to_dict()}")


# ── 5. OUTLIER DETECTION & CAPPING ───────────────────────────────────────────
print("\n[Step 4] Capping outliers using IQR method...")

# Mark negative BILL amounts (overpayment / credit balance — useful signal)
for i in range(1, 7):
    df[f"BILL_NEG_{i}"] = (df[f"BILL_AMT{i}"] < 0).astype(int)

# Cap extreme values at 1st and 99th percentile for financial columns
cap_cols = [f"BILL_AMT{i}" for i in range(1, 7)] + \
           [f"PAY_AMT{i}"  for i in range(1, 7)] + \
           ["LIMIT_BAL"]

for col in cap_cols:
    p01 = df[col].quantile(0.01)
    p99 = df[col].quantile(0.99)
    df[col] = df[col].clip(lower=p01, upper=p99)

print(f"  Capped {len(cap_cols)} columns at 1st–99th percentile")
print(f"  Added {6} BILL_NEG_* indicator columns for negative bill amounts")


# ── 6. FEATURE ENGINEERING ───────────────────────────────────────────────────
print("\n[Step 5] Engineering new features...")

# Credit utilisation rate: how much of the credit limit is being used
# Higher utilisation → higher financial pressure → higher default risk
for i in range(1, 7):
    df[f"UTIL_RATE_{i}"] = df[f"BILL_AMT{i}"] / (df["LIMIT_BAL"] + 1)

# Payment rate: how much of the bill was actually paid back
# Lower payment rate → higher default risk
for i in range(1, 7):
    df[f"PAY_RATE_{i}"] = df[f"PAY_AMT{i}"] / (df[f"BILL_AMT{i}"].abs() + 1)

# Bill trend: is the outstanding balance growing or shrinking?
# Positive (growing) → higher default risk
df["BILL_TREND"] = df["BILL_AMT1"] - df["BILL_AMT6"]

# Payment trend: is the customer paying more or less over time?
df["PAY_AMT_TREND"] = df["PAY_AMT1"] - df["PAY_AMT6"]

# Consecutive delay count: how many months did the customer delay payment?
# (PAY > 0 means delayed; values: 1=1 month delay, ..., 8=8 months delay)
pay_status_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
df["DELAY_COUNT"] = (df[pay_status_cols] > 0).sum(axis=1)

# Max delay severity: worst single-month delay
df["MAX_DELAY"] = df[pay_status_cols].max(axis=1)

# Average utilisation across 6 months
df["AVG_UTIL_RATE"] = df[[f"UTIL_RATE_{i}" for i in range(1, 7)]].mean(axis=1)

engineered = (
    [f"UTIL_RATE_{i}" for i in range(1, 7)] +
    [f"PAY_RATE_{i}"  for i in range(1, 7)] +
    ["BILL_TREND", "PAY_AMT_TREND", "DELAY_COUNT", "MAX_DELAY", "AVG_UTIL_RATE"]
)
print(f"  Created {len(engineered)} engineered features")


# ── 7. LOG TRANSFORMATION FOR SKEWED COLUMNS ─────────────────────────────────
print("\n[Step 6] Applying log transformation to skewed financial columns...")

# BILL_AMT and PAY_AMT are heavily right-skewed.
# np.sign * log1p handles both positive and negative values safely.
log_cols = [f"BILL_AMT{i}" for i in range(1, 7)] + \
           [f"PAY_AMT{i}"  for i in range(1, 7)]

for col in log_cols:
    df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))

print(f"  Log-transformed {len(log_cols)} columns")

# Plot distributions AFTER log transform for comparison
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, col in zip(axes, cont_sample):
    df[col].hist(bins=60, ax=ax, color="darkorange", edgecolor="white")
    ax.set_title(col)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
plt.suptitle("Feature Distributions (After Log Transform)", fontsize=13)
plt.tight_layout()
plt.savefig("figures/05_feature_distributions_after.png", dpi=150)
plt.close()


# ── 8. DEFINE FEATURE GROUPS ─────────────────────────────────────────────────
print("\n[Step 7] Defining feature groups...")

# --- For Transformer (temporal 7-token scheme, inspired by Ifte0002) ---
# Token 0: Static demographic & financial profile
# Tokens 1-6: Monthly behaviour (one token per month, oldest to newest)

categorical_cols = ["SEX", "EDUCATION", "MARRIAGE"]   # static categorical
static_num_cols  = ["LIMIT_BAL", "AGE"]                # static numerical

# Temporal columns ordered oldest → newest (month 6 → month 1)
pay_cols  = ["PAY_6",  "PAY_5",  "PAY_4",  "PAY_3",  "PAY_2",  "PAY_0"]
bill_cols = ["BILL_AMT6", "BILL_AMT5", "BILL_AMT4",
             "BILL_AMT3", "BILL_AMT2", "BILL_AMT1"]
amt_cols  = ["PAY_AMT6", "PAY_AMT5", "PAY_AMT4",
             "PAY_AMT3", "PAY_AMT2", "PAY_AMT1"]

# Engineered features (added to static token)
static_eng_cols = ["BILL_TREND", "PAY_AMT_TREND", "DELAY_COUNT",
                   "MAX_DELAY", "AVG_UTIL_RATE"]

# Monthly engineered features (per-month, added to monthly tokens)
monthly_util_cols = [f"UTIL_RATE_{i}" for i in range(6, 0, -1)]   # 6→1
monthly_pay_rate_cols = [f"PAY_RATE_{i}" for i in range(6, 0, -1)]  # 6→1

# Bill negativity indicators (static — summarise credit behaviour)
bill_neg_cols = [f"BILL_NEG_{i}" for i in range(1, 7)]

# All features for Random Forest (flat structure)
all_feature_cols = (categorical_cols + static_num_cols + static_eng_cols +
                    bill_neg_cols + pay_cols + bill_cols + amt_cols +
                    monthly_util_cols + monthly_pay_rate_cols)

print(f"  Categorical (static)    : {len(categorical_cols)}")
print(f"  Numerical (static)      : {len(static_num_cols)}")
print(f"  Engineered (static)     : {len(static_eng_cols)}")
print(f"  Temporal pay cols       : {len(pay_cols)} months")
print(f"  Temporal bill/amt cols  : {len(bill_cols)} + {len(amt_cols)} months")
print(f"  Monthly util/pay_rate   : {len(monthly_util_cols)} + {len(monthly_pay_rate_cols)}")
print(f"  Total RF features       : {len(all_feature_cols)}")


# ── 9. TRAIN / VALIDATION / TEST SPLIT ───────────────────────────────────────
print("\n[Step 8] Splitting data (70% train / 15% val / 15% test)...")

X = df[all_feature_cols].copy()
y = df["target"].copy()

# Split BEFORE any scaling or oversampling to prevent data leakage
X_train_raw, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y)

X_val_raw, X_test_raw, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp)

print(f"  Train : {len(X_train_raw):,}  (default rate: {y_train.mean():.2%})")
print(f"  Val   : {len(X_val_raw):,}   (default rate: {y_val.mean():.2%})")
print(f"  Test  : {len(X_test_raw):,}   (default rate: {y_test.mean():.2%})")


# ── 10. CLASS IMBALANCE — SMOTE (or fallback to random oversampling) ──────────
print("\n[Step 9] Handling class imbalance (training set only)...")

continuous_for_smote = static_num_cols + static_eng_cols + bill_neg_cols + \
                       bill_cols + amt_cols + monthly_util_cols + monthly_pay_rate_cols
cat_for_smote = categorical_cols + pay_cols   # treated as int for SMOTE

if SMOTE_AVAILABLE:
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_raw, y_train)
    X_train_balanced = pd.DataFrame(X_train_balanced, columns=all_feature_cols)
    y_train_balanced = pd.Series(y_train_balanced, name="target")
    print(f"  SMOTE applied → balanced train size: {len(X_train_balanced):,}")
else:
    # Fallback: random oversampling
    train_df_tmp = X_train_raw.copy()
    train_df_tmp["target"] = y_train.values
    majority = train_df_tmp[train_df_tmp["target"] == 0]
    minority = train_df_tmp[train_df_tmp["target"] == 1]
    minority_up = resample(minority, replace=True,
                           n_samples=len(majority), random_state=RANDOM_STATE)
    balanced = pd.concat([majority, minority_up]).sample(
        frac=1, random_state=RANDOM_STATE)
    X_train_balanced = balanced[all_feature_cols]
    y_train_balanced = balanced["target"]
    print(f"  Random oversampling → balanced train size: {len(X_train_balanced):,}")

print(f"  Default rate after balancing: {y_train_balanced.mean():.2%}")


# ── 11. NORMALISATION ─────────────────────────────────────────────────────────
print("\n[Step 10] Normalising continuous features (StandardScaler)...")

# Features to scale: everything except categorical & PAY status (ordinal)
scale_cols = (static_num_cols + static_eng_cols + bill_neg_cols +
              bill_cols + amt_cols + monthly_util_cols + monthly_pay_rate_cols)

# Fit ONLY on training data, then apply to val and test (no leakage)
scaler = StandardScaler()

X_train_scaled = X_train_balanced.copy()
X_val_scaled   = X_val_raw.copy()
X_test_scaled  = X_test_raw.copy()

X_train_scaled[scale_cols] = scaler.fit_transform(X_train_balanced[scale_cols])
X_val_scaled[scale_cols]   = scaler.transform(X_val_raw[scale_cols])
X_test_scaled[scale_cols]  = scaler.transform(X_test_raw[scale_cols])

print(f"  Scaled {len(scale_cols)} continuous columns")
print(f"  Train mean (sample): {X_train_scaled[scale_cols[:3]].mean().round(3).to_dict()}")
print(f"  Train std  (sample): {X_train_scaled[scale_cols[:3]].std().round(3).to_dict()}")


# ── 12. CATEGORICAL RE-INDEXING (0-based for Embedding layers) ───────────────
print("\n[Step 11] Re-indexing categoricals to 0-based integers...")

for split in [X_train_scaled, X_val_scaled, X_test_scaled]:
    split["SEX"]       = split["SEX"]       - 1        # 1,2       → 0,1
    split["EDUCATION"] = split["EDUCATION"] - 1        # 1,2,3,4   → 0,1,2,3
    split["MARRIAGE"]  = split["MARRIAGE"]  - 1        # 1,2,3     → 0,1,2
    split[pay_cols]    = split[pay_cols]    + 2        # -2..8     → 0..10

vocab_sizes = {
    "SEX"       : int(X_train_scaled["SEX"].max())       + 1,   # 2
    "EDUCATION" : int(X_train_scaled["EDUCATION"].max()) + 1,   # 4
    "MARRIAGE"  : int(X_train_scaled["MARRIAGE"].max())  + 1,   # 3
}
for p in pay_cols:
    vocab_sizes[p] = int(X_train_scaled[p].max()) + 1           # 11

print("  Vocabulary sizes:")
for k, v in vocab_sizes.items():
    print(f"    {k}: {v}")


# ── 13. PYTORCH DATASET — TEMPORAL 7-TOKEN SCHEME ────────────────────────────
print("\n[Step 12] Building PyTorch Datasets (temporal 7-token scheme)...")

#  Token 0  : static token  → [LIMIT_BAL, AGE] (num) + [SEX, EDUCATION, MARRIAGE] (cat)
#             + engineered static features + BILL_NEG indicators
#  Tokens 1–6: monthly tokens → [BILL_AMT, PAY_AMT, UTIL_RATE, PAY_RATE] (num)
#             + [PAY_status] (cat), one token per month (oldest→newest)

class CreditCardDataset(Dataset):
    """
    Returns a structured dict per sample:
      static_num  : FloatTensor (n_static_num,)       — scaled static numerics
      static_cat  : LongTensor  (3,)                  — SEX, EDUCATION, MARRIAGE
      monthly_num : FloatTensor (6, 4)                — per month: BILL, PAY_AMT, UTIL, PAY_RATE
      monthly_pay : LongTensor  (6,)                  — per month: repayment status (0-indexed)
      label       : FloatTensor scalar                — 0 or 1
    """
    def __init__(self, X, y):
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)

        self._static_num_cols = static_num_cols + static_eng_cols + bill_neg_cols
        self._n_months = 6

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        row = self.X.iloc[idx]

        # Static numerical token
        static_num = torch.tensor(
            row[self._static_num_cols].values.astype(float),
            dtype=torch.float32
        )

        # Static categorical token
        static_cat = torch.tensor(
            [int(row["SEX"]), int(row["EDUCATION"]), int(row["MARRIAGE"])],
            dtype=torch.long
        )

        # Monthly tokens: 6 months, each with 4 numeric values + 1 PAY status
        monthly_num = []
        monthly_pay = []
        for b, a, u, pr, p in zip(bill_cols, amt_cols,
                                   monthly_util_cols, monthly_pay_rate_cols,
                                   pay_cols):
            monthly_num.append([float(row[b]), float(row[a]),
                                 float(row[u]), float(row[pr])])
            monthly_pay.append(int(row[p]))

        monthly_num = torch.tensor(monthly_num, dtype=torch.float32)   # (6, 4)
        monthly_pay = torch.tensor(monthly_pay, dtype=torch.long)      # (6,)

        label = torch.tensor(float(self.y.iloc[idx]), dtype=torch.float32)

        return {
            "static_num"  : static_num,
            "static_cat"  : static_cat,
            "monthly_num" : monthly_num,
            "monthly_pay" : monthly_pay,
            "label"       : label
        }


train_dataset = CreditCardDataset(X_train_scaled, y_train_balanced)
val_dataset   = CreditCardDataset(X_val_scaled,   y_val)
test_dataset  = CreditCardDataset(X_test_scaled,  y_test)

BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# Sanity check
sample = next(iter(train_loader))
print("  Batch shapes:")
for k, v in sample.items():
    print(f"    {k:15s}: {list(v.shape)}")


# ── 14. RANDOM FOREST PIPELINE ───────────────────────────────────────────────
print("\n[Step 13] Preparing Random Forest pipeline (OneHotEncoder)...")

# RF uses unbalanced training data with class weights instead of SMOTE
# (RF handles imbalance differently via class_weight='balanced')
rf_cat_cols = categorical_cols + pay_cols
rf_num_cols = [c for c in all_feature_cols if c not in rf_cat_cols]

preprocessor_rf = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), rf_cat_cols),
    ("num", "passthrough", rf_num_cols)
])

# Fit ONLY on the original (unbalanced) training set
X_train_rf = preprocessor_rf.fit_transform(X_train_raw)
X_val_rf   = preprocessor_rf.transform(X_val_raw)
X_test_rf  = preprocessor_rf.transform(X_test_raw)

if issparse(X_train_rf):
    X_train_rf = X_train_rf.toarray()
    X_val_rf   = X_val_rf.toarray()
    X_test_rf  = X_test_rf.toarray()

print(f"  RF train shape: {X_train_rf.shape}")
print(f"  RF val   shape: {X_val_rf.shape}")
print(f"  RF test  shape: {X_test_rf.shape}")


# ── 15. SAVE ALL OUTPUTS ─────────────────────────────────────────────────────
print("\n[Step 14] Saving all processed outputs...")

# Transformer datasets
torch.save(train_dataset, "processed/train_dataset.pt")
torch.save(val_dataset,   "processed/val_dataset.pt")
torch.save(test_dataset,  "processed/test_dataset.pt")

# RF arrays
np.save("processed/X_train_rf.npy", X_train_rf)
np.save("processed/X_val_rf.npy",   X_val_rf)
np.save("processed/X_test_rf.npy",  X_test_rf)
np.save("processed/y_train.npy",    y_train_balanced.values)
np.save("processed/y_val.npy",      y_val.values)
np.save("processed/y_test.npy",     y_test.values)

# Metadata for model-building teammates
metadata = {
    # Feature groups
    "categorical_cols"     : categorical_cols,
    "static_num_cols"      : static_num_cols,
    "static_eng_cols"      : static_eng_cols,
    "bill_neg_cols"        : bill_neg_cols,
    "pay_cols"             : pay_cols,
    "bill_cols"            : bill_cols,
    "amt_cols"             : amt_cols,
    "monthly_util_cols"    : monthly_util_cols,
    "monthly_pay_rate_cols": monthly_pay_rate_cols,
    "all_feature_cols"     : all_feature_cols,
    # Transformer architecture hints
    "n_static_num"   : len(static_num_cols + static_eng_cols + bill_neg_cols),
    "n_static_cat"   : len(categorical_cols),   # 3
    "n_months"       : 6,
    "n_monthly_num"  : 4,    # BILL_AMT, PAY_AMT, UTIL_RATE, PAY_RATE per month
    "seq_len"        : 7,    # 1 static token + 6 monthly tokens
    "vocab_sizes"    : vocab_sizes,
    # Preprocessing objects
    "scaler"         : scaler,
    "preprocessor_rf": preprocessor_rf,
}

with open("processed/metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("  Saved: processed/train_dataset.pt")
print("  Saved: processed/val_dataset.pt")
print("  Saved: processed/test_dataset.pt")
print("  Saved: processed/X_*_rf.npy, y_*.npy")
print("  Saved: processed/metadata.pkl")


# ── 16. PIPELINE SUMMARY ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PIPELINE SUMMARY")
print("=" * 60)
summary = {
    "Dataset"              : "UCI Credit Card Default — 30,000 records",
    "Missing values"       : "None",
    "Outlier handling"     : "1st–99th percentile capping",
    "Categorical cleaning" : "EDUCATION {0,5,6}→4; MARRIAGE {0}→3",
    "Feature engineering"  : f"{len(engineered)} new features created",
    "Log transform"        : f"{len(log_cols)} skewed columns",
    "Split"                : "70% train / 15% val / 15% test (stratified)",
    "Class imbalance"      : "SMOTE" if SMOTE_AVAILABLE else "Random oversampling",
    "Normalisation"        : "StandardScaler (fit on train only — no leakage)",
    "Tokenisation scheme"  : "Temporal 7-token: 1 static + 6 monthly tokens",
    "Transformer inputs"   : "static_num, static_cat, monthly_num, monthly_pay",
    "RF inputs"            : f"OneHotEncoder on cats; passthrough on nums → {X_train_rf.shape[1]} features",
}
for k, v in summary.items():
    print(f"  {k:<24}: {v}")

print("\n  Key metadata for model-building teammates:")
print(f"    seq_len       : {metadata['seq_len']}  (transformer sequence length)")
print(f"    n_static_num  : {metadata['n_static_num']}  (static numerical features)")
print(f"    n_static_cat  : {metadata['n_static_cat']}   (SEX, EDUCATION, MARRIAGE)")
print(f"    n_months      : {metadata['n_months']}   (temporal tokens)")
print(f"    n_monthly_num : {metadata['n_monthly_num']}   (features per monthly token)")
print(f"    vocab_sizes   : {metadata['vocab_sizes']}")
print("\n  All outputs saved to /processed/")
print("  All figures  saved to /figures/")
print("=" * 60)
print("  Ryan_Data_Cleaning.py — COMPLETE")
print("=" * 60)
