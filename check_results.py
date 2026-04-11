import torch
import numpy as np
import pickle
from torch.utils.data import Dataset

# ── Column definitions must be re-declared here so torch.load can reconstruct the Dataset object ──

categorical_cols      = ["SEX", "EDUCATION", "MARRIAGE"]
static_num_cols       = ["LIMIT_BAL", "AGE"]
static_eng_cols       = ["BILL_TREND", "PAY_AMT_TREND", "DELAY_COUNT",
                          "MAX_DELAY", "AVG_UTIL_RATE"]
bill_neg_cols         = [f"BILL_NEG_{i}" for i in range(1, 7)]
pay_cols              = ["PAY_6",  "PAY_5",  "PAY_4",  "PAY_3",  "PAY_2",  "PAY_0"]
bill_cols             = ["BILL_AMT6","BILL_AMT5","BILL_AMT4",
                          "BILL_AMT3","BILL_AMT2","BILL_AMT1"]
amt_cols              = ["PAY_AMT6","PAY_AMT5","PAY_AMT4",
                          "PAY_AMT3","PAY_AMT2","PAY_AMT1"]
monthly_util_cols     = [f"UTIL_RATE_{i}" for i in range(6, 0, -1)]
monthly_pay_rate_cols = [f"PAY_RATE_{i}"  for i in range(6, 0, -1)]

# ── CreditCardDataset must be re-defined here (must match the class in Ryan_Data_Cleaning.py) ──
class CreditCardDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)
        self._static_num_cols = static_num_cols + static_eng_cols + bill_neg_cols
        self._n_months = 6

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        row = self.X.iloc[idx]
        static_num = torch.tensor(
            row[self._static_num_cols].values.astype(float), dtype=torch.float32)
        static_cat = torch.tensor(
            [int(row["SEX"]), int(row["EDUCATION"]), int(row["MARRIAGE"])],
            dtype=torch.long)
        monthly_num, monthly_pay = [], []
        for b, a, u, pr, p in zip(bill_cols, amt_cols,
                                   monthly_util_cols, monthly_pay_rate_cols, pay_cols):
            monthly_num.append([float(row[b]), float(row[a]),
                                 float(row[u]), float(row[pr])])
            monthly_pay.append(int(row[p]))
        monthly_num = torch.tensor(monthly_num, dtype=torch.float32)
        monthly_pay = torch.tensor(monthly_pay, dtype=torch.long)
        label = torch.tensor(float(self.y.iloc[idx]), dtype=torch.float32)
        return {"static_num": static_num, "static_cat": static_cat,
                "monthly_num": monthly_num, "monthly_pay": monthly_pay, "label": label}


# ── 1. Load Metadata ──────────────────────────────────────────────────────────
with open("processed/metadata.pkl", "rb") as f:
    meta = pickle.load(f)

print("=" * 50)
print("METADATA SUMMARY")
print("=" * 50)
print(f"Transformer sequence length : {meta['seq_len']} tokens")
print(f"Static numerical features   : {meta['n_static_num']}")
print(f"Static categorical features : {meta['n_static_cat']}")
print(f"Number of monthly tokens    : {meta['n_months']}")
print(f"Features per monthly token  : {meta['n_monthly_num']}")
print(f"Categorical vocab sizes     : {meta['vocab_sizes']}")


# ── 2. Load PyTorch Datasets (for Transformer model) ─────────────────────────
# weights_only=False is required because we saved custom Dataset objects, not just model weights
train_ds = torch.load("processed/train_dataset.pt", weights_only=False)
val_ds   = torch.load("processed/val_dataset.pt",   weights_only=False)
test_ds  = torch.load("processed/test_dataset.pt",  weights_only=False)

print("\n" + "=" * 50)
print("PYTORCH DATASET SIZES")
print("=" * 50)
print(f"Training set   : {len(train_ds):,} records")
print(f"Validation set : {len(val_ds):,} records")
print(f"Test set       : {len(test_ds):,} records")

# Inspect a single sample to verify the token structure
sample = train_ds[0]
print("\nSingle sample tensor shapes:")
for k, v in sample.items():
    print(f"  {k:15s}: shape={list(v.shape)}, dtype={v.dtype}")


# ── 3. Load Random Forest Arrays ─────────────────────────────────────────────
X_train_rf = np.load("processed/X_train_rf.npy")
X_val_rf   = np.load("processed/X_val_rf.npy")
X_test_rf  = np.load("processed/X_test_rf.npy")
y_train    = np.load("processed/y_train.npy")
y_val      = np.load("processed/y_val.npy")
y_test     = np.load("processed/y_test.npy")

print("\n" + "=" * 50)
print("RANDOM FOREST DATA")
print("=" * 50)
print(f"Training set   : {X_train_rf.shape}  ({X_train_rf.shape[1]} features after one-hot encoding)")
print(f"Validation set : {X_val_rf.shape}")
print(f"Test set       : {X_test_rf.shape}")
print(f"Training default rate   : {y_train.mean():.2%}  (balanced by SMOTE)")
print(f"Validation default rate : {y_val.mean():.2%}  (original distribution)")
print(f"Test default rate       : {y_test.mean():.2%}  (original distribution)")

print("\n" + "=" * 50)
print("All outputs loaded successfully!")
print("=" * 50)
