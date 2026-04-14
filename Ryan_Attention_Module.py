# =============================================================================
# Ryan_Attention_Module.py
# CW2 – Credit Card Default Prediction
# Responsibility: Attention Mechanism Components for the Transformer Model
#
# This module provides all building blocks needed for the Transformer model:
#   1. Positional Encoding       — tells the model the order of tokens
#   2. Token Embedding           — converts raw input into vectors
#   3. Multi-Head Self-Attention — the core attention mechanism
#   4. Transformer Block         — one full layer (attention + feed-forward)
#   5. CreditTransformer         — the complete model ready to train
#
# Usage:
#   from Ryan_Attention_Module import CreditTransformer, get_model_config
#   config = get_model_config()
#   model  = CreditTransformer(**config)
#
# Designed to work directly with outputs from Ryan_Data_Cleaning.py
# =============================================================================

import math
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ── 1. POSITIONAL ENCODING ────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Adds positional information to token embeddings.

    The Transformer has no built-in sense of order — without this, it treats
    Token 1 (oldest month) and Token 6 (newest month) as identical positions.
    Positional encoding stamps each token with a unique position signal using
    sine and cosine waves of different frequencies.

    Why sine/cosine?
    - They produce unique patterns for every position
    - The model can easily learn relative distances between positions
    - Values stay bounded between -1 and 1 (no scale issues)

    Args:
        d_model (int): Embedding dimension (must match token embeddings)
        max_len (int): Maximum sequence length supported (default 20, we use 7)
    """
    def __init__(self, d_model: int, max_len: int = 20):
        super().__init__()

        # Build a (max_len, d_model) table of positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        # Each dimension gets a different frequency
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)   # even dimensions → sine
        pe[:, 1::2] = torch.cos(position * div_term)   # odd dimensions  → cosine

        # register_buffer: saves pe as part of the model but NOT as a trainable parameter
        self.register_buffer("pe", pe.unsqueeze(0))    # shape: (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token embeddings of shape (batch, seq_len, d_model)
        Returns:
            x with positional encoding added, same shape
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]                # broadcast across batch


# ── 2. TOKEN EMBEDDING LAYER ─────────────────────────────────────────────────

class CreditTokenEmbedding(nn.Module):
    """
    Converts raw tabular input into a sequence of token embeddings.

    Follows the temporal 7-token scheme from Ryan_Data_Cleaning.py:
        Token 0     : Static profile (demographics + engineered features)
        Tokens 1–6  : Monthly behaviour (one token per month, oldest → newest)

    Each categorical feature gets its own Embedding layer.
    Each set of numerical features gets a Linear projection layer.
    All projections map into the same d_model space so they can be combined.

    Args:
        d_model    (int): Embedding dimension for all tokens
        n_static   (int): Number of static numerical features
        vocab_sizes (dict): Vocabulary size for each categorical feature
    """
    def __init__(self, d_model: int, n_static: int, vocab_sizes: dict):
        super().__init__()

        # --- Static categorical embeddings ---
        # Each category (SEX, EDUCATION, MARRIAGE) gets its own lookup table.
        # Embedding(vocab_size, d_model) maps an integer ID → a d_model vector.
        self.sex_emb       = nn.Embedding(vocab_sizes["SEX"],       d_model)
        self.education_emb = nn.Embedding(vocab_sizes["EDUCATION"], d_model)
        self.marriage_emb  = nn.Embedding(vocab_sizes["MARRIAGE"],  d_model)

        # --- Static numerical projection ---
        # Projects n_static numerical features → d_model space
        self.static_num_proj = nn.Linear(n_static, d_model)

        # --- Static token fusion ---
        # Combines all static signals into one unified token
        self.static_fusion = nn.Linear(d_model, d_model)

        # --- Monthly numerical projection ---
        # Each month has 4 values: BILL_AMT, PAY_AMT, UTIL_RATE, PAY_RATE
        self.monthly_num_proj = nn.Linear(4, d_model)

        # --- Monthly repayment status embedding ---
        # PAY_0..PAY_6 are re-indexed to 0–10 in preprocessing (11 possible values)
        # Use the largest vocab size across all PAY columns
        pay_vocab = max(v for k, v in vocab_sizes.items() if k.startswith("PAY"))
        self.pay_status_emb = nn.Embedding(pay_vocab, d_model)

        # --- Month position embedding ---
        # Distinguishes "which month" this token represents (0=oldest, 5=newest)
        self.month_emb = nn.Embedding(6, d_model)

        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        static_num:  torch.Tensor,   # (batch, n_static)
        static_cat:  torch.Tensor,   # (batch, 3)       — SEX, EDUCATION, MARRIAGE
        monthly_num: torch.Tensor,   # (batch, 6, 4)    — per-month numerics
        monthly_pay: torch.Tensor,   # (batch, 6)       — per-month PAY status
    ) -> torch.Tensor:               # (batch, 7, d_model)

        batch_size = static_num.size(0)

        # ── Build Token 0: Static profile ────────────────────────────────────
        # Project numerical static features
        num_vec = self.static_num_proj(static_num)              # (batch, d_model)

        # Look up categorical embeddings and sum them
        sex_vec  = self.sex_emb(static_cat[:, 0])               # (batch, d_model)
        edu_vec  = self.education_emb(static_cat[:, 1])         # (batch, d_model)
        mar_vec  = self.marriage_emb(static_cat[:, 2])          # (batch, d_model)

        # Fuse all static signals into one token
        static_token = self.static_fusion(
            num_vec + sex_vec + edu_vec + mar_vec
        ).unsqueeze(1)                                          # (batch, 1, d_model)

        # ── Build Tokens 1–6: Monthly behaviour ───────────────────────────────
        # Project the 4 numerical values per month
        monthly_num_vec = self.monthly_num_proj(monthly_num)    # (batch, 6, d_model)

        # Look up repayment status embedding for each month
        pay_vec = self.pay_status_emb(monthly_pay)              # (batch, 6, d_model)

        # Add a learned "which month" signal (0=oldest … 5=newest)
        month_idx = torch.arange(6, device=monthly_num.device) \
                         .unsqueeze(0).expand(batch_size, 6)
        month_vec = self.month_emb(month_idx)                   # (batch, 6, d_model)

        # Sum all monthly signals
        monthly_tokens = monthly_num_vec + pay_vec + month_vec  # (batch, 6, d_model)

        # ── Concatenate static token + 6 monthly tokens ───────────────────────
        tokens = torch.cat([static_token, monthly_tokens], dim=1)  # (batch, 7, d_model)

        return self.dropout(tokens)


# ── 3. MULTI-HEAD SELF-ATTENTION ──────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """
    The core attention mechanism.

    Intuition: imagine each token asking "which other tokens should I pay
    attention to?" A token representing a month with severe late payment would
    strongly attend to the final default label. Multi-head means we run
    multiple independent attention operations in parallel, each learning to
    focus on different relationships.

    Mechanically:
        1. Each token is projected into three vectors: Query (Q), Key (K), Value (V)
        2. Attention score = dot product of Q with all K's → how relevant is each token?
        3. Scores are scaled then softmaxed → attention weights (sum to 1)
        4. Output = weighted sum of V's using those attention weights
        5. Multiple heads → split d_model across heads, run in parallel, concatenate

    Args:
        d_model   (int): Total embedding dimension
        num_heads (int): Number of parallel attention heads (d_model must be divisible)
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads   # dimension per head

        # Linear projections for Q, K, V and the output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Token embeddings of shape (batch, seq_len, d_model)
        Returns:
            out          : Attended output,  shape (batch, seq_len, d_model)
            attn_weights : Attention weights, shape (batch, num_heads, seq_len, seq_len)
                           → useful for visualising which tokens attend to which
        """
        batch_size, seq_len, _ = x.shape

        # Project input into Q, K, V
        Q = self.W_q(x)   # (batch, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape into (batch, num_heads, seq_len, head_dim)
        def split_heads(t):
            return t.view(batch_size, seq_len, self.num_heads, self.head_dim) \
                    .transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # Scaled dot-product attention
        # Scale by sqrt(head_dim) to prevent vanishing gradients from large dot products
        scale   = math.sqrt(self.head_dim)
        scores  = torch.matmul(Q, K.transpose(-2, -1)) / scale   # (batch, heads, seq, seq)
        weights = torch.softmax(scores, dim=-1)                   # attention weights

        # Weighted sum of values
        context = torch.matmul(weights, V)                        # (batch, heads, seq, head_dim)

        # Merge heads back → (batch, seq_len, d_model)
        context = context.transpose(1, 2).contiguous() \
                         .view(batch_size, seq_len, self.d_model)

        out = self.W_o(context)
        return out, weights


# ── 4. TRANSFORMER BLOCK ─────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    One complete Transformer layer.

    Structure:
        Input
          ↓
        Multi-Head Self-Attention
          ↓  (+ residual connection → Add & LayerNorm)
        Feed-Forward Network  [Linear → ReLU → Dropout → Linear]
          ↓  (+ residual connection → Add & LayerNorm)
        Output

    Residual connections (x + sublayer(x)) help gradients flow during training.
    LayerNorm stabilises activations so training converges reliably.

    Args:
        d_model  (int): Embedding dimension
        num_heads (int): Number of attention heads
        ff_dim   (int): Hidden size of the feed-forward layer (usually 2–4× d_model)
        dropout  (float): Dropout probability for regularisation
    """
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()

        # Attention sub-layer
        self.attn    = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1   = nn.LayerNorm(d_model)
        self.drop1   = nn.Dropout(dropout)

        # Feed-forward sub-layer
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Shape (batch, seq_len, d_model)
        Returns:
            x           : Output of same shape
            attn_weights: Attention weights for this block (for interpretability)
        """
        # Attention sub-layer with residual connection
        attn_out, attn_weights = self.attn(x)
        x = self.norm1(x + self.drop1(attn_out))

        # Feed-forward sub-layer with residual connection
        ff_out = self.ff(x)
        x = self.norm2(x + self.drop2(ff_out))

        return x, attn_weights


# ── 5. FULL CREDIT TRANSFORMER MODEL ─────────────────────────────────────────

class CreditTransformer(nn.Module):
    """
    Complete Transformer model for credit default prediction.

    Pipeline:
        Raw input (static_num, static_cat, monthly_num, monthly_pay)
          ↓
        CreditTokenEmbedding  → 7 token embeddings of size d_model
          ↓
        PositionalEncoding    → adds position signal to each token
          ↓
        N × TransformerBlock  → learns relationships across tokens
          ↓
        Mean Pooling          → collapses 7 tokens into one fixed vector
          ↓
        Linear Classifier     → outputs a single logit (default probability)

    Args:
        d_model   (int): Embedding size for all tokens (default 64)
        num_heads (int): Attention heads per block (default 4)
        ff_dim    (int): Feed-forward hidden size (default 128)
        num_layers(int): Number of stacked Transformer blocks (default 2)
        dropout   (float): Dropout rate (default 0.1)
        n_static  (int): Number of static numerical features
        vocab_sizes(dict): Vocabulary sizes for categorical features
    """
    def __init__(
        self,
        d_model:    int  = 64,
        num_heads:  int  = 4,
        ff_dim:     int  = 128,
        num_layers: int  = 2,
        dropout:    float = 0.1,
        n_static:   int  = 13,
        vocab_sizes: dict = None,
    ):
        super().__init__()

        if vocab_sizes is None:
            raise ValueError("vocab_sizes must be provided. Load from processed/metadata.pkl")

        # Token embedding layer
        self.token_embedding = CreditTokenEmbedding(d_model, n_static, vocab_sizes)

        # Positional encoding (seq_len = 7: 1 static + 6 monthly)
        self.positional_encoding = PositionalEncoding(d_model, max_len=10)

        # Stack of Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Final classifier: maps pooled representation → single logit
        self.classifier = nn.Linear(d_model, 1)

    def forward(
        self,
        static_num:  torch.Tensor,
        static_cat:  torch.Tensor,
        monthly_num: torch.Tensor,
        monthly_pay: torch.Tensor,
    ):
        """
        Args:
            static_num  : (batch, n_static)  — scaled static numerical features
            static_cat  : (batch, 3)          — SEX, EDUCATION, MARRIAGE indices
            monthly_num : (batch, 6, 4)       — monthly BILL, PAY_AMT, UTIL, PAY_RATE
            monthly_pay : (batch, 6)          — monthly repayment status indices

        Returns:
            logits       : (batch,)   — raw score before sigmoid; pass to BCEWithLogitsLoss
            attention_maps: list of attention weight tensors, one per block
                           shape per map: (batch, num_heads, 7, 7)
                           → use for visualising which months influenced the prediction
        """
        # Step 1: Convert inputs to token embeddings
        x = self.token_embedding(static_num, static_cat, monthly_num, monthly_pay)

        # Step 2: Add positional encoding
        x = self.positional_encoding(x)

        # Step 3: Pass through stacked Transformer blocks
        attention_maps = []
        for block in self.blocks:
            x, attn = block(x)
            attention_maps.append(attn)

        # Step 4: Mean pooling — average across the 7 tokens
        # This gives one fixed-size vector representing the whole customer
        pooled = x.mean(dim=1)                      # (batch, d_model)

        # Step 5: Classify
        logits = self.classifier(pooled).squeeze(-1) # (batch,)

        return logits, attention_maps


# ── 6. HELPER: LOAD CONFIG FROM METADATA ─────────────────────────────────────

def get_model_config(metadata_path: str = "processed/metadata.pkl") -> dict:
    """
    Loads vocab sizes and feature counts from Ryan_Data_Cleaning.py outputs,
    and returns a config dict ready to pass into CreditTransformer.

    Usage:
        config = get_model_config()
        model  = CreditTransformer(**config)
    """
    with open(metadata_path, "rb") as f:
        meta = pickle.load(f)

    return {
        "d_model"    : 64,
        "num_heads"  : 4,
        "ff_dim"     : 128,
        "num_layers" : 2,
        "dropout"    : 0.1,
        "n_static"   : meta["n_static_num"],
        "vocab_sizes": meta["vocab_sizes"],
    }


# ── 7. QUICK SANITY CHECK (run this file directly to verify everything works) ─

if __name__ == "__main__":
    print("=" * 60)
    print("  Ryan_Attention_Module.py — Sanity Check")
    print("=" * 60)

    # Load config from preprocessing outputs
    config = get_model_config()
    print(f"\nModel config loaded from metadata.pkl:")
    for k, v in config.items():
        print(f"  {k:<12}: {v}")

    # Build the model
    model = CreditTransformer(**config)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel built successfully!")
    print(f"  Total trainable parameters: {total_params:,}")

    # Load one batch from the preprocessed training data
    from torch.utils.data import Dataset

    # Column definitions needed to load the Dataset object
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

    class CreditCardDataset(Dataset):
        def __init__(self, X, y):
            self.X = X.reset_index(drop=True)
            self.y = y.reset_index(drop=True)
            self._static_num_cols = static_num_cols + static_eng_cols + bill_neg_cols
        def __len__(self): return len(self.y)
        def __getitem__(self, idx):
            row = self.X.iloc[idx]
            static_num  = torch.tensor(row[self._static_num_cols].values.astype(float), dtype=torch.float32)
            static_cat  = torch.tensor([int(row["SEX"]), int(row["EDUCATION"]), int(row["MARRIAGE"])], dtype=torch.long)
            monthly_num, monthly_pay = [], []
            for b, a, u, pr, p in zip(bill_cols, amt_cols, monthly_util_cols, monthly_pay_rate_cols, pay_cols):
                monthly_num.append([float(row[b]), float(row[a]), float(row[u]), float(row[pr])])
                monthly_pay.append(int(row[p]))
            monthly_num = torch.tensor(monthly_num, dtype=torch.float32)
            monthly_pay = torch.tensor(monthly_pay, dtype=torch.long)
            label = torch.tensor(float(self.y.iloc[idx]), dtype=torch.float32)
            return {"static_num": static_num, "static_cat": static_cat,
                    "monthly_num": monthly_num, "monthly_pay": monthly_pay, "label": label}

    train_ds = torch.load("processed/train_dataset.pt", weights_only=False)
    loader   = DataLoader(train_ds, batch_size=32, shuffle=False)
    batch    = next(iter(loader))

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits, attention_maps = model(
            batch["static_num"],
            batch["static_cat"],
            batch["monthly_num"],
            batch["monthly_pay"],
        )

    print(f"\nForward pass successful!")
    print(f"  Input batch size        : {batch['static_num'].shape[0]}")
    print(f"  Output logits shape     : {list(logits.shape)}")
    print(f"  Number of attention maps: {len(attention_maps)}")
    print(f"  Attention map shape     : {list(attention_maps[0].shape)}")
    print(f"  (batch, heads, seq, seq): ({attention_maps[0].shape[0]}, "
          f"{attention_maps[0].shape[1]}, "
          f"{attention_maps[0].shape[2]}, "
          f"{attention_maps[0].shape[3]})")

    # Show what the attention map looks like for the first sample
    token_labels = ["Static", "Month6", "Month5", "Month4", "Month3", "Month2", "Month1"]
    attn_sample  = attention_maps[-1][0].mean(dim=0)  # average across heads, first sample
    print(f"\nAttention weights (last block, first sample, averaged across heads):")
    print(f"  Rows = query token, Columns = key token being attended to")
    print(f"  {'':>8}", end="")
    for label in token_labels:
        print(f"  {label:>7}", end="")
    print()
    for i, row_label in enumerate(token_labels):
        print(f"  {row_label:>8}", end="")
        for j in range(len(token_labels)):
            print(f"  {attn_sample[i, j].item():>7.3f}", end="")
        print()

    print("\n" + "=" * 60)
    print("  All checks passed! Ready for model training.")
    print("=" * 60)
    print("\nTo use this module in your training notebook:")
    print("  from Ryan_Attention_Module import CreditTransformer, get_model_config")
    print("  config = get_model_config()")
    print("  model  = CreditTransformer(**config)")
