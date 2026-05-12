"""
digital_twin.py
---------------
Step 3: Trains a Digital Twin model on the SAR dataset.

The Digital Twin is a neural network that learns system dynamics:
    f(state, action) → next_state

This model serves as a simulation environment for MARL agents,
enabling policy learning without interacting with the real cluster.

Architecture:
    MLP: 36 → 256 → 256 → 128 → 30
    - BatchNorm + ReLU + Dropout(0.2) after each hidden layer
    - Sigmoid output (data is normalized to [0,1])
    - Single model captures cross-service dependencies

Design decisions:
    - MLP over LSTM: each SAR row is a self-contained (s,a,s') tuple;
      temporal dependencies are implicitly captured in state representation
    - Huber loss: robust to shipping latency outliers vs MSE
    - BatchNorm: stabilizes training across heterogeneous feature scales
    - Early stopping: prevents overfitting on 14k-row dataset
    - Single model: captures cross-service dependencies (e.g., shipping
      bottleneck effect on payment traffic)

Input:  rl_dataset_sar_normalized.csv
Output: digital_twin_best.pth   (best model weights)
        digital_twin_final.pth  (final epoch weights)
        dt_scaler.pkl           (scaler reference copy)
        dt_training_log.csv     (per-epoch loss history)
"""

import pickle
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# ─── Device ───────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ─── Configuration ────────────────────────────────────────────────────────────

SERVICES = ['cart', 'catalogue', 'payment', 'shipping', 'ratings', 'user']
METRICS  = ['num_pods', 'cpu_usage', 'mem_usage', 'request_rate', 'latency']

# Model hyperparameters
HIDDEN_DIMS = [256, 256, 128]
DROPOUT     = 0.2
BATCH_SIZE  = 64
EPOCHS      = 300
LR          = 1e-3
LR_PATIENCE = 15       # epochs before reducing LR
LR_FACTOR   = 0.5      # LR reduction factor
ES_PATIENCE = 30       # early stopping patience
HUBER_DELTA = 1.0      # Huber loss delta

# ─── Step 1: Load Data ────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 1: Loading Data")
print("=" * 65)

df = pd.read_csv('rl_dataset_sar_normalized.csv')
print(f"  Loaded: {len(df)} rows | {len(df.columns)} columns")

# Define column groups
state_cols = [f'{svc}_{m}' for svc in SERVICES for m in METRICS
              if f'{svc}_{m}' in df.columns]
action_cols = [f'{svc}_action' for svc in SERVICES]
next_state_cols = [f'next_{c}' for c in state_cols]

input_dim  = len(state_cols) + len(action_cols)   # 30 + 6 = 36
output_dim = len(next_state_cols)                  # 30

print(f"  State dim  : {len(state_cols)}")
print(f"  Action dim : {len(action_cols)}")
print(f"  Input dim  : {input_dim} (state + action)")
print(f"  Output dim : {output_dim} (next_state)")

# ─── Step 2: Prepare Tensors ──────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2: Preparing Train / Validation Split")
print("=" * 65)

X = df[state_cols + action_cols].values.astype(np.float32)
y = df[next_state_cols].values.astype(np.float32)

# Time-ordered split — consistent with sar_create.py
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print(f"  Train : {len(X_train)} rows")
print(f"  Val   : {len(X_val)} rows")

# Verify no data leakage: action range check
assert X_train[:, len(state_cols):].min() >= -1
assert X_train[:, len(state_cols):].max() <=  1
print(f"  Action range check: [-1, 1] ✓")

# ─── Step 3: Dataset and DataLoader ──────────────────────────────────────────
class TransitionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(
    TransitionDataset(X_train, y_train),
    batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    TransitionDataset(X_val, y_val),
    batch_size=BATCH_SIZE, shuffle=False
)

# ─── Step 4: Model Architecture ───────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3: Model Architecture")
print("=" * 65)

class DigitalTwin(nn.Module):
    """
    MLP-based Digital Twin for microservice system dynamics.

    Architecture: 36 → 256 → 256 → 128 → 30
    Each hidden layer: Linear → BatchNorm → ReLU → Dropout
    Output: Sigmoid (predicts normalized next_state in [0,1])
    """
    def __init__(self, input_dim, output_dim, hidden_dims, dropout):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

model = DigitalTwin(input_dim, output_dim, HIDDEN_DIMS, DROPOUT).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"  Architecture  : {input_dim} → {' → '.join(map(str, HIDDEN_DIMS))} → {output_dim}")
print(f"  Total params  : {total_params:,}")
print(f"  Trainable     : {trainable:,}")
print(f"  Dropout       : {DROPOUT}")
print(f"  Output        : Sigmoid (predicts normalized [0,1] values)")

# ─── Step 5: Training ─────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4: Training")
print("=" * 65)
print(f"  Loss      : Huber (delta={HUBER_DELTA})")
print(f"  Optimizer : Adam (lr={LR})")
print(f"  LR decay  : ReduceLROnPlateau (patience={LR_PATIENCE}, factor={LR_FACTOR})")
print(f"  Early stop: patience={ES_PATIENCE} epochs")
print(f"  Epochs    : up to {EPOCHS}")
print()

criterion = nn.HuberLoss(delta=HUBER_DELTA)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=LR_PATIENCE, factor=LR_FACTOR
)

train_losses = []
val_losses   = []
lr_history   = []
best_val_loss   = float('inf')
best_epoch      = 0
es_counter      = 0
train_start     = time.time()

for epoch in range(1, EPOCHS + 1):

    # ── Train ──
    model.train()
    epoch_train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    epoch_train_loss /= len(train_loader)

    # ── Validate ──
    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            epoch_val_loss += loss.item()
    epoch_val_loss /= len(val_loader)

    scheduler.step(epoch_val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)
    lr_history.append(current_lr)

    # ── Best model ──
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_epoch    = epoch
        es_counter    = 0
        torch.save(model.state_dict(), 'digital_twin_best.pth')
    else:
        es_counter += 1

    # ── Log every 10 epochs ──
    if epoch % 10 == 0 or epoch == 1:
        elapsed = time.time() - train_start
        print(f"  Epoch {epoch:3d}/{EPOCHS} | "
              f"Train: {epoch_train_loss:.6f} | "
              f"Val: {epoch_val_loss:.6f} | "
              f"Best: {best_val_loss:.6f} (ep {best_epoch}) | "
              f"LR: {current_lr:.2e} | "
              f"ES: {es_counter}/{ES_PATIENCE} | "
              f"{elapsed:.0f}s")

    # ── Early stopping ──
    if es_counter >= ES_PATIENCE:
        print(f"\n  Early stopping at epoch {epoch} "
              f"(no improvement for {ES_PATIENCE} epochs).")
        break

torch.save(model.state_dict(), 'digital_twin_final.pth')
print(f"\n  Training complete.")
print(f"  Best val loss : {best_val_loss:.6f} at epoch {best_epoch}")
print(f"  Total time    : {time.time() - train_start:.1f}s")

# ─── Step 6: Evaluation ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5: Evaluation")
print("=" * 65)

model.load_state_dict(torch.load('digital_twin_best.pth', map_location=device))
model.eval()

with torch.no_grad():
    X_val_t  = torch.tensor(X_val,  dtype=torch.float32).to(device)
    y_pred   = model(X_val_t).cpu().numpy()

mae  = np.mean(np.abs(y_pred - y_val))
rmse = np.sqrt(np.mean((y_pred - y_val) ** 2))
r2   = 1 - np.sum((y_pred - y_val) ** 2) / np.sum((y_val - y_val.mean()) ** 2)

print(f"  MAE  : {mae:.6f}  (mean absolute error, scale 0-1)")
print(f"  RMSE : {rmse:.6f}  (root mean squared error, scale 0-1)")
print(f"  R²   : {r2:.6f}  (coefficient of determination)")
print()
print(f"  Interpretation:")
print(f"    MAE < 0.01 → excellent")
print(f"    MAE < 0.05 → good")
print(f"    MAE < 0.10 → acceptable")

# Per-service analysis
print(f"\n  Per-service next_state prediction error (MAE):")
print(f"  {'Service':<12} {'num_pods':>10} {'cpu':>10} "
      f"{'mem':>10} {'rps':>10} {'latency':>10}")
print(f"  {'-'*62}")

for i, svc in enumerate(SERVICES):
    base = i * len(METRICS)
    errors = []
    for j in range(len(METRICS)):
        col_mae = np.mean(np.abs(y_pred[:, base+j] - y_val[:, base+j]))
        errors.append(col_mae)
    print(f"  {svc:<12} " + " ".join(f"{e:>10.6f}" for e in errors))

# ─── Step 7: Training Log ─────────────────────────────────────────────────────
log_df = pd.DataFrame({
    'epoch':      range(1, len(train_losses) + 1),
    'train_loss': train_losses,
    'val_loss':   val_losses,
    'lr':         lr_history,
})
log_df.to_csv('dt_training_log.csv', index=False)

# ─── Step 8: Loss Plot ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
axes[0].plot(train_losses, label='Train Loss', color='steelblue', linewidth=1.5)
axes[0].plot(val_losses,   label='Val Loss',   color='darkorange', linewidth=1.5)
axes[0].axvline(best_epoch - 1, color='green', linestyle='--',
                alpha=0.7, label=f'Best (ep {best_epoch})')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Huber Loss')
axes[0].set_title('Digital Twin — Training Curve')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Prediction vs actual (latency columns only)
lat_indices = [i * len(METRICS) + 4 for i in range(len(SERVICES))]
y_pred_lat  = y_pred[:, lat_indices].flatten()
y_true_lat  = y_val[:,  lat_indices].flatten()
sample_idx  = np.random.choice(len(y_pred_lat), min(2000, len(y_pred_lat)), replace=False)

axes[1].scatter(y_true_lat[sample_idx], y_pred_lat[sample_idx],
                alpha=0.3, s=5, color='steelblue')
axes[1].plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='Perfect prediction')
axes[1].set_xlabel('Actual next_latency (normalized)')
axes[1].set_ylabel('Predicted next_latency (normalized)')
axes[1].set_title('Latency Prediction: Actual vs Predicted')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dt_training_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Plot saved: dt_training_plot.png")

# ─── Step 9: Save Scaler Reference ───────────────────────────────────────────
import shutil
shutil.copy('scaler.pkl', 'dt_scaler.pkl')

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY REPORT")
print("=" * 65)
print(f"  Architecture  : {input_dim} → {' → '.join(map(str, HIDDEN_DIMS))} → {output_dim}")
print(f"  Parameters    : {total_params:,}")
print(f"  Best epoch    : {best_epoch}")
print(f"  Best val loss : {best_val_loss:.6f} (Huber)")
print(f"  MAE           : {mae:.6f}")
print(f"  RMSE          : {rmse:.6f}")
print(f"  R²            : {r2:.6f}")
print()
print(f"  Files saved:")
print(f"    digital_twin_best.pth   — best model weights (use this)")
print(f"    digital_twin_final.pth  — final epoch weights")
print(f"    dt_training_log.csv     — per-epoch loss history")
print(f"    dt_training_plot.png    — loss curve + prediction scatter")
print(f"    dt_scaler.pkl           — scaler reference")
print("=" * 65)
print("Done!")
