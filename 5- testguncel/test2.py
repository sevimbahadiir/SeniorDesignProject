"""
test2.py
--------
Step 3b: Digital Twin Trajectory Validation

Tests whether the Digital Twin can reliably simulate multi-step rollouts.
A good DT should maintain MAE < 0.08 over 8 consecutive steps.

Method:
    1. Load DT model + scaler
    2. From validation set, pick N starting states
    3. For each: run 8-step autoregressive rollout (DT output → next input)
    4. Compare predicted trajectory vs actual trajectory
    5. Report per-step MAE, cumulative drift, and per-service breakdown

Pass criterion:
    - Step-8 MAE < 0.08  (acceptable)
    - Step-8 MAE < 0.05  (good)
    - Step-8 MAE < 0.03  (excellent)

Input:  digital_twin_best.pth, dt_scaler.pkl, rl_dataset_sar_normalized.csv
Output: trajectory_results.csv, trajectory_plot.png
"""

import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Config ───────────────────────────────────────────────────────────────────
ROLLOUT_STEPS  = 8          # how many steps to simulate forward
N_TRAJECTORIES = 200        # number of starting states to test
RANDOM_SEED    = 42
VAL_SPLIT      = 0.2        # must match dt2.py

SERVICES = ['cart', 'catalogue', 'payment', 'shipping', 'ratings', 'user']
METRICS  = ['num_pods', 'cpu_usage', 'mem_usage', 'request_rate', 'latency']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ─── Model Definition (must match dt2.py exactly) ────────────────────────────
class DigitalTwin(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = h
        layers += [nn.Linear(prev_dim, output_dim), nn.Sigmoid()]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ─── Step 1: Load Data ────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 1: Loading Data & Model")
print("=" * 65)

df = pd.read_csv('rl_dataset_sar_normalized.csv')
print(f"  Dataset: {len(df)} rows")

state_cols      = [f'{svc}_{m}' for svc in SERVICES for m in METRICS
                   if f'{svc}_{m}' in df.columns]
action_cols     = [f'{svc}_action' for svc in SERVICES]
next_state_cols = [f'next_{c}' for c in state_cols]

input_dim  = len(state_cols) + len(action_cols)
output_dim = len(next_state_cols)

# Time-ordered val split (same as dt2.py)
split_idx = int(len(df) * (1 - VAL_SPLIT))
df_val    = df.iloc[split_idx:].reset_index(drop=True)
print(f"  Val set : {len(df_val)} rows (last {VAL_SPLIT*100:.0f}%)")

# ─── Step 2: Load Model ───────────────────────────────────────────────────────
model = DigitalTwin(input_dim, output_dim, [256, 256, 128]).to(device)
model.load_state_dict(torch.load('digital_twin_best.pth', map_location=device))
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"  Model   : {input_dim} -> 256 -> 256 -> 128 -> {output_dim}")
print(f"  Params  : {total_params:,}")
print(f"  Weights : digital_twin_best.pth OK")

with open('dt_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print(f"  Scaler  : dt_scaler.pkl OK")

# ─── Step 3: Trajectory Rollout ───────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2: Running Trajectory Rollouts")
print("=" * 65)

np.random.seed(RANDOM_SEED)

max_start = len(df_val) - ROLLOUT_STEPS - 1
if max_start < N_TRAJECTORIES:
    N_TRAJECTORIES = max_start
    print(f"  Warning: reduced to {N_TRAJECTORIES} trajectories (val set size)")

start_indices = np.random.choice(max_start, size=N_TRAJECTORIES, replace=False)
print(f"  Trajectories  : {N_TRAJECTORIES}")
print(f"  Rollout steps : {ROLLOUT_STEPS}")
print()

pred_trajectories = np.zeros((N_TRAJECTORIES, ROLLOUT_STEPS, output_dim))
true_trajectories = np.zeros((N_TRAJECTORIES, ROLLOUT_STEPS, output_dim))

for traj_idx, start in enumerate(start_indices):
    current_state = df_val.iloc[start][state_cols].values.astype(np.float32)

    for step in range(ROLLOUT_STEPS):
        row = df_val.iloc[start + step]

        true_next = row[next_state_cols].values.astype(np.float32)
        true_trajectories[traj_idx, step] = true_next

        action = row[action_cols].values.astype(np.float32)

        x = np.concatenate([current_state, action])
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_next = model(x_tensor).cpu().numpy().squeeze()

        pred_trajectories[traj_idx, step] = pred_next
        current_state = pred_next  # autoregressive

    if (traj_idx + 1) % 50 == 0:
        print(f"  Processed {traj_idx + 1}/{N_TRAJECTORIES} trajectories...")

print(f"\n  Rollout complete.")

# ─── Step 4: Metrics ──────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3: Per-Step Metrics")
print("=" * 65)

step_mae  = []
step_rmse = []
step_r2   = []

print(f"  {'Step':>4}  {'MAE':>8}  {'RMSE':>8}  {'R2':>8}  {'Status'}")
print(f"  {'-'*50}")

for step in range(ROLLOUT_STEPS):
    pred = pred_trajectories[:, step, :]
    true = true_trajectories[:, step, :]

    mae  = np.mean(np.abs(pred - true))
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    r2   = 1 - np.sum((pred - true) ** 2) / np.sum((true - true.mean()) ** 2)

    step_mae.append(mae)
    step_rmse.append(rmse)
    step_r2.append(r2)

    if mae < 0.03:
        status = "excellent ++"
    elif mae < 0.05:
        status = "good +"
    elif mae < 0.08:
        status = "acceptable +"
    else:
        status = "degraded X"

    print(f"  Step {step+1:>1}  {mae:>8.5f}  {rmse:>8.5f}  {r2:>8.4f}  {status}")

final_mae = step_mae[-1]
print(f"\n  Final step MAE: {final_mae:.5f}")
if final_mae < 0.03:
    print("  -> Excellent trajectory accuracy. DT is reliable for MARL training.")
elif final_mae < 0.05:
    print("  -> Good trajectory accuracy. DT suitable for MARL training.")
elif final_mae < 0.08:
    print("  -> Acceptable. DT usable; some drift at longer horizons.")
else:
    print("  -> Warning: significant drift. Consider shorter rollouts or DT retraining.")

# ─── Step 5: Per-Service Breakdown (final step) ───────────────────────────────
print("\n" + "=" * 65)
print("STEP 4: Per-Service Error at Step 8")
print("=" * 65)
print(f"  {'Service':<12} {'num_pods':>10} {'cpu':>10} "
      f"{'mem':>10} {'rps':>10} {'latency':>10}  {'avg':>8}")
print(f"  {'-'*72}")

for i, svc in enumerate(SERVICES):
    base   = i * len(METRICS)
    pred_s = pred_trajectories[:, -1, base:base+len(METRICS)]
    true_s = true_trajectories[:, -1, base:base+len(METRICS)]
    errs   = np.mean(np.abs(pred_s - true_s), axis=0)
    avg    = errs.mean()
    print(f"  {svc:<12} " + " ".join(f"{e:>10.5f}" for e in errs) + f"  {avg:>8.5f}")

# ─── Step 6: Save Results CSV ─────────────────────────────────────────────────
results_df = pd.DataFrame({
    'step' : list(range(1, ROLLOUT_STEPS + 1)),
    'mae'  : step_mae,
    'rmse' : step_rmse,
    'r2'   : step_r2,
})
results_df.to_csv('trajectory_results.csv', index=False)
print(f"\n  Saved: trajectory_results.csv")

# ─── Step 7: Plots ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Digital Twin -- Trajectory Validation', fontsize=14, fontweight='bold')

steps = list(range(1, ROLLOUT_STEPS + 1))

# Plot 1: MAE per step
axes[0, 0].plot(steps, step_mae, 'o-', color='steelblue', linewidth=2, markersize=6)
axes[0, 0].axhline(0.08, color='red',    linestyle='--', alpha=0.7, label='Acceptable (0.08)')
axes[0, 0].axhline(0.05, color='orange', linestyle='--', alpha=0.7, label='Good (0.05)')
axes[0, 0].axhline(0.03, color='green',  linestyle='--', alpha=0.7, label='Excellent (0.03)')
axes[0, 0].set_xlabel('Rollout Step')
axes[0, 0].set_ylabel('MAE')
axes[0, 0].set_title('MAE per Rollout Step')
axes[0, 0].set_xticks(steps)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: R2 per step
axes[0, 1].plot(steps, step_r2, 'o-', color='darkorange', linewidth=2, markersize=6)
axes[0, 1].axhline(0.9, color='green', linestyle='--', alpha=0.7, label='R2=0.9 threshold')
axes[0, 1].set_xlabel('Rollout Step')
axes[0, 1].set_ylabel('R2')
axes[0, 1].set_title('R2 per Rollout Step')
axes[0, 1].set_xticks(steps)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Sample trajectory shipping latency
svc_idx    = SERVICES.index('shipping')
lat_idx    = svc_idx * len(METRICS) + METRICS.index('latency')
sample_ids = np.random.choice(N_TRAJECTORIES, 5, replace=False)

for tid in sample_ids:
    axes[1, 0].plot(steps, true_trajectories[tid, :, lat_idx],
                    'o--', alpha=0.5, linewidth=1)
axes[1, 0].set_prop_cycle(None)
for tid in sample_ids:
    axes[1, 0].plot(steps, pred_trajectories[tid, :, lat_idx],
                    's-', alpha=0.8, linewidth=1.5)
axes[1, 0].set_xlabel('Rollout Step')
axes[1, 0].set_ylabel('Normalized Latency')
axes[1, 0].set_title('Shipping Latency: True (dashed) vs Pred (solid)')
axes[1, 0].set_xticks(steps)
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Per-service MAE at final step
svc_mae_final = []
for i, svc in enumerate(SERVICES):
    base = i * len(METRICS)
    mae_s = np.mean(np.abs(
        pred_trajectories[:, -1, base:base+len(METRICS)] -
        true_trajectories[:, -1, base:base+len(METRICS)]
    ))
    svc_mae_final.append(mae_s)

colors = ['steelblue' if m < 0.05 else 'darkorange' if m < 0.08 else 'red'
          for m in svc_mae_final]
axes[1, 1].bar(SERVICES, svc_mae_final, color=colors)
axes[1, 1].axhline(0.08, color='red',    linestyle='--', alpha=0.7, label='Acceptable (0.08)')
axes[1, 1].axhline(0.05, color='orange', linestyle='--', alpha=0.7, label='Good (0.05)')
axes[1, 1].set_xlabel('Service')
axes[1, 1].set_ylabel('MAE (Step 8)')
axes[1, 1].set_title('Per-Service MAE at Final Rollout Step')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('trajectory_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: trajectory_plot.png")

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("TRAJECTORY VALIDATION SUMMARY")
print("=" * 65)
print(f"  Trajectories tested : {N_TRAJECTORIES}")
print(f"  Rollout horizon     : {ROLLOUT_STEPS} steps")
print()
print(f"  Step 1 MAE : {step_mae[0]:.5f}")
print(f"  Step 4 MAE : {step_mae[3]:.5f}")
print(f"  Step 8 MAE : {step_mae[7]:.5f}  <- key criterion")
print()
print(f"  Step 1 R2  : {step_r2[0]:.4f}")
print(f"  Step 4 R2  : {step_r2[3]:.4f}")
print(f"  Step 8 R2  : {step_r2[7]:.4f}")
print()

if final_mae < 0.08:
    print("  PASS -- DT trajectory quality sufficient for MARL training.")
    print("  -> Proceed to: train_ippo_15k.py")
else:
    print("  FAIL -- DT drift too high for reliable MARL training.")
    print("  -> Consider: shorter rollout horizon or DT architecture changes.")

print()
print(f"  Files saved:")
print(f"    trajectory_results.csv  -- per-step metrics")
print(f"    trajectory_plot.png     -- 4-panel visualization")
print("=" * 65)
print("Done!")
