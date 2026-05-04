"""
dt_trajectory_test.py
---------------------
Trajectory-based validation of the Digital Twin model.

Point-wise validation (MAE, R²) evaluates each transition independently.
Trajectory validation simulates how DT is actually used in MARL:
    the model's own predictions feed into the next step.

This tests whether prediction errors accumulate (compound) over time,
which is the critical failure mode for model-based RL.

Method:
    1. Sample real trajectories from the SAR dataset
       (consecutive rows within the same run)
    2. Give DT the real starting state (t=0)
    3. Feed DT's predicted next_state back as input for t+1, t+2, ...
    4. Compare DT's trajectory against the real trajectory at each step

Metrics:
    - MAE per step (does error grow over time?)
    - Per-service latency comparison (which services diverge fastest?)
    - Visual trajectory plots (real vs predicted)

If MAE grows slowly and stays below 0.10 at step 20,
the DT is considered reliable for MARL training.
"""

import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Configuration ────────────────────────────────────────────────────────────

SERVICES     = ['cart', 'catalogue', 'payment', 'shipping', 'ratings', 'user']
METRICS      = ['num_pods', 'cpu_usage', 'mem_usage', 'request_rate', 'latency']
HIDDEN_DIMS  = [256, 256, 128]
DROPOUT      = 0.2
TRAJ_LEN     = 20    # steps per trajectory
N_TRAJ       = 5     # trajectories per load level
TEST_LOADS   = ['40user', '200user', '400user']  # low, mid, high load

# ─── Model Definition (must match digital_twin.py) ────────────────────────────
class DigitalTwin(nn.Module):
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
        layers += [nn.Linear(prev_dim, output_dim), nn.Sigmoid()]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ─── Step 1: Load Data and Model ──────────────────────────────────────────────
print("=" * 65)
print("STEP 1: Loading Data and Model")
print("=" * 65)

df_norm = pd.read_csv('rl_dataset_sar_normalized.csv')
df_raw  = pd.read_csv('rl_dataset_sar_raw.csv')

state_cols      = [f'{s}_{m}' for s in SERVICES for m in METRICS
                   if f'{s}_{m}' in df_norm.columns]
action_cols     = [f'{s}_action' for s in SERVICES]
next_state_cols = [f'next_{c}' for c in state_cols]

input_dim  = len(state_cols) + len(action_cols)
output_dim = len(next_state_cols)

device = torch.device('cpu')
model  = DigitalTwin(input_dim, output_dim, HIDDEN_DIMS, DROPOUT).to(device)
model.load_state_dict(torch.load('digital_twin_best.pth', map_location=device))
model.eval()

print(f"  Model loaded   : digital_twin_best.pth")
print(f"  Dataset rows   : {len(df_norm)}")
print(f"  State dims     : {len(state_cols)}")
print(f"  Trajectory len : {TRAJ_LEN} steps")
print(f"  Trajectories   : {N_TRAJ} per load level")
print(f"  Load levels    : {TEST_LOADS}")

# ─── Step 2: Trajectory Simulation ───────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2: Running Trajectory Simulations")
print("=" * 65)

def simulate_trajectory(start_idx, df_norm, df_raw, length):
    """
    Simulate a trajectory of given length starting from start_idx.

    Returns:
        real_traj  : (length, 30) — actual next_states from dataset
        dt_traj    : (length, 30) — DT predicted next_states
        actions    : (length, 6)  — actions taken
    """
    real_traj = []
    dt_traj   = []
    actions   = []

    # Starting state from real data
    current_state = df_norm.iloc[start_idx][state_cols].values.astype(np.float32)

    for step in range(length):
        idx    = start_idx + step
        if idx >= len(df_norm):
            break

        action     = df_norm.iloc[idx][action_cols].values.astype(np.float32)
        real_next  = df_norm.iloc[idx][next_state_cols].values.astype(np.float32)

        # DT prediction using current (possibly predicted) state
        state_action = np.concatenate([current_state, action])
        with torch.no_grad():
            inp      = torch.tensor(state_action, dtype=torch.float32).unsqueeze(0)
            dt_next  = model(inp).squeeze(0).numpy()

        real_traj.append(real_next)
        dt_traj.append(dt_next)
        actions.append(action)

        # Feed DT's prediction as next state (this is the key test)
        current_state = dt_next

    return np.array(real_traj), np.array(dt_traj), np.array(actions)

# Run trajectories for each load level
all_results = {}
np.random.seed(42)

for load in TEST_LOADS:
    load_df   = df_norm[df_norm['user_load'] == load].reset_index(drop=True)
    load_raw  = df_raw[df_raw['user_load'] == load].reset_index(drop=True)

    if len(load_df) < TRAJ_LEN + N_TRAJ:
        print(f"  {load}: not enough rows, skipping.")
        continue

    # Sample N_TRAJ random starting points
    # Ensure enough room for full trajectory
    max_start   = len(load_df) - TRAJ_LEN - 1
    start_idxs  = np.random.choice(max_start, N_TRAJ, replace=False)

    load_results = []
    for i, start in enumerate(start_idxs):
        # Map back to full df index
        full_start = df_norm[df_norm['user_load'] == load].index[start]
        real, pred, acts = simulate_trajectory(full_start, df_norm, df_raw, TRAJ_LEN)
        load_results.append({'real': real, 'pred': pred, 'actions': acts,
                             'start_idx': start})

    all_results[load] = load_results
    print(f"  {load:<12}: {N_TRAJ} trajectories × {TRAJ_LEN} steps simulated ✓")

# ─── Step 3: Compute Metrics ──────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3: Computing Metrics")
print("=" * 65)

# MAE per step across all trajectories and load levels
all_real = []
all_pred = []
for load, results in all_results.items():
    for r in results:
        all_real.append(r['real'])
        all_pred.append(r['pred'])

all_real = np.array(all_real)  # (N_total_traj, TRAJ_LEN, 30)
all_pred = np.array(all_pred)

# MAE at each step (averaged over all trajectories and features)
mae_per_step = np.mean(np.abs(all_real - all_pred), axis=(0, 2))

print(f"\n  MAE per step (averaged over all trajectories and features):")
print(f"  {'Step':>6} {'MAE':>10} {'Assessment'}")
print(f"  {'-'*35}")
for step in range(TRAJ_LEN):
    mae   = mae_per_step[step]
    assess = "excellent" if mae < 0.05 else ("good" if mae < 0.08 else
             ("acceptable" if mae < 0.10 else "HIGH"))
    print(f"  {step+1:>6} {mae:>10.6f} {assess}")

# Per-service latency MAE at step 1, 5, 10, 20
print(f"\n  Per-service latency MAE at key steps:")
print(f"  {'Service':<12} {'Step 1':>10} {'Step 5':>10} "
      f"{'Step 10':>10} {'Step 20':>10}")
print(f"  {'-'*55}")

lat_indices = {svc: i * len(METRICS) + 4 for i, svc in enumerate(SERVICES)}
for svc, lat_idx in lat_indices.items():
    maes = []
    for step in [0, 4, 9, 19]:
        if step < TRAJ_LEN:
            mae = np.mean(np.abs(all_real[:, step, lat_idx] -
                                 all_pred[:, step, lat_idx]))
            maes.append(mae)
        else:
            maes.append(float('nan'))
    print(f"  {svc:<12} " + " ".join(f"{m:>10.6f}" for m in maes))

# Error growth rate
step1_mae  = mae_per_step[0]
step20_mae = mae_per_step[TRAJ_LEN - 1]
growth     = (step20_mae - step1_mae) / step1_mae * 100
print(f"\n  Error growth (step 1 → step 20): {growth:+.1f}%")
print(f"  Step  1 MAE: {step1_mae:.6f}")
print(f"  Step 20 MAE: {step20_mae:.6f}")

if step20_mae < 0.10:
    verdict = "RELIABLE for MARL training ✓"
elif step20_mae < 0.15:
    verdict = "ACCEPTABLE for MARL training ⚠"
else:
    verdict = "HIGH error accumulation — use with caution ✗"
print(f"\n  Verdict: {verdict}")

# ─── Step 4: Plots ────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4: Generating Plots")
print("=" * 65)

# Plot 1: MAE per step
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, TRAJ_LEN + 1), mae_per_step,
        marker='o', color='steelblue', linewidth=2, markersize=5)
ax.axhline(0.05, color='green',  linestyle='--', alpha=0.7, label='Excellent (0.05)')
ax.axhline(0.08, color='orange', linestyle='--', alpha=0.7, label='Good (0.08)')
ax.axhline(0.10, color='red',    linestyle='--', alpha=0.7, label='Limit (0.10)')
ax.set_xlabel('Trajectory Step')
ax.set_ylabel('MAE (normalized scale)')
ax.set_title('Digital Twin — Trajectory Error Accumulation')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('dt_trajectory_mae.png', dpi=150)
plt.close()
print(f"  ✓ dt_trajectory_mae.png")

# Plot 2: Real vs DT trajectory for latency (per service, one trajectory each)
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

# Use first trajectory from 200user (mid load)
sample_load = '200user' if '200user' in all_results else list(all_results.keys())[0]
sample      = all_results[sample_load][0]
steps       = range(1, len(sample['real']) + 1)

for i, svc in enumerate(SERVICES):
    lat_idx = lat_indices[svc]
    real_lat = sample['real'][:, lat_idx]
    pred_lat = sample['pred'][:, lat_idx]

    axes[i].plot(steps, real_lat, label='Real',      color='steelblue',
                 linewidth=2, marker='o', markersize=4)
    axes[i].plot(steps, pred_lat, label='DT Predict', color='darkorange',
                 linewidth=2, linestyle='--', marker='s', markersize=4)
    axes[i].set_title(f'{svc} — latency (normalized)')
    axes[i].set_xlabel('Step')
    axes[i].set_ylabel('Normalized latency')
    axes[i].legend(fontsize=8)
    axes[i].grid(True, alpha=0.3)

    # Shade error region
    axes[i].fill_between(steps, real_lat, pred_lat,
                          alpha=0.15, color='red', label='Error')

plt.suptitle(f'Real vs DT Trajectory — {sample_load} '
             f'(starting index {sample["start_idx"]})',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('dt_trajectory_latency.png', dpi=150)
plt.close()
print(f"  ✓ dt_trajectory_latency.png")

# Plot 3: MAE per step per load level
fig, ax = plt.subplots(figsize=(10, 5))
colors = {'40user': 'steelblue', '200user': 'darkorange', '400user': 'crimson'}

for load, results in all_results.items():
    real_stack = np.array([r['real'] for r in results])
    pred_stack = np.array([r['pred'] for r in results])
    load_mae   = np.mean(np.abs(real_stack - pred_stack), axis=(0, 2))
    ax.plot(range(1, TRAJ_LEN + 1), load_mae,
            label=load, color=colors.get(load, 'gray'),
            linewidth=2, marker='o', markersize=4)

ax.axhline(0.10, color='red', linestyle='--', alpha=0.7, label='Limit (0.10)')
ax.set_xlabel('Trajectory Step')
ax.set_ylabel('MAE (normalized scale)')
ax.set_title('Trajectory Error per Load Level')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('dt_trajectory_by_load.png', dpi=150)
plt.close()
print(f"  ✓ dt_trajectory_by_load.png")

# ─── Step 5: Save Results ─────────────────────────────────────────────────────
results_df = pd.DataFrame({
    'step':     range(1, TRAJ_LEN + 1),
    'mae':      mae_per_step,
    'assessment': ['excellent' if m < 0.05 else ('good' if m < 0.08 else
                   ('acceptable' if m < 0.10 else 'HIGH'))
                   for m in mae_per_step]
})
results_df.to_csv('dt_trajectory_results.csv', index=False)
print(f"  ✓ dt_trajectory_results.csv")

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"  Trajectories tested : {len(TEST_LOADS) * N_TRAJ}")
print(f"  Steps per trajectory: {TRAJ_LEN}")
print(f"  Load levels         : {TEST_LOADS}")
print(f"  Step 1  MAE         : {step1_mae:.6f}")
print(f"  Step 20 MAE         : {step20_mae:.6f}")
print(f"  Error growth        : {growth:+.1f}%")
print(f"  Verdict             : {verdict}")
print()
print(f"  Files saved:")
print(f"    dt_trajectory_mae.png       — MAE per step")
print(f"    dt_trajectory_latency.png   — Real vs DT latency per service")
print(f"    dt_trajectory_by_load.png   — MAE per step by load level")
print(f"    dt_trajectory_results.csv   — numerical results")
print("=" * 65)
print("Done!")
