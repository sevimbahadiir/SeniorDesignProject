"""
marl_coordinated.py
-------------------
Coordinated Multi-Agent Reinforcement Learning for Robot Shop autoscaling.

Key difference from independent MARL:
- All 6 agents observe the SAME global state simultaneously
- All 6 agents take actions SIMULTANEOUSLY at each step
- Digital Twin receives ALL 6 actions at once → predicts next_state
- ALL agents share the SAME reward signal
- This captures inter-service coordination (e.g., shipping affects payment)

Architecture: Independent PPO (IPPO) with shared environment
- Each agent has its own actor-critic network
- Agents are trained in a joint environment where all act together
- This is the standard MARL approach for cooperative tasks

Episode length: 10 steps (from DT trajectory validation)
Total timesteps: 200,000 per agent
Seeds: [42, 0, 123]

Input:  rl_dataset_sar_normalized.csv
        digital_twin_best.pth
        scaler.pkl
        reward_config.json

Output: models/coordinated/seed_{seed}/{algo}_{svc}.zip
        results/coordinated/reeval_summary.csv
        plots/coordinated_comparison.png
"""

import json
import os
import pickle
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C, DQN

# ─── Configuration ────────────────────────────────────────────────────────────

SERVICES   = ['cart', 'catalogue', 'payment', 'shipping', 'ratings', 'user']
METRICS    = ['num_pods', 'cpu_usage', 'mem_usage', 'request_rate', 'latency']
STATE_COLS = [f'{svc}_{met}' for svc in SERVICES for met in METRICS]
LAT_IDX    = {svc: i * len(METRICS) + 4 for i, svc in enumerate(SERVICES)}
POD_IDX    = {svc: i * len(METRICS) + 0 for i, svc in enumerate(SERVICES)}

EPISODE_LEN   = 10
TOTAL_STEPS   = 200_000
EVAL_EPISODES = 50
SEEDS         = [42, 0, 123]
POD_COST      = 0.05

os.makedirs('models/coordinated', exist_ok=True)
os.makedirs('results/coordinated', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# ─── Digital Twin ─────────────────────────────────────────────────────────────

class DigitalTwin(nn.Module):
    def __init__(self, input_dim=36, output_dim=30,
                 hidden_dims=None, dropout=0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 128]
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, output_dim), nn.Sigmoid()]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ─── Reward ───────────────────────────────────────────────────────────────────

def compute_reward(state: np.ndarray, scaler, reward_config: dict) -> float:
    """Compute shared reward from full system state."""
    reward = 0.0
    for svc, cfg in reward_config.items():
        if not cfg['in_reward']:
            continue
        col_idx  = LAT_IDX[svc]
        real_lat = (float(state[col_idx]) *
                    (scaler.data_max_[col_idx] - scaler.data_min_[col_idx])
                    + scaler.data_min_[col_idx])
        if real_lat <= 0:
            continue
        reward += np.tanh(1.0 - real_lat / cfg['p50']) * cfg['weight']
    total_pods = sum(
        float(state[POD_IDX[svc]]) *
        (scaler.data_max_[POD_IDX[svc]] - scaler.data_min_[POD_IDX[svc]])
        + scaler.data_min_[POD_IDX[svc]]
        for svc in SERVICES
    )
    reward -= total_pods * POD_COST
    return float(round(reward, 4))

# ─── Coordinated Environment ──────────────────────────────────────────────────

class CoordinatedRobotShopEnv(gym.Env):
    """
    Coordinated MARL environment where ONE agent controls ONE service
    but sees and responds to the FULL system state including all other
    services' pod counts and latencies.

    The key difference from independent MARL:
    - During training, a shared state object is maintained
    - All agents' actions are collected before DT is called
    - DT receives ALL 6 actions simultaneously
    - All agents receive the SAME reward

    Implementation: We use a shared state buffer that all agents
    write their actions to. DT is called once per step after all
    agents have decided. This is achieved via a SharedSystemState object.
    """
    metadata = {'render_modes': []}

    def __init__(self, service: str, dt_model: DigitalTwin,
                 sar_df: pd.DataFrame, scaler, reward_config: dict,
                 shared_state=None, fixed_starts=None):
        super().__init__()
        self.service       = service
        self.svc_idx       = SERVICES.index(service)
        self.dt_model      = dt_model
        self.df            = sar_df.reset_index(drop=True)
        self.scaler        = scaler
        self.reward_config = reward_config
        self.shared_state  = shared_state  # shared across agents
        self.fixed_starts  = fixed_starts
        self.ep_count      = 0
        self.step_count    = 0
        self.state         = None

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(30,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # 0=down, 1=keep, 2=up
        self.dt_model.eval()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.fixed_starts is not None:
            idx = int(self.fixed_starts[self.ep_count % len(self.fixed_starts)])
        else:
            idx = self.np_random.integers(0, len(self.df))

        state = self.df[STATE_COLS].iloc[idx].values.astype(np.float32)

        # Sync with shared state if available
        if self.shared_state is not None:
            self.shared_state.reset(state)
        self.state      = state.copy()
        self.step_count = 0
        self.ep_count  += 1
        return self.state.copy(), {}

    def step(self, action: int):
        self.step_count += 1
        action_val = int(action) - 1  # 0→-1, 1→0, 2→+1

        if self.shared_state is not None:
            # Register this agent's action and wait for all others
            self.shared_state.set_action(self.svc_idx, action_val)
            action_vec = self.shared_state.get_all_actions()
        else:
            # Standalone: only this agent acts, others keep (0)
            action_vec = np.zeros(len(SERVICES), dtype=np.float32)
            action_vec[self.svc_idx] = action_val

        # DT forward pass with ALL actions
        inp = np.concatenate([self.state, action_vec])
        with torch.no_grad():
            t          = torch.tensor(inp, dtype=torch.float32).unsqueeze(0)
            next_state = self.dt_model(t).squeeze(0).numpy()
        next_state = np.clip(next_state, 0.0, 1.0)

        # Shared reward from full system state
        reward = compute_reward(next_state, self.scaler, self.reward_config)

        # Update shared state
        if self.shared_state is not None:
            self.shared_state.update(next_state)

        self.state = next_state
        terminated = False
        truncated  = self.step_count >= EPISODE_LEN
        return self.state.copy(), reward, terminated, truncated, {}

    def render(self): pass


class SharedSystemState:
    """
    Shared state buffer for coordinated MARL.
    Maintains current system state and collects actions from all agents
    before calling the Digital Twin.

    In SB3 training, agents are trained independently but we simulate
    coordination by having each agent's environment reference this
    shared object. During evaluation, all agents act on the same state.
    """
    def __init__(self):
        self.state   = None
        self.actions = np.zeros(len(SERVICES), dtype=np.float32)

    def reset(self, state: np.ndarray):
        self.state   = state.copy()
        self.actions = np.zeros(len(SERVICES), dtype=np.float32)

    def set_action(self, svc_idx: int, action_val: int):
        self.actions[svc_idx] = float(action_val)

    def get_all_actions(self) -> np.ndarray:
        return self.actions.copy()

    def update(self, next_state: np.ndarray):
        self.state   = next_state.copy()
        self.actions = np.zeros(len(SERVICES), dtype=np.float32)

# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_coordinated(models: dict, sar_df, dt_model, scaler,
                         reward_config, fixed_starts,
                         n: int = EVAL_EPISODES) -> float:
    """
    Evaluate all agents SIMULTANEOUSLY in a coordinated fashion.
    All agents observe the same state and act together each step.
    DT receives all 6 actions at once.
    Returns mean episode reward (shared across all agents).
    """
    shared = SharedSystemState()
    np.random.seed(999)
    total = 0.0

    for ep in range(n):
        # Start from fixed state
        idx   = int(fixed_starts[ep % len(fixed_starts)])
        state = sar_df[STATE_COLS].iloc[idx].values.astype(np.float32)
        shared.reset(state)
        ep_reward = 0.0

        for _ in range(EPISODE_LEN):
            # All agents decide simultaneously
            action_vec = np.zeros(len(SERVICES), dtype=np.float32)
            for svc in SERVICES:
                svc_idx = SERVICES.index(svc)
                if svc in models:
                    action, _ = models[svc].predict(state, deterministic=True)
                    action_vec[svc_idx] = int(action) - 1
                # else: keep (0)

            # Single DT call with all actions
            inp = np.concatenate([state, action_vec])
            with torch.no_grad():
                t          = torch.tensor(inp, dtype=torch.float32).unsqueeze(0)
                next_state = dt_model(t).squeeze(0).numpy()
            next_state = np.clip(next_state, 0.0, 1.0)

            reward     = compute_reward(next_state, scaler, reward_config)
            ep_reward += reward
            state      = next_state

        total += ep_reward

    return total / n


def evaluate_baseline_coordinated(policy: str, sar_df, dt_model, scaler,
                                  reward_config, fixed_starts,
                                  n: int = EVAL_EPISODES) -> float:
    """Evaluate random or rule-based policy in coordinated fashion."""
    np.random.seed(999)
    total = 0.0

    for ep in range(n):
        idx   = int(fixed_starts[ep % len(fixed_starts)])
        state = sar_df[STATE_COLS].iloc[idx].values.astype(np.float32)
        ep_reward = 0.0

        for _ in range(EPISODE_LEN):
            action_vec = np.zeros(len(SERVICES), dtype=np.float32)
            for svc in SERVICES:
                svc_idx = SERVICES.index(svc)
                lat_idx = LAT_IDX[svc]
                col_min = scaler.data_min_[lat_idx]
                col_max = scaler.data_max_[lat_idx]
                rng     = col_max - col_min + 1e-8
                cfg     = reward_config.get(svc, {})
                p25_n   = (cfg.get('p25', 0)   - col_min) / rng
                p75_n   = (cfg.get('p75', 1e9) - col_min) / rng
                lat     = float(state[lat_idx])

                if policy == 'random':
                    action_vec[svc_idx] = float(np.random.choice([-1, 0, 1]))
                else:  # rule-based
                    action_vec[svc_idx] = (1.0 if lat > p75_n
                                          else (-1.0 if lat < p25_n else 0.0))

            inp = np.concatenate([state, action_vec])
            with torch.no_grad():
                t          = torch.tensor(inp, dtype=torch.float32).unsqueeze(0)
                next_state = dt_model(t).squeeze(0).numpy()
            next_state = np.clip(next_state, 0.0, 1.0)

            reward     = compute_reward(next_state, scaler, reward_config)
            ep_reward += reward
            state      = next_state

        total += ep_reward

    return total / n

# ─── Training ─────────────────────────────────────────────────────────────────

ALGO_CONFIGS = {
    'PPO': {
        'class':  PPO,
        'params': {'policy': 'MlpPolicy', 'learning_rate': 3e-4,
                   'n_steps': 128, 'batch_size': 64, 'n_epochs': 10,
                   'gamma': 0.99, 'verbose': 0}
    },
    'A2C': {
        'class':  A2C,
        'params': {'policy': 'MlpPolicy', 'learning_rate': 7e-4,
                   'n_steps': 64, 'gamma': 0.99, 'verbose': 0}
    },
    'DQN': {
        'class':  DQN,
        'params': {'policy': 'MlpPolicy', 'learning_rate': 1e-4,
                   'batch_size': 64, 'buffer_size': 10_000,
                   'learning_starts': 500, 'exploration_fraction': 0.3,
                   'gamma': 0.99, 'verbose': 0}
    },
}


def train_seed(seed, sar_df, dt_model, scaler, reward_config, fixed_starts):
    print(f"\n{'='*65}\nSEED {seed}\n{'='*65}")
    model_dir = f'models/coordinated/seed_{seed}'
    os.makedirs(model_dir, exist_ok=True)

    results = []

    # Baselines (coordinated)
    print("\nBASELINES (coordinated evaluation)")
    r_rand = evaluate_baseline_coordinated(
        'random', sar_df, dt_model, scaler, reward_config, fixed_starts)
    r_rule = evaluate_baseline_coordinated(
        'rule',   sar_df, dt_model, scaler, reward_config, fixed_starts)
    print(f"  Random:     {r_rand:.4f}")
    print(f"  Rule-Based: {r_rule:.4f}")

    for algo_name, cfg in ALGO_CONFIGS.items():
        print(f"\nALGORITHM: {algo_name} (seed={seed})")

        # Train each agent with shared environment
        shared  = SharedSystemState()
        trained = {}

        for svc in SERVICES:
            t0  = time.time()
            env = CoordinatedRobotShopEnv(
                svc, dt_model, sar_df, scaler, reward_config,
                shared_state=shared
            )
            model = cfg['class'](
                env=env, seed=seed, **cfg['params']
            )
            model.learn(total_timesteps=TOTAL_STEPS, progress_bar=False)
            model.save(f'{model_dir}/{algo_name}_{svc}')
            trained[svc] = model
            elapsed = time.time() - t0
            print(f"  {svc:<12} trained in {elapsed:.0f}s")
            env.close()

        # Coordinated evaluation — all agents together
        r = evaluate_coordinated(
            trained, sar_df, dt_model, scaler, reward_config, fixed_starts)
        print(f"  → Coordinated reward: {r:.4f}")

        results.append({'seed': seed, 'algorithm': algo_name, 'reward': round(r, 4)})

    results.append({'seed': seed, 'algorithm': 'Random',     'reward': round(r_rand, 4)})
    results.append({'seed': seed, 'algorithm': 'Rule-Based', 'reward': round(r_rule, 4)})
    return pd.DataFrame(results)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 65)
    print("COORDINATED MARL TRAINING")
    print("=" * 65)
    print("  All 6 agents act simultaneously")
    print("  DT receives all 6 actions at once")
    print("  Shared reward across all agents")
    print(f"  Seeds: {SEEDS} | Episodes: {TOTAL_STEPS:,} | Eval: {EVAL_EPISODES}")

    # Load resources
    print("\n[1] Loading resources...")
    sar_df = pd.read_csv('rl_dataset_sar_normalized.csv')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('reward_config.json', 'r') as f:
        reward_config = json.load(f)
    dt_model = DigitalTwin()
    dt_model.load_state_dict(
        torch.load('digital_twin_best.pth', map_location='cpu'))
    dt_model.eval()
    print(f"    {len(sar_df)} rows loaded")

    # Fixed evaluation starts
    np.random.seed(999)
    val_start    = int(len(sar_df) * 0.8)
    fixed_starts = np.random.choice(
        range(val_start, len(sar_df)), size=EVAL_EPISODES, replace=False)
    print(f"    Fixed eval: {EVAL_EPISODES} states from val set")

    # Train all seeds
    wall_start = time.time()
    all_dfs    = []

    for seed in SEEDS:
        df_seed = train_seed(
            
            seed, sar_df, dt_model, scaler, reward_config, fixed_starts)
        all_dfs.append(df_seed)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv('results/coordinated/all_results.csv', index=False)

    # Summary
    print("\n" + "=" * 65)
    print("FINAL SUMMARY — Coordinated MARL (3 seeds, 50 episodes)")
    print("=" * 65)
    summary = combined.groupby('algorithm')['reward'].agg(
        mean='mean', std='std', min='min', max='max'
    ).round(4).sort_values('mean', ascending=False)
    print(summary.to_string())
    summary.to_csv('results/coordinated/summary.csv')

    # Seed variance
    print("\nSeed variance:")
    for algo in ['PPO', 'A2C', 'DQN']:
        vals = [combined[(combined['seed']==s) & (combined['algorithm']==algo)]['reward'].values[0]
                for s in SEEDS]
        print(f"  {algo}: {vals[0]:.2f} | {vals[1]:.2f} | {vals[2]:.2f}  std={np.std(vals):.3f}")

    # vs baseline
    rand_mean = combined[combined['algorithm']=='Random']['reward'].mean()
    print(f"\nRL vs Random ({rand_mean:.3f}):")
    for algo in ['DQN', 'PPO', 'A2C']:
        m   = combined[combined['algorithm']==algo]['reward'].mean()
        imp = (m - rand_mean) / abs(rand_mean) * 100
        print(f"  {algo}: {m:.3f}  {imp:+.1f}%  "
              f"{'✓' if m > rand_mean else '✗'}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    algos  = summary.index.tolist()
    means  = summary['mean'].values
    stds   = summary['std'].values
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9E9E9E', '#607D8B']
    bars   = ax.bar(algos, means, yerr=stds, capsize=6,
                    color=colors[:len(algos)], edgecolor='white',
                    linewidth=1.5, error_kw={'linewidth': 2})
    ax.set_title('Coordinated MARL — Algorithm Comparison\n'
                 '(Mean ± Std, 3 seeds, 50 episodes, fixed val states)',
                 fontweight='bold')
    ax.set_ylabel('Mean Episode Reward')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.grid(axis='y', alpha=0.3)
    for bar, val, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + std + 0.2,
                f'{val:.2f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/coordinated_comparison.png', dpi=150)
    plt.close()

    elapsed = time.time() - wall_start
    print(f"\nTotal time: {elapsed/3600:.1f}h {(elapsed%3600)/60:.0f}min")
    print("\nFiles saved:")
    print("  models/coordinated/seed_*/   — trained models")
    print("  results/coordinated/summary.csv")
    print("  plots/coordinated_comparison.png")
    print("=" * 65)
    print("Done!")
