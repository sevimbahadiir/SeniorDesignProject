"""
marl_train.py
-------------
Step 4: Multi-Agent Reinforcement Learning training using the Digital Twin.

Each microservice is controlled by an independent agent (IPPO approach).
Three algorithms are compared: PPO, A2C, DQN.
Two baselines are included: Random Policy, Rule-Based Policy.

Architecture:
    - 6 independent agents (one per service)
    - Each agent observes the full 30-dim system state
    - Each agent selects an action: scale_down(-1), keep(0), scale_up(+1)
    - Digital Twin simulates system response
    - Reward computed using tanh-compressed continuous reward (Option C)
    - Reward config loaded from reward_config.json for consistency with SAR step

Episode length: 10 steps (derived from DT trajectory validation)
Total timesteps: 200,000 per agent

Design decisions:
    - IPPO: decentralized execution, implicit coordination through shared DT
    - PPO over A2C: more stable, established robustness
    - DQN: value-based comparison (policy-based vs value-based analysis)
    - RecurrentPPO excluded: 10-step episodes don't require memory
    - Rule-based baseline: answers "why ML over simple rules?"
    - Random baseline: lower bound comparison
    - reward_config.json: same p50 thresholds as SAR step (reproducibility)

Input:  rl_dataset_sar_normalized.csv
        digital_twin_best.pth
        scaler.pkl
        reward_config.json

Output: models/          — trained model weights
        results/         — comparison CSVs
        plots/           — comparison charts
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
from stable_baselines3.common.env_checker import check_env

os.makedirs("models",  exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("plots",   exist_ok=True)

# ─── Configuration ────────────────────────────────────────────────────────────

SERVICES = ['cart', 'catalogue', 'payment', 'shipping', 'ratings', 'user']
METRICS  = ['num_pods', 'cpu_usage', 'mem_usage', 'request_rate', 'latency']

STATE_COLS = [f'{svc}_{met}' for svc in SERVICES for met in METRICS]

# Feature indices within the 30-dim state vector
LAT_IDX = {svc: i * len(METRICS) + 4 for i, svc in enumerate(SERVICES)}
POD_IDX = {svc: i * len(METRICS) + 0 for i, svc in enumerate(SERVICES)}

# Episode length from DT trajectory validation (reliable up to step 10)
EPISODE_LEN   = 10
TOTAL_STEPS   = 200_000
EVAL_EPISODES = 10

POD_COST = 0.05  # must match sar_create.py

# ─── Load Reward Config ───────────────────────────────────────────────────────
# Loaded from reward_config.json to ensure identical p50 thresholds
# as used in sar_create.py (reproducibility)

def load_reward_config(path: str = 'reward_config.json') -> dict:
    with open(path, 'r') as f:
        cfg = json.load(f)
    print(f"  Loaded reward_config.json")
    print(f"  {'Service':<12} {'p50 (ms)':>10} {'Weight':>8} {'In Reward':>10}")
    print(f"  {'-'*44}")
    for svc, v in cfg.items():
        in_r = "YES" if v['in_reward'] else "NO (Option C)"
        print(f"  {svc:<12} {v['p50']:>10.2f} {v['weight']:>8.1f} {in_r:>10}")
    return cfg

# ─── Digital Twin Model ───────────────────────────────────────────────────────

class DigitalTwin(nn.Module):
    """MLP Digital Twin — must match digital_twin.py architecture."""
    def __init__(self, input_dim=36, output_dim=30,
                 hidden_dims=None, dropout=0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 128]
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers += [nn.Linear(prev, output_dim), nn.Sigmoid()]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ─── Reward Function ──────────────────────────────────────────────────────────

def compute_reward(state: np.ndarray, scaler, reward_config: dict) -> float:
    """
    Compute tanh-compressed reward from normalized state.
    Uses p50 thresholds from reward_config.json (same as sar_create.py).
    Payment excluded (Option C).
    """
    reward = 0.0

    for svc, cfg in reward_config.items():
        if not cfg['in_reward']:
            continue  # payment excluded

        # Inverse transform: normalized → real ms
        col_idx  = LAT_IDX[svc]
        norm_lat = float(state[col_idx])
        real_lat = (norm_lat *
                    (scaler.data_max_[col_idx] - scaler.data_min_[col_idx])
                    + scaler.data_min_[col_idx])

        if real_lat <= 0:
            continue

        reward += np.tanh(1.0 - real_lat / cfg['p50']) * cfg['weight']

    # Pod cost penalty
    total_pods = 0.0
    for svc in SERVICES:
        col_idx  = POD_IDX[svc]
        norm_pod = float(state[col_idx])
        real_pod = (norm_pod *
                    (scaler.data_max_[col_idx] - scaler.data_min_[col_idx])
                    + scaler.data_min_[col_idx])
        total_pods += real_pod

    reward -= total_pods * POD_COST
    return float(round(reward, 4))

# ─── Gymnasium Environment ────────────────────────────────────────────────────

class RobotShopEnv(gym.Env):
    """
    Digital Twin-based Gymnasium environment for one microservice agent.

    Observation: full 30-dim normalized system state
    Action:      Discrete(3) — 0=scale_down, 1=keep, 2=scale_up
    Reward:      tanh-compressed (reward_config.json, Option C)
    Episode:     EPISODE_LEN steps (10, from DT trajectory validation)
    """
    metadata = {'render_modes': []}

    def __init__(self, service: str, dt_model: DigitalTwin,
                 sar_df: pd.DataFrame, scaler, reward_config: dict):
        super().__init__()
        self.service       = service
        self.svc_idx       = SERVICES.index(service)
        self.dt_model      = dt_model
        self.df            = sar_df.reset_index(drop=True)
        self.scaler        = scaler
        self.reward_config = reward_config
        self.step_count    = 0
        self.state         = None

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(30,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # 0=down, 1=keep, 2=up

        self.dt_model.eval()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        idx         = self.np_random.integers(0, len(self.df))
        self.state  = self.df[STATE_COLS].iloc[idx].values.astype(np.float32)
        self.step_count = 0
        return self.state.copy(), {}

    def step(self, action: int):
        self.step_count += 1

        # Build action vector: only this agent's service acts
        action_vec = np.zeros(len(SERVICES), dtype=np.float32)
        action_vec[self.svc_idx] = int(action) - 1  # 0→-1, 1→0, 2→+1

        # DT forward pass
        inp = np.concatenate([self.state, action_vec])
        with torch.no_grad():
            t          = torch.tensor(inp, dtype=torch.float32).unsqueeze(0)
            next_state = self.dt_model(t).squeeze(0).numpy()

        next_state = np.clip(next_state, 0.0, 1.0)
        reward     = compute_reward(next_state, self.scaler, self.reward_config)
        self.state = next_state

        terminated = False
        truncated  = self.step_count >= EPISODE_LEN
        return self.state.copy(), reward, terminated, truncated, {}

    def render(self):
        pass

# ─── Baselines ────────────────────────────────────────────────────────────────

def evaluate_random(env: RobotShopEnv,
                    n_episodes: int = EVAL_EPISODES) -> float:
    """Random policy baseline — lower bound."""
    total = 0.0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done   = False
        while not done:
            action               = env.action_space.sample()
            obs, r, term, trunc, _ = env.step(action)
            total += r
            done   = term or trunc
    return total / n_episodes


def evaluate_rule_based(env: RobotShopEnv, reward_config: dict,
                        n_episodes: int = EVAL_EPISODES) -> float:
    """
    Rule-based baseline:
        latency > p75 → scale up
        latency < p25 → scale down
        else          → keep

    Thresholds from reward_config.json (same source as reward function).
    Answers: 'why ML over simple rules?'
    """
    svc     = env.service
    lat_idx = LAT_IDX[svc]
    scaler  = env.scaler

    col_min = scaler.data_min_[lat_idx]
    col_max = scaler.data_max_[lat_idx]
    rng     = col_max - col_min + 1e-8

    cfg     = reward_config.get(svc, {})
    p25_norm = (cfg.get('p25', 0)   - col_min) / rng
    p75_norm = (cfg.get('p75', 1e9) - col_min) / rng

    total = 0.0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done   = False
        while not done:
            lat    = obs[lat_idx]
            if lat > p75_norm:
                action = 2   # scale up
            elif lat < p25_norm:
                action = 0   # scale down
            else:
                action = 1   # keep
            obs, r, term, trunc, _ = env.step(action)
            total += r
            done   = term or trunc
    return total / n_episodes


def evaluate_model(model, env: RobotShopEnv,
                   n_episodes: int = EVAL_EPISODES) -> float:
    """Evaluate a trained SB3 model."""
    total = 0.0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done   = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            total += r
            done   = term or trunc
    return total / n_episodes

# ─── Algorithm Configs ────────────────────────────────────────────────────────

ALGO_CONFIGS = {
    'PPO': {
        'class':  PPO,
        'params': {
            'policy':        'MlpPolicy',
            'learning_rate': 3e-4,
            'n_steps':       128,
            'batch_size':    64,
            'n_epochs':      10,
            'gamma':         0.99,
            'verbose':       0,
        }
    },
    'A2C': {
        'class':  A2C,
        'params': {
            'policy':        'MlpPolicy',
            'learning_rate': 7e-4,
            'n_steps':       64,
            'gamma':         0.99,
            'verbose':       0,
        }
    },
    'DQN': {
        'class':  DQN,
        'params': {
            'policy':               'MlpPolicy',
            'learning_rate':        1e-4,
            'batch_size':           64,
            'buffer_size':          10_000,
            'learning_starts':      500,
            'exploration_fraction': 0.3,
            'gamma':                0.99,
            'verbose':              0,
        }
    },
}

# ─── Training ─────────────────────────────────────────────────────────────────

def train_all(sar_df, dt_model, scaler, reward_config):
    results    = []
    wall_start = time.time()

    # ── Baselines ──
    print("\n" + "=" * 65)
    print("BASELINES")
    print("=" * 65)

    for svc in SERVICES:
        env = RobotShopEnv(svc, dt_model, sar_df, scaler, reward_config)

        r_random = evaluate_random(env, EVAL_EPISODES)
        r_rule   = evaluate_rule_based(env, reward_config, EVAL_EPISODES)

        print(f"  {svc:<12}  Random: {r_random:>8.4f}  "
              f"Rule-based: {r_rule:>8.4f}")

        results.append({'algorithm': 'Random',     'service': svc,
                        'reward': round(r_random, 4), 'train_time_s': 0})
        results.append({'algorithm': 'Rule-Based', 'service': svc,
                        'reward': round(r_rule, 4),   'train_time_s': 0})
        env.close()

    # ── RL Algorithms ──
    for algo_name, cfg in ALGO_CONFIGS.items():
        print("\n" + "=" * 65)
        print(f"ALGORITHM: {algo_name}")
        print("=" * 65)

        for svc in SERVICES:
            t0  = time.time()
            env = RobotShopEnv(svc, dt_model, sar_df, scaler, reward_config)

            model = cfg['class'](env=env, **cfg['params'])
            model.learn(total_timesteps=TOTAL_STEPS, progress_bar=False)
            model.save(f"models/{algo_name}_{svc}")

            reward  = evaluate_model(model, env, EVAL_EPISODES)
            elapsed = time.time() - t0

            print(f"  {svc:<12}  reward: {reward:>8.4f}  "
                  f"time: {elapsed:.0f}s")

            results.append({
                'algorithm':    algo_name,
                'service':      svc,
                'reward':       round(reward, 4),
                'train_time_s': round(elapsed, 1),
            })
            env.close()

    total_time = time.time() - wall_start
    print(f"\n  Total training time: {total_time/60:.1f} min")
    return pd.DataFrame(results)

# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_results(df: pd.DataFrame):

    # Plot 1: Algorithm comparison + heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    algo_means = (df.groupby('algorithm')['reward']
                  .mean().sort_values(ascending=False))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']
    bars   = axes[0].bar(algo_means.index, algo_means.values,
                         color=colors[:len(algo_means)],
                         edgecolor='white', linewidth=1.5)
    axes[0].set_title('Algorithm Comparison\n(Mean Reward Across All Services)',
                      fontweight='bold')
    axes[0].set_ylabel('Mean Episode Reward')
    axes[0].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[0].grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, algo_means.values):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.2,
                     f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

    pivot = df.pivot_table(index='algorithm', columns='service',
                           values='reward', aggfunc='mean')
    im = axes[1].imshow(pivot.values, cmap='RdYlGn', aspect='auto')
    axes[1].set_xticks(range(len(pivot.columns)))
    axes[1].set_xticklabels(pivot.columns, rotation=45, ha='right')
    axes[1].set_yticks(range(len(pivot.index)))
    axes[1].set_yticklabels(pivot.index)
    axes[1].set_title('Per-Service Reward Heatmap', fontweight='bold')
    plt.colorbar(im, ax=axes[1])
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            axes[1].text(j, i, f'{pivot.values[i, j]:.1f}',
                        ha='center', va='center',
                        fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig('plots/marl_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ plots/marl_comparison.png")

    # Plot 2: Per-service bar chart
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes   = axes.flatten()
    algos  = df['algorithm'].unique()
    colors = {'Random': '#9E9E9E', 'Rule-Based': '#607D8B',
              'PPO': '#2196F3', 'A2C': '#4CAF50', 'DQN': '#FF9800'}

    for i, svc in enumerate(SERVICES):
        svc_df  = df[df['service'] == svc].set_index('algorithm')
        rewards = [svc_df.loc[a, 'reward']
                   if a in svc_df.index else 0 for a in algos]
        clrs    = [colors.get(a, '#333') for a in algos]
        bars    = axes[i].bar(algos, rewards, color=clrs,
                              edgecolor='white', linewidth=1.5)
        axes[i].set_title(f'{svc}', fontweight='bold')
        axes[i].set_ylabel('Mean Reward')
        axes[i].tick_params(axis='x', rotation=30)
        axes[i].grid(axis='y', alpha=0.3)
        axes[i].axhline(0, color='black', linewidth=0.8, linestyle='--')

    plt.suptitle('Per-Service Algorithm Comparison',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/marl_per_service.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ plots/marl_per_service.png")

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 65)
    print("MARL TRAINING — Robot Shop Autoscaling")
    print("=" * 65)
    print(f"  Episode length : {EPISODE_LEN} steps")
    print(f"  Total timesteps: {TOTAL_STEPS:,} per agent")
    print(f"  Algorithms     : PPO, A2C, DQN + Random, Rule-Based")
    print(f"  Services       : {SERVICES}")
    print(f"  Reward config  : reward_config.json (auto p50)")

    print("\n[1] Loading SAR dataset...")
    sar_df = pd.read_csv('rl_dataset_sar_normalized.csv')
    print(f"    {len(sar_df)} rows loaded.")

    print("\n[2] Loading scaler...")
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("    scaler.pkl loaded.")

    print("\n[3] Loading reward config...")
    reward_config = load_reward_config('reward_config.json')

    print("\n[4] Loading Digital Twin...")
    dt_model = DigitalTwin()
    dt_model.load_state_dict(
        torch.load('digital_twin_best.pth', map_location='cpu')
    )
    dt_model.eval()
    print("    digital_twin_best.pth loaded.")

    print("\n[5] Checking environment...")
    test_env = RobotShopEnv('cart', dt_model, sar_df, scaler, reward_config)
    check_env(test_env, warn=True)
    obs, _ = test_env.reset()
    total_r = 0
    for _ in range(EPISODE_LEN):
        obs, r, term, trunc, _ = test_env.step(test_env.action_space.sample())
        total_r += r
    print(f"    Environment OK. Episode reward (random): {total_r:.4f}")
    test_env.close()

    print("\n[6] Training...")
    results_df = train_all(sar_df, dt_model, scaler, reward_config)

    results_df.to_csv('results/marl_results.csv', index=False)
    print("\n[7] Results saved: results/marl_results.csv")

    print("\n" + "=" * 65)
    print("SUMMARY — Mean Reward by Algorithm")
    print("=" * 65)
    summary = (results_df.groupby('algorithm')['reward']
               .agg(['mean', 'std', 'min', 'max'])
               .round(4)
               .sort_values('mean', ascending=False))
    summary.columns = ['Mean', 'Std', 'Min', 'Max']
    print(summary.to_string())
    summary.to_csv('results/algorithm_comparison.csv')

    print("\n" + "=" * 65)
    print("SUMMARY — Mean Reward by Service")
    print("=" * 65)
    svc_summary = (results_df
                   .groupby(['service', 'algorithm'])['reward']
                   .mean()
                   .unstack())
    print(svc_summary.round(4).to_string())
    svc_summary.to_csv('results/service_comparison.csv')

    print("\n[8] Generating plots...")
    plot_results(results_df)

    print("\n" + "=" * 65)
    print("DONE!")
    print(f"  models/   — trained agent weights (18 models)")
    print(f"  results/  — marl_results.csv, algorithm_comparison.csv")
    print(f"  plots/    — marl_comparison.png, marl_per_service.png")
    print("=" * 65)