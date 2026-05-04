"""
marl_train_multiseed.py
-----------------------
Runs MARL training with 3 random seeds for statistical reliability.

Each seed produces independent training runs. Final results report
mean ± std across seeds, which is the standard for academic publication.

Seeds: 42, 0, 123
Total runs: 3 seeds × 3 algorithms × 6 services = 54 training runs
Estimated time: ~3 × 65 min = ~3.5 hours

Output:
    results/seed_{seed}/marl_results.csv   — per-seed results
    results/multiseed_summary.csv          — mean ± std across seeds
    results/multiseed_per_service.csv      — per-service mean ± std
    plots/multiseed_comparison.png         — final comparison plot
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

os.makedirs("results", exist_ok=True)
os.makedirs("plots",   exist_ok=True)

# ─── Configuration ────────────────────────────────────────────────────────────

SERVICES    = ['cart', 'catalogue', 'payment', 'shipping', 'ratings', 'user']
METRICS     = ['num_pods', 'cpu_usage', 'mem_usage', 'request_rate', 'latency']
STATE_COLS  = [f'{svc}_{met}' for svc in SERVICES for met in METRICS]
LAT_IDX     = {svc: i * len(METRICS) + 4 for i, svc in enumerate(SERVICES)}
POD_IDX     = {svc: i * len(METRICS) + 0 for i, svc in enumerate(SERVICES)}

EPISODE_LEN   = 10
TOTAL_STEPS   = 200_000
EVAL_EPISODES = 10
SEEDS         = [42, 0, 123]
POD_COST      = 0.05

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
    reward = 0.0
    for svc, cfg in reward_config.items():
        if not cfg['in_reward']:
            continue
        col_idx  = LAT_IDX[svc]
        norm_lat = float(state[col_idx])
        real_lat = (norm_lat *
                    (scaler.data_max_[col_idx] - scaler.data_min_[col_idx])
                    + scaler.data_min_[col_idx])
        if real_lat <= 0:
            continue
        reward += np.tanh(1.0 - real_lat / cfg['p50']) * cfg['weight']
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

# ─── Environment ──────────────────────────────────────────────────────────────

class RobotShopEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, service, dt_model, sar_df, scaler, reward_config):
        super().__init__()
        self.service       = service
        self.svc_idx       = SERVICES.index(service)
        self.dt_model      = dt_model
        self.df            = sar_df.reset_index(drop=True)
        self.scaler        = scaler
        self.reward_config = reward_config
        self.step_count    = 0
        self.state         = None
        self.observation_space = spaces.Box(0.0, 1.0, shape=(30,), dtype=np.float32)
        self.action_space      = spaces.Discrete(3)
        self.dt_model.eval()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        idx         = self.np_random.integers(0, len(self.df))
        self.state  = self.df[STATE_COLS].iloc[idx].values.astype(np.float32)
        self.step_count = 0
        return self.state.copy(), {}

    def step(self, action):
        self.step_count += 1
        action_vec = np.zeros(len(SERVICES), dtype=np.float32)
        action_vec[self.svc_idx] = int(action) - 1
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

def evaluate_random(env, n=EVAL_EPISODES):
    total = 0.0
    for _ in range(n):
        obs, _ = env.reset()
        done = False
        while not done:
            obs, r, term, trunc, _ = env.step(env.action_space.sample())
            total += r
            done = term or trunc
    return total / n

def evaluate_rule_based(env, reward_config, n=EVAL_EPISODES):
    svc     = env.service
    lat_idx = LAT_IDX[svc]
    scaler  = env.scaler
    rng     = scaler.data_max_[lat_idx] - scaler.data_min_[lat_idx] + 1e-8
    cfg     = reward_config.get(svc, {})
    p25_n   = (cfg.get('p25', 0)   - scaler.data_min_[lat_idx]) / rng
    p75_n   = (cfg.get('p75', 1e9) - scaler.data_min_[lat_idx]) / rng
    total = 0.0
    for _ in range(n):
        obs, _ = env.reset()
        done = False
        while not done:+
            lat    = obs[lat_idx]
            action = 2 if lat > p75_n else (0 if lat < p25_n else 1)
            obs, r, term, trunc, _ = env.step(action)
            total += r
            done = term or trunc
    return total / n

def evaluate_model(model, env, n=EVAL_EPISODES):
    total = 0.0
    for _ in range(n):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            total += r
            done = term or trunc
    return total / n

# ─── Algo configs ─────────────────────────────────────────────────────────────

def get_algo_configs(seed):
    return {
        'PPO': {
            'class':  PPO,
            'params': {'policy': 'MlpPolicy', 'learning_rate': 3e-4,
                       'n_steps': 128, 'batch_size': 64, 'n_epochs': 10,
                       'gamma': 0.99, 'seed': seed, 'verbose': 0}
        },
        'A2C': {
            'class':  A2C,
            'params': {'policy': 'MlpPolicy', 'learning_rate': 7e-4,
                       'n_steps': 64, 'gamma': 0.99,
                       'seed': seed, 'verbose': 0}
        },
        'DQN': {
            'class':  DQN,
            'params': {'policy': 'MlpPolicy', 'learning_rate': 1e-4,
                       'batch_size': 64, 'buffer_size': 10_000,
                       'learning_starts': 500, 'exploration_fraction': 0.3,
                       'gamma': 0.99, 'seed': seed, 'verbose': 0}
        },
    }

# ─── Single seed run ──────────────────────────────────────────────────────────

def run_seed(seed, sar_df, dt_model, scaler, reward_config):
    print(f"\n{'='*65}")
    print(f"SEED {seed}")
    print(f"{'='*65}")

    seed_dir = f"results/seed_{seed}"
    model_dir = f"models/seed_{seed}"
    os.makedirs(seed_dir,  exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    results = []

    # Baselines (seed-independent but run per seed for consistency)
    print("\nBASELINES")
    for svc in SERVICES:
        env = RobotShopEnv(svc, dt_model, sar_df, scaler, reward_config)
        env.reset(seed=seed)
        r_rand = evaluate_random(env)
        r_rule = evaluate_rule_based(env, reward_config)
        print(f"  {svc:<12}  Random: {r_rand:>8.4f}  Rule-based: {r_rule:>8.4f}")
        results.append({'seed': seed, 'algorithm': 'Random',
                        'service': svc, 'reward': round(r_rand, 4)})
        results.append({'seed': seed, 'algorithm': 'Rule-Based',
                        'service': svc, 'reward': round(r_rule, 4)})
        env.close()

    # RL algorithms
    for algo_name, cfg in get_algo_configs(seed).items():
        print(f"\nALGORITHM: {algo_name}")
        for svc in SERVICES:
            t0  = time.time()
            env = RobotShopEnv(svc, dt_model, sar_df, scaler, reward_config)
            model = cfg['class'](env=env, **cfg['params'])
            model.learn(total_timesteps=TOTAL_STEPS, progress_bar=False)
            model.save(f"{model_dir}/{algo_name}_{svc}")
            reward  = evaluate_model(model, env)
            elapsed = time.time() - t0
            print(f"  {svc:<12}  reward: {reward:>8.4f}  time: {elapsed:.0f}s")
            results.append({'seed': seed, 'algorithm': algo_name,
                            'service': svc, 'reward': round(reward, 4)})
            env.close()

    df = pd.DataFrame(results)
    df.to_csv(f"{seed_dir}/marl_results.csv", index=False)
    return df

# ─── Aggregate across seeds ───────────────────────────────────────────────────

def aggregate_seeds(all_dfs):
    combined = pd.concat(all_dfs, ignore_index=True)

    # Mean ± std by algorithm
    summary = combined.groupby('algorithm')['reward'].agg(
        mean='mean', std='std', min='min', max='max'
    ).round(4).sort_values('mean', ascending=False)

    # Mean ± std by algorithm × service
    per_svc = combined.groupby(['algorithm', 'service'])['reward'].agg(
        mean='mean', std='std'
    ).round(4).unstack('service')

    return combined, summary, per_svc

# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot_multiseed(summary, combined):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart with error bars
    algos  = summary.index.tolist()
    means  = summary['mean'].values
    stds   = summary['std'].values
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']

    bars = axes[0].bar(algos, means, yerr=stds, capsize=5,
                       color=colors[:len(algos)],
                       edgecolor='white', linewidth=1.5, error_kw={'linewidth': 2})
    axes[0].set_title('Algorithm Comparison\n(Mean ± Std across 3 seeds)',
                      fontweight='bold')
    axes[0].set_ylabel('Mean Episode Reward')
    axes[0].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[0].grid(axis='y', alpha=0.3)
    for bar, val, std in zip(bars, means, stds):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + std + 0.3,
                     f'{val:.2f}', ha='center', va='bottom',
                     fontsize=9, fontweight='bold')

    # Violin plot per algorithm
    rl_data = combined[combined['algorithm'].isin(['PPO', 'A2C', 'DQN'])]
    algo_list = ['PPO', 'A2C', 'DQN']
    data_list = [rl_data[rl_data['algorithm'] == a]['reward'].values
                 for a in algo_list]
    vp = axes[1].violinplot(data_list, positions=range(len(algo_list)),
                             showmeans=True, showmedians=True)
    axes[1].set_xticks(range(len(algo_list)))
    axes[1].set_xticklabels(algo_list)
    axes[1].set_title('Reward Distribution (RL algorithms, 3 seeds × 6 services)',
                      fontweight='bold')
    axes[1].set_ylabel('Episode Reward')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/multiseed_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ plots/multiseed_comparison.png")

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 65)
    print("MARL MULTI-SEED TRAINING")
    print("=" * 65)
    print(f"  Seeds          : {SEEDS}")
    print(f"  Episode length : {EPISODE_LEN}")
    print(f"  Total timesteps: {TOTAL_STEPS:,} per agent")
    print(f"  Est. total time: ~{len(SEEDS) * 65} min")

    print("\n[1] Loading data...")
    sar_df = pd.read_csv('rl_dataset_sar_normalized.csv')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('reward_config.json', 'r') as f:
        reward_config = json.load(f)
    dt_model = DigitalTwin()
    dt_model.load_state_dict(
        torch.load('digital_twin_best.pth', map_location='cpu')
    )
    dt_model.eval()
    print(f"    {len(sar_df)} rows | scaler | reward_config | DT loaded")

    wall_start = time.time()
    all_dfs = []

    for seed in SEEDS:
        df_seed = run_seed(seed, sar_df, dt_model, scaler, reward_config)
        all_dfs.append(df_seed)

    # Aggregate
    combined, summary, per_svc = aggregate_seeds(all_dfs)

    combined.to_csv('results/multiseed_all.csv', index=False)
    summary.to_csv('results/multiseed_summary.csv')
    per_svc.to_csv('results/multiseed_per_service.csv')

    print("\n" + "=" * 65)
    print("FINAL SUMMARY — Mean ± Std (3 seeds)")
    print("=" * 65)
    print(summary.to_string())

    print("\n" + "=" * 65)
    print("PER-SERVICE MEAN REWARD")
    print("=" * 65)
    mean_only = combined.groupby(['algorithm', 'service'])['reward'].mean().unstack()
    print(mean_only.round(3).to_string())

    print("\n[Plots]")
    plot_multiseed(summary, combined)

    elapsed = time.time() - wall_start
    print(f"\nTotal time: {elapsed/3600:.1f} h {(elapsed%3600)/60:.0f} min")
    print("\nFiles saved:")
    print("  results/seed_*/marl_results.csv  — per-seed results")
    print("  results/multiseed_summary.csv    — mean ± std by algorithm")
    print("  results/multiseed_per_service.csv")
    print("  plots/multiseed_comparison.png")
    print("=" * 65)
