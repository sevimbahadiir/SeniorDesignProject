"""
mappo_train.py
--------------
Multi-Agent Proximal Policy Optimization (MAPPO) for Robot Shop
microservice autoscaling using Ray RLlib.

Architecture: CTDE (Centralized Training, Decentralized Execution)
    - Each agent controls one microservice
    - During training: centralized critic sees global state
    - During execution: each agent acts on its own observation
    - Agents communicate implicitly through shared DT environment

Observation per agent (36 dim):
    - Global system state: 30 dim (all services)
    - Own last action: 1 dim
    - Other 5 agents' last actions: 5 dim

Action per agent:
    Discrete(3): 0=scale_down, 1=keep, 2=scale_up

Reward:
    Shared tanh-compressed reward (latency + pod cost)
    Loaded from reward_config.json for consistency with SAR step

Episode length: 10 steps (from DT trajectory validation)
Seeds: [42, 0, 123]
Eval episodes: 50 (fixed val-set starts)

Input:  rl_dataset_sar_normalized.csv
        digital_twin_best.pth
        scaler.pkl
        reward_config.json

Output: models/mappo/          — trained model checkpoints
        results/mappo/         — comparison CSVs
        plots/mappo_comparison.png
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

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces

os.makedirs('models/mappo', exist_ok=True)
os.makedirs('results/mappo', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# ─── Configuration ────────────────────────────────────────────────────────────

SERVICES   = ['cart', 'catalogue', 'payment', 'shipping', 'ratings', 'user']
METRICS    = ['num_pods', 'cpu_usage', 'mem_usage', 'request_rate', 'latency']
STATE_COLS = [f'{svc}_{met}' for svc in SERVICES for met in METRICS]
LAT_IDX    = {svc: i * len(METRICS) + 4 for i, svc in enumerate(SERVICES)}
POD_IDX    = {svc: i * len(METRICS) + 0 for i, svc in enumerate(SERVICES)}

EPISODE_LEN   = 10     # from DT trajectory validation
EVAL_EPISODES = 50     # fixed val-set starts
SEEDS         = [42, 0, 123]
POD_COST      = 0.05

# Observation: global state (30) + own last action (1) + others' actions (5)
OBS_DIM = 30 + 1 + 5   # 36

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

# ─── Multi-Agent Environment ──────────────────────────────────────────────────

class RobotShopMAEnv(MultiAgentEnv):
    """
    MAPPO Multi-Agent Environment for Robot Shop autoscaling.

    CTDE implementation:
    - Each agent observes: global_state(30) + own_action(1) + others_actions(5)
    - All agents act simultaneously each step
    - DT receives all 6 actions at once → single forward pass
    - All agents share the same reward (cooperative)

    This is the key difference from IPPO:
    - Agents see each other's last actions in their observation
    - This allows implicit communication and coordination
    - Centralized training: PPO critic can use full joint observation
    """
    metadata = {'render_modes': []}

    def __init__(self, env_config: dict):
        super().__init__()

        self.sar_df       = env_config['sar_df']
        self.dt_model     = env_config['dt_model']
        self.scaler       = env_config['scaler']
        self.reward_cfg   = env_config['reward_config']
        self.fixed_starts = env_config.get('fixed_starts', None)

        self.ep_count   = 0
        self.step_count = 0
        self.state      = None
        self.last_actions = {svc: 0 for svc in SERVICES}  # 0 = keep

        # Agent IDs
        self.agents         = SERVICES
        self.possible_agents = SERVICES

        # Spaces
        self.observation_spaces = {
            svc: spaces.Box(low=-1.0, high=1.0,
                           shape=(OBS_DIM,), dtype=np.float32)
            for svc in SERVICES
        }
        self.action_spaces = {
            svc: spaces.Discrete(3)
            for svc in SERVICES
        }
        # RLlib requires these
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self.dt_model.eval()

    def _get_obs(self, svc: str) -> np.ndarray:
        """
        Build observation for one agent:
        [global_state(30), own_last_action(1), others_last_actions(5)]
        """
        own_action    = np.array([self.last_actions[svc]], dtype=np.float32)
        other_actions = np.array(
            [self.last_actions[s] for s in SERVICES if s != svc],
            dtype=np.float32
        )
        return np.concatenate([self.state, own_action, other_actions])

    def reset(self, *, seed=None, options=None):
        if self.fixed_starts is not None:
            idx = int(self.fixed_starts[self.ep_count % len(self.fixed_starts)])
        else:
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, len(self.sar_df))

        self.state        = self.sar_df[STATE_COLS].iloc[idx].values.astype(np.float32)
        self.step_count   = 0
        self.ep_count    += 1
        self.last_actions = {svc: 0 for svc in SERVICES}

        obs = {svc: self._get_obs(svc) for svc in SERVICES}
        return obs, {}

    def step(self, action_dict: dict):
        self.step_count += 1

        # Build joint action vector
        action_vec = np.zeros(len(SERVICES), dtype=np.float32)
        for svc, action in action_dict.items():
            svc_idx           = SERVICES.index(svc)
            action_val        = int(action) - 1  # 0→-1, 1→0, 2→+1
            action_vec[svc_idx]       = action_val
            self.last_actions[svc]    = action_val

        # Single DT forward pass with all actions
        inp = np.concatenate([self.state, action_vec])
        with torch.no_grad():
            t          = torch.tensor(inp, dtype=torch.float32).unsqueeze(0)
            next_state = self.dt_model(t).squeeze(0).numpy()
        next_state = np.clip(next_state, 0.0, 1.0)

        # Shared reward
        reward = compute_reward(next_state, self.scaler, self.reward_cfg)
        self.state = next_state

        # Build per-agent outputs
        done      = self.step_count >= EPISODE_LEN
        obs       = {svc: self._get_obs(svc) for svc in SERVICES}
        rewards   = {svc: reward for svc in SERVICES}
        terminateds = {svc: False for svc in SERVICES}
        terminateds['__all__'] = False
        truncateds  = {svc: done for svc in SERVICES}
        truncateds['__all__'] = done
        infos = {svc: {} for svc in SERVICES}

        return obs, rewards, terminateds, truncateds, infos

# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_mappo(algo, sar_df, dt_model, scaler, reward_config,
                   fixed_starts, n=EVAL_EPISODES) -> float:
    """Evaluate trained MAPPO agents in coordinated fashion."""
    total = 0.0
    last_actions = {svc: 0 for svc in SERVICES}

    for ep in range(n):
        idx   = int(fixed_starts[ep % len(fixed_starts)])
        state = sar_df[STATE_COLS].iloc[idx].values.astype(np.float32)
        last_actions = {svc: 0 for svc in SERVICES}
        ep_reward = 0.0

        for _ in range(EPISODE_LEN):
            # Build obs and get actions from all agents
            action_vec = np.zeros(len(SERVICES), dtype=np.float32)
            for svc in SERVICES:
                own_act    = np.array([last_actions[svc]], dtype=np.float32)
                other_acts = np.array(
                    [last_actions[s] for s in SERVICES if s != svc],
                    dtype=np.float32
                )
                obs    = np.concatenate([state, own_act, other_acts])
                action = algo.compute_single_action(obs, policy_id=svc)
                action_vec[SERVICES.index(svc)] = int(action) - 1
                last_actions[svc] = int(action) - 1

            # DT forward pass
            inp = np.concatenate([state, action_vec])
            with torch.no_grad():
                t          = torch.tensor(inp, dtype=torch.float32).unsqueeze(0)
                next_state = dt_model(t).squeeze(0).numpy()
            next_state = np.clip(next_state, 0.0, 1.0)

            ep_reward += compute_reward(next_state, scaler, reward_config)
            state      = next_state

        total += ep_reward

    return total / n


def evaluate_baseline(policy, sar_df, dt_model, scaler,
                      reward_config, fixed_starts, n=EVAL_EPISODES) -> float:
    """Random or rule-based baseline in coordinated fashion."""
    total = 0.0
    np.random.seed(999)

    for ep in range(n):
        idx   = int(fixed_starts[ep % len(fixed_starts)])
        state = sar_df[STATE_COLS].iloc[idx].values.astype(np.float32)
        ep_reward = 0.0

        for _ in range(EPISODE_LEN):
            action_vec = np.zeros(len(SERVICES), dtype=np.float32)
            for svc in SERVICES:
                lat_idx = LAT_IDX[svc]
                col_min = scaler.data_min_[lat_idx]
                col_max = scaler.data_max_[lat_idx]
                rng_s   = col_max - col_min + 1e-8
                cfg     = reward_config.get(svc, {})
                p25_n   = (cfg.get('p25', 0)   - col_min) / rng_s
                p75_n   = (cfg.get('p75', 1e9) - col_min) / rng_s
                lat     = float(state[lat_idx])

                if policy == 'random':
                    action_vec[SERVICES.index(svc)] = float(
                        np.random.choice([-1, 0, 1]))
                else:
                    action_vec[SERVICES.index(svc)] = (
                        1.0 if lat > p75_n else (-1.0 if lat < p25_n else 0.0))

            inp = np.concatenate([state, action_vec])
            with torch.no_grad():
                t          = torch.tensor(inp, dtype=torch.float32).unsqueeze(0)
                next_state = dt_model(t).squeeze(0).numpy()
            next_state  = np.clip(next_state, 0.0, 1.0)
            ep_reward  += compute_reward(next_state, scaler, reward_config)
            state       = next_state

        total += ep_reward

    return total / n

# ─── Training ─────────────────────────────────────────────────────────────────

def train_mappo(sar_df, dt_model, scaler, reward_config,
                fixed_starts, seed: int, iterations: int = 100):
    """Train MAPPO for one seed."""

    env_config = {
        'sar_df':       sar_df,
        'dt_model':     dt_model,
        'scaler':       scaler,
        'reward_config': reward_config,
        'fixed_starts': None,  # random starts during training
    }

    config = (
        PPOConfig()
        .environment(
            env=RobotShopMAEnv,
            env_config=env_config,
        )
        .framework('torch')
        .multi_agent(
            policies={svc: (None,
                           spaces.Box(-1., 1., (OBS_DIM,), np.float32),
                           spaces.Discrete(3),
                           {})
                     for svc in SERVICES},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
        )
        .training(
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            num_epochs=10,
            minibatch_size=64,
            train_batch_size=500,
            use_gae=True,
        )
        .rollouts(
            num_rollout_workers=0,  # single worker for reproducibility
        )
        .resources(
            num_gpus=0,
        )
        .debugging(seed=seed)
    )

    algo = config.build()

    print(f"\n  Training MAPPO (seed={seed}, {iterations} iterations)...")
    t0 = time.time()
    for i in range(iterations):
        result = algo.train()
        if (i + 1) % 20 == 0:
            mean_r = result.get('episode_reward_mean', float('nan'))
            elapsed = time.time() - t0
            print(f"    Iter {i+1:>4}/{iterations}  "
                  f"reward_mean={mean_r:.4f}  "
                  f"time={elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"  Training done: {elapsed:.0f}s")

    # Save checkpoint
    ckpt_dir = f'models/mappo/seed_{seed}'
    os.makedirs(ckpt_dir, exist_ok=True)
    algo.save(ckpt_dir)
    print(f"  Checkpoint saved: {ckpt_dir}")

    return algo

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 65)
    print("MAPPO TRAINING — Robot Shop Autoscaling")
    print("=" * 65)
    print(f"  Algorithm      : MAPPO (Multi-Agent PPO via RLlib)")
    print(f"  Paradigm       : CTDE (Centralized Train, Decentralized Exec)")
    print(f"  Observation    : global state(30) + own action(1) + others(5) = 36")
    print(f"  Episode length : {EPISODE_LEN} steps")
    print(f"  Seeds          : {SEEDS}")
    print(f"  Eval episodes  : {EVAL_EPISODES} (fixed val states)")

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
    print(f"    {len(sar_df)} rows | scaler | reward_config | DT loaded")

    # Fixed evaluation starts
    np.random.seed(999)
    val_start    = int(len(sar_df) * 0.8)
    fixed_starts = np.random.choice(
        range(val_start, len(sar_df)), size=EVAL_EPISODES, replace=False)
    print(f"    Fixed eval: {EVAL_EPISODES} states from val set")

    # Baselines
    print("\n[2] Computing baselines...")
    r_random = evaluate_baseline(
        'random', sar_df, dt_model, scaler, reward_config, fixed_starts)
    r_rule   = evaluate_baseline(
        'rule',   sar_df, dt_model, scaler, reward_config, fixed_starts)
    print(f"    Random:     {r_random:.4f}")
    print(f"    Rule-Based: {r_rule:.4f}")

    # Initialize Ray
    print("\n[3] Initializing Ray...")
    ray.init(ignore_reinit_error=True, num_cpus=4)

    # Train across seeds
    wall_start = time.time()
    results    = []

    for seed in SEEDS:
        print(f"\n{'='*55}\nSEED {seed}\n{'='*55}")
        algo = train_mappo(
            sar_df, dt_model, scaler, reward_config,
            fixed_starts, seed=seed, iterations=150
        )

        # Evaluate
        r_mappo = evaluate_mappo(
            algo, sar_df, dt_model, scaler, reward_config, fixed_starts)
        print(f"  MAPPO reward: {r_mappo:.4f}")

        results.append({'seed': seed, 'algorithm': 'MAPPO',
                        'reward': round(r_mappo, 4)})
        results.append({'seed': seed, 'algorithm': 'Random',
                        'reward': round(r_random, 4)})
        results.append({'seed': seed, 'algorithm': 'Rule-Based',
                        'reward': round(r_rule, 4)})

        algo.stop()

    ray.shutdown()

    # Summary
    df = pd.DataFrame(results)
    df.to_csv('results/mappo/all_results.csv', index=False)

    print("\n" + "=" * 65)
    print("SUMMARY — MAPPO (3 seeds, 50 episodes, fixed val states)")
    print("=" * 65)
    summary = df.groupby('algorithm')['reward'].agg(
        mean='mean', std='std', min='min', max='max'
    ).round(4).sort_values('mean', ascending=False)
    print(summary.to_string())
    summary.to_csv('results/mappo/summary.csv')

    # Seed variance
    print("\nSeed variance:")
    for algo in ['MAPPO']:
        vals = [df[(df['seed']==s) & (df['algorithm']==algo)]['reward'].values[0]
                for s in SEEDS]
        print(f"  {algo}: {vals[0]:.2f} | {vals[1]:.2f} | {vals[2]:.2f}"
              f"  std={np.std(vals):.3f}")

    rand_mean = df[df['algorithm']=='Random']['reward'].mean()
    mappo_mean = df[df['algorithm']=='MAPPO']['reward'].mean()
    imp = (mappo_mean - rand_mean) / abs(rand_mean) * 100
    print(f"\nMAPPO vs Random: {imp:+.1f}%  "
          f"{'✓' if mappo_mean > rand_mean else '✗'}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    algo_means = summary['mean']
    algo_stds  = summary['std']
    colors = ['#2196F3', '#9E9E9E', '#607D8B']
    bars = ax.bar(summary.index, algo_means, yerr=algo_stds,
                  capsize=6, color=colors, edgecolor='white',
                  linewidth=1.5, error_kw={'linewidth': 2})
    ax.set_title('MAPPO — Algorithm Comparison\n'
                 '(Mean ± Std, 3 seeds, 50 episodes, fixed val states)',
                 fontweight='bold')
    ax.set_ylabel('Mean Episode Reward')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.grid(axis='y', alpha=0.3)
    for bar, val, std in zip(bars, algo_means, algo_stds):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + std + 0.2,
                f'{val:.2f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/mappo_comparison.png', dpi=150)
    plt.close()

    elapsed = time.time() - wall_start
    print(f"\nTotal time: {elapsed/3600:.1f}h {(elapsed%3600)/60:.0f}min")
    print("\nFiles saved:")
    print("  models/mappo/seed_*/     — checkpoints")
    print("  results/mappo/summary.csv")
    print("  plots/mappo_comparison.png")
    print("=" * 65)
    print("Done!")