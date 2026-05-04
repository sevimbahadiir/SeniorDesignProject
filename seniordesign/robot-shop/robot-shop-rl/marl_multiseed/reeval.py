"""
reeval.py
---------
Re-evaluates all trained MARL models with:
1. 50 evaluation episodes (vs 10 before)
2. Fixed evaluation states (val set, same for all seeds/algos)

This removes the starting-state bias that caused high variance
across seeds in the original evaluation.

Usage: Run from the same folder as your models/ directory.
"""

import json, os, pickle
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
EVAL_EPISODES = 50
POD_COST      = 0.05
SEEDS         = [42, 0, 123]

# All files are in the CURRENT directory — run from your marl_multiseed folder
SAR_FILE      = 'rl_dataset_sar_normalized.csv'
SCALER_FILE   = 'scaler.pkl'
REWARD_FILE   = 'reward_config.json'
DT_FILE       = 'digital_twin_best.pth'
MODELS_DIR    = 'models'
RESULTS_DIR   = 'results'

os.makedirs(RESULTS_DIR, exist_ok=True)

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

def compute_reward(state, scaler, reward_config):
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
    for svc in SERVICES:
        col_idx  = POD_IDX[svc]
        real_pod = (float(state[col_idx]) *
                    (scaler.data_max_[col_idx] - scaler.data_min_[col_idx])
                    + scaler.data_min_[col_idx])
        reward -= real_pod * POD_COST
    return float(round(reward, 4))

# ─── Environment ──────────────────────────────────────────────────────────────

class RobotShopEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, service, dt_model, sar_df, scaler,
                 reward_config, fixed_starts=None):
        super().__init__()
        self.service       = service
        self.svc_idx       = SERVICES.index(service)
        self.dt_model      = dt_model
        self.df            = sar_df.reset_index(drop=True)
        self.scaler        = scaler
        self.reward_config = reward_config
        self.fixed_starts  = fixed_starts
        self.ep_count      = 0
        self.step_count    = 0
        self.state         = None
        self.observation_space = spaces.Box(0., 1., shape=(30,), dtype=np.float32)
        self.action_space      = spaces.Discrete(3)
        self.dt_model.eval()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.fixed_starts is not None:
            idx = int(self.fixed_starts[self.ep_count % len(self.fixed_starts)])
        else:
            idx = self.np_random.integers(0, len(self.df))
        self.state      = self.df[STATE_COLS].iloc[idx].values.astype(np.float32)
        self.step_count = 0
        self.ep_count  += 1
        return self.state.copy(), {}

    def step(self, action):
        self.step_count += 1
        action_vec = np.zeros(len(SERVICES), dtype=np.float32)
        action_vec[self.svc_idx] = int(action) - 1
        inp = np.concatenate([self.state, action_vec])
        with torch.no_grad():
            t = torch.tensor(inp, dtype=torch.float32).unsqueeze(0)
            next_state = self.dt_model(t).squeeze(0).numpy()
        next_state = np.clip(next_state, 0., 1.)
        reward     = compute_reward(next_state, self.scaler, self.reward_config)
        self.state = next_state
        terminated = False
        truncated  = self.step_count >= EPISODE_LEN
        return self.state.copy(), reward, terminated, truncated, {}

    def render(self): pass

# ─── Evaluate ─────────────────────────────────────────────────────────────────

def evaluate(model_or_policy, env, n=EVAL_EPISODES):
    env.ep_count = 0
    total = 0.
    for _ in range(n):
        obs, _ = env.reset()
        done = False
        while not done:
            if model_or_policy == 'random':
                action = env.action_space.sample()
            elif model_or_policy == 'rule':
                lat    = obs[LAT_IDX[env.service]]
                col    = LAT_IDX[env.service]
                rng    = env.scaler.data_max_[col] - env.scaler.data_min_[col] + 1e-8
                cfg    = env.reward_config.get(env.service, {})
                p25    = (cfg.get('p25', 0)   - env.scaler.data_min_[col]) / rng
                p75    = (cfg.get('p75', 1e9) - env.scaler.data_min_[col]) / rng
                action = 2 if lat > p75 else (0 if lat < p25 else 1)
            else:
                action, _ = model_or_policy.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            total += r
            done = term or trunc
    return total / n

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 65)
    print("RE-EVALUATION (50 episodes, fixed val-set starts)")
    print("=" * 65)

    # Load shared resources
    print("\n[1] Loading resources...")
    sar_df = pd.read_csv(SAR_FILE)
    with open(SCALER_FILE, 'rb') as f:
        scaler = pickle.load(f)
    with open(REWARD_FILE, 'r') as f:
        reward_config = json.load(f)
    dt_model = DigitalTwin()
    dt_model.load_state_dict(torch.load(DT_FILE, map_location='cpu'))
    dt_model.eval()
    print(f"    {len(sar_df)} rows | scaler | reward_config | DT loaded")

    # Fixed evaluation states from val set
    np.random.seed(999)
    val_start    = int(len(sar_df) * 0.8)
    FIXED_STARTS = np.random.choice(
        range(val_start, len(sar_df)), size=EVAL_EPISODES, replace=False
    )
    print(f"    Fixed eval starts: {EVAL_EPISODES} states from val set")

    # Re-evaluate
    results = []
    for seed in SEEDS:
        print(f"\n{'='*55}\nSEED {seed}\n{'='*55}")
        model_dir = os.path.join(MODELS_DIR, f'seed_{seed}')

        for svc in SERVICES:
            env = RobotShopEnv(svc, dt_model, sar_df, scaler,
                               reward_config, fixed_starts=FIXED_STARTS)

            r_rand = evaluate('random', env)
            r_rule = evaluate('rule',   env)
            results.append({'seed': seed, 'algorithm': 'Random',
                            'service': svc, 'reward': round(r_rand, 4)})
            results.append({'seed': seed, 'algorithm': 'Rule-Based',
                            'service': svc, 'reward': round(r_rule, 4)})

            print(f"  {svc:<12}: rand={r_rand:.2f}  rule={r_rule:.2f}", end='')

            for algo in ['PPO', 'A2C', 'DQN']:
                model_path = os.path.join(model_dir, f'{algo}_{svc}')
                try:
                    cls   = {'PPO': PPO, 'A2C': A2C, 'DQN': DQN}[algo]
                    model = cls.load(model_path, env=env)
                    r     = evaluate(model, env)
                except Exception as e:
                    print(f"\n    WARNING: could not load {algo}_{svc}: {e}")
                    r = float('nan')
                results.append({'seed': seed, 'algorithm': algo,
                                'service': svc, 'reward': round(r, 4)})
                print(f"  {algo}={r:.2f}", end='')
            print()
            env.close()

    # Save and summarize
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, 'reeval_results.csv'), index=False)

    print("\n" + "=" * 65)
    print("SUMMARY — Mean ± Std (3 seeds, 50 episodes, fixed starts)")
    print("=" * 65)
    summary = df.groupby('algorithm')['reward'].agg(
        mean='mean', std='std', min='min', max='max'
    ).round(4).sort_values('mean', ascending=False)
    print(summary.to_string())
    summary.to_csv(os.path.join(RESULTS_DIR, 'reeval_summary.csv'))

    print("\nSeed variance (std should be < 1.0 now):")
    for algo in ['PPO', 'A2C', 'DQN', 'Random']:
        vals = [df[(df['seed'] == s) & (df['algorithm'] == algo)]['reward'].mean()
                for s in SEEDS]
        print(f"  {algo:<12}: {vals[0]:.2f} | {vals[1]:.2f} | {vals[2]:.2f}"
              f"  std={np.std(vals):.3f}")

    rand_mean = df[df['algorithm'] == 'Random']['reward'].mean()
    print(f"\nRL vs Random ({rand_mean:.3f}):")
    for algo in ['DQN', 'PPO', 'A2C']:
        m   = df[df['algorithm'] == algo]['reward'].mean()
        imp = (m - rand_mean) / abs(rand_mean) * 100
        print(f"  {algo}: {m:.3f}  {imp:+.1f}%  "
              f"{'✓ beats random' if m > rand_mean else '✗ below random'}")

    print("\nSaved: results/reeval_results.csv")
    print("       results/reeval_summary.csv")
    print("Done!")