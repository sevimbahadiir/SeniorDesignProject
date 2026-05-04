"""
mappo_train.py — MAPPO for Robot Shop autoscaling.
CTDE: Centralized Training, Decentralized Execution.
Uses Ray RLlib old API stack for stability.
"""

import json, os, pickle, time
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
import torch, torch.nn as nn
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
from gymnasium import spaces

os.makedirs('models/mappo', exist_ok=True)
os.makedirs('results/mappo', exist_ok=True)
os.makedirs('plots', exist_ok=True)

SERVICES   = ['cart', 'catalogue', 'payment', 'shipping', 'ratings', 'user']
METRICS    = ['num_pods', 'cpu_usage', 'mem_usage', 'request_rate', 'latency']
STATE_COLS = [f'{s}_{m}' for s in SERVICES for m in METRICS]
LAT_IDX    = {s: i*5+4 for i,s in enumerate(SERVICES)}
POD_IDX    = {s: i*5+0 for i,s in enumerate(SERVICES)}
EPISODE_LEN   = 10
EVAL_EPISODES = 50
SEEDS         = [42, 0, 123]
POD_COST      = 0.05
OBS_DIM       = 36

class DigitalTwin(nn.Module):
    def __init__(self):
        super().__init__()
        layers, prev = [], 36
        for h in [256, 256, 128]:
            layers += [nn.Linear(prev,h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers += [nn.Linear(prev, 30), nn.Sigmoid()]
        self.network = nn.Sequential(*layers)
    def forward(self, x): return self.network(x)

def compute_reward(state, scaler, cfg):
    r = 0.0
    for svc, c in cfg.items():
        if not c['in_reward']: continue
        idx = LAT_IDX[svc]
        lat = float(state[idx])*(scaler.data_max_[idx]-scaler.data_min_[idx])+scaler.data_min_[idx]
        if lat > 0: r += np.tanh(1.0 - lat/c['p50'])*c['weight']
    pods = sum(float(state[POD_IDX[s]])*(scaler.data_max_[POD_IDX[s]]-scaler.data_min_[POD_IDX[s]])+scaler.data_min_[POD_IDX[s]] for s in SERVICES)
    return float(round(r - pods*POD_COST, 4))

# Global shared resources (loaded once, used in env)
_SHARED = {}

class RobotShopMAEnv(MultiAgentEnv):
    def __init__(self, env_config=None):
        super().__init__()
        self.df     = _SHARED['df']
        self.dt     = _SHARED['dt']
        self.scaler = _SHARED['scaler']
        self.rcfg   = _SHARED['rcfg']
        self.agents = SERVICES
        self.possible_agents = SERVICES
        self.observation_space = spaces.Box(-1., 1., (OBS_DIM,), np.float32)
        self.action_space      = spaces.Discrete(3)
        self.observation_spaces = {s: self.observation_space for s in SERVICES}
        self.action_spaces      = {s: self.action_space for s in SERVICES}
        self.state = None
        self.last_actions = {s: 0 for s in SERVICES}
        self.steps = 0

    def _obs(self, svc):
        own   = np.array([self.last_actions[svc]], dtype=np.float32)
        other = np.array([self.last_actions[s] for s in SERVICES if s!=svc], dtype=np.float32)
        return np.concatenate([self.state, own, other])

    def reset(self, *, seed=None, options=None):
        rng = np.random.default_rng(seed)
        idx = int(rng.integers(0, len(self.df)))
        self.state = self.df[STATE_COLS].iloc[idx].values.astype(np.float32)
        self.last_actions = {s: 0 for s in SERVICES}
        self.steps = 0
        return {s: self._obs(s) for s in SERVICES}, {}

    def step(self, action_dict):
        self.steps += 1
        av = np.zeros(len(SERVICES), dtype=np.float32)
        for s, a in action_dict.items():
            v = int(a)-1
            av[SERVICES.index(s)] = v
            self.last_actions[s]  = v
        inp = np.concatenate([self.state, av])
        with torch.no_grad():
            ns = self.dt(torch.tensor(inp, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
        ns = np.clip(ns, 0., 1.)
        r  = compute_reward(ns, self.scaler, self.rcfg)
        self.state = ns
        done  = self.steps >= EPISODE_LEN
        obs   = {s: self._obs(s) for s in SERVICES}
        rews  = {s: r for s in SERVICES}
        terms = {s: False for s in SERVICES}; terms['__all__'] = False
        truncs= {s: done  for s in SERVICES}; truncs['__all__'] = done
        return obs, rews, terms, truncs, {s:{} for s in SERVICES}

def eval_mappo(algo, sar_df, dt, scaler, rcfg, fixed_starts, n=EVAL_EPISODES):
    total = 0.
    for ep in range(n):
        idx   = int(fixed_starts[ep % len(fixed_starts)])
        state = sar_df[STATE_COLS].iloc[idx].values.astype(np.float32)
        last  = {s: 0 for s in SERVICES}
        ep_r  = 0.
        for _ in range(EPISODE_LEN):
            av = np.zeros(len(SERVICES), dtype=np.float32)
            for s in SERVICES:
                obs = np.concatenate([state,
                      np.array([last[s]], dtype=np.float32),
                      np.array([last[x] for x in SERVICES if x!=s], dtype=np.float32)])
                a = algo.compute_single_action(obs, policy_id=s)
                av[SERVICES.index(s)] = int(a)-1
                last[s] = int(a)-1
            inp = np.concatenate([state, av])
            with torch.no_grad():
                ns = dt(torch.tensor(inp, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
            ns    = np.clip(ns, 0., 1.)
            ep_r += compute_reward(ns, scaler, rcfg)
            state = ns
        total += ep_r
    return total/n

def eval_baseline(policy, sar_df, dt, scaler, rcfg, fixed_starts, n=EVAL_EPISODES):
    np.random.seed(999); total = 0.
    for ep in range(n):
        idx   = int(fixed_starts[ep % len(fixed_starts)])
        state = sar_df[STATE_COLS].iloc[idx].values.astype(np.float32)
        ep_r  = 0.
        for _ in range(EPISODE_LEN):
            av = np.zeros(len(SERVICES), dtype=np.float32)
            for s in SERVICES:
                lat = float(state[LAT_IDX[s]])
                mn  = scaler.data_min_[LAT_IDX[s]]
                mx  = scaler.data_max_[LAT_IDX[s]]
                c   = rcfg.get(s, {})
                p25 = (c.get('p25',0)-mn)/(mx-mn+1e-8)
                p75 = (c.get('p75',1e9)-mn)/(mx-mn+1e-8)
                if policy=='random':
                    av[SERVICES.index(s)] = float(np.random.choice([-1,0,1]))
                else:
                    av[SERVICES.index(s)] = 1. if lat>p75 else (-1. if lat<p25 else 0.)
            inp = np.concatenate([state, av])
            with torch.no_grad():
                ns = dt(torch.tensor(inp, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
            ns    = np.clip(ns, 0., 1.)
            ep_r += compute_reward(ns, scaler, rcfg)
            state = ns
        total += ep_r
    return total/n

if __name__ == '__main__':
    print("="*65)
    print("MAPPO TRAINING — Robot Shop Autoscaling")
    print("="*65)

    # Load into global shared dict (avoids serialization issues with Ray)
    print("\n[1] Loading resources...")
    sar_df = pd.read_csv('rl_dataset_sar_normalized.csv')
    with open('scaler.pkl','rb') as f: scaler=pickle.load(f)
    with open('reward_config.json','r') as f: rcfg=json.load(f)
    dt = DigitalTwin()
    dt.load_state_dict(torch.load('digital_twin_best.pth', map_location='cpu'))
    dt.eval()

    _SHARED['df']     = sar_df.reset_index(drop=True)
    _SHARED['dt']     = dt
    _SHARED['scaler'] = scaler
    _SHARED['rcfg']   = rcfg
    print(f"    {len(sar_df)} rows loaded")

    np.random.seed(999)
    val_start    = int(len(sar_df)*0.8)
    fixed_starts = np.random.choice(range(val_start,len(sar_df)), size=EVAL_EPISODES, replace=False)

    print("\n[2] Baselines...")
    r_rand = eval_baseline('random', sar_df, dt, scaler, rcfg, fixed_starts)
    r_rule = eval_baseline('rule',   sar_df, dt, scaler, rcfg, fixed_starts)
    print(f"    Random:     {r_rand:.4f}")
    print(f"    Rule-Based: {r_rule:.4f}")

    print("\n[3] Initializing Ray...")
    ray.init(ignore_reinit_error=True, num_cpus=2, log_to_driver=False)

    results = []
    wall_start = time.time()

    for seed in SEEDS:
        print(f"\n{'='*55}\nSEED {seed}\n{'='*55}")

        config = (
            PPOConfig()
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .environment(env=RobotShopMAEnv)
            .framework('torch')
            .multi_agent(
                policies={s: PolicySpec(
                    observation_space=spaces.Box(-1.,1.,(OBS_DIM,),np.float32),
                    action_space=spaces.Discrete(3))
                    for s in SERVICES},
                policy_mapping_fn=lambda agent_id, *a, **k: agent_id,
            )
            .training(
                lr=3e-4, gamma=0.99, lambda_=0.95,
                clip_param=0.2, entropy_coeff=0.01,
                num_epochs=10, minibatch_size=64,
                train_batch_size=600, use_gae=True,
            )
            .env_runners(num_env_runners=0)
            .resources(num_gpus=0)
            .debugging(seed=seed)
        )

        algo = config.build()
        print(f"  Training (150 iterations)...")
        t0 = time.time()

        for i in range(150):
            res = algo.train()
            if (i+1) % 30 == 0:
                # Try multiple possible reward keys
                r = (res.get('episode_reward_mean') or
                     res.get('sampler_results', {}).get('episode_reward_mean') or
                     res.get('env_runners', {}).get('episode_reward_mean') or
                     float('nan'))
                print(f"    Iter {i+1:>3}/150  reward={r:.4f}  time={time.time()-t0:.0f}s")

        ckpt = f'models/mappo/seed_{seed}'
        os.makedirs(ckpt, exist_ok=True)
        algo.save(ckpt)

        r_mappo = eval_mappo(algo, sar_df, dt, scaler, rcfg, fixed_starts)
        print(f"  MAPPO eval reward: {r_mappo:.4f}")
        algo.stop()

        results.append({'seed':seed,'algorithm':'MAPPO',     'reward':round(r_mappo,4)})
        results.append({'seed':seed,'algorithm':'Random',    'reward':round(r_rand,4)})
        results.append({'seed':seed,'algorithm':'Rule-Based','reward':round(r_rule,4)})

    ray.shutdown()

    df_res = pd.DataFrame(results)
    df_res.to_csv('results/mappo/all_results.csv', index=False)

    print("\n"+"="*65)
    print("SUMMARY (3 seeds, 50 episodes, fixed val states)")
    print("="*65)
    summary = df_res.groupby('algorithm')['reward'].agg(
        mean='mean', std='std', min='min', max='max'
    ).round(4).sort_values('mean', ascending=False)
    print(summary.to_string())
    summary.to_csv('results/mappo/summary.csv')

    vals = [df_res[(df_res['seed']==s)&(df_res['algorithm']=='MAPPO')]['reward'].values[0] for s in SEEDS]
    print(f"\nSeed variance: {vals[0]:.2f}|{vals[1]:.2f}|{vals[2]:.2f}  std={np.std(vals):.3f}")

    rand_m  = df_res[df_res['algorithm']=='Random']['reward'].mean()
    mappo_m = df_res[df_res['algorithm']=='MAPPO']['reward'].mean()
    imp = (mappo_m-rand_m)/abs(rand_m)*100
    print(f"MAPPO vs Random: {imp:+.1f}%  {'✓' if mappo_m>rand_m else '✗'}")

    # Plot
    fig, ax = plt.subplots(figsize=(8,6))
    means = summary['mean']; stds = summary['std']
    colors = ['#2196F3','#9E9E9E','#607D8B']
    bars = ax.bar(summary.index, means, yerr=stds, capsize=6,
                  color=colors[:len(means)], edgecolor='white',
                  linewidth=1.5, error_kw={'linewidth':2})
    ax.set_title('MAPPO — Coordinated Multi-Agent PPO\n(Mean±Std, 3 seeds, 50 episodes)', fontweight='bold')
    ax.set_ylabel('Mean Episode Reward')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.grid(axis='y', alpha=0.3)
    for bar, val, std in zip(bars, means, stds):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+std+0.2,
                f'{val:.2f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/mappo_comparison.png', dpi=150)
    plt.close()

    elapsed = time.time()-wall_start
    print(f"\nTotal: {elapsed/3600:.1f}h {(elapsed%3600)/60:.0f}min")
    print("Done!")