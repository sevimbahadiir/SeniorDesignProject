"""
xai_all_algos.py
----------------
XAI Pipeline — Tum Algoritmalar (IPPO, MAPPO, DQN, A2C)

Her algoritma icin:
  Step 1: Behavior dataset toplama
  Step 2: KernelSHAP (cache varsa atlar)
  Step 3: Rule-based diagnosis
  Step 4: LLM verbalization (OpenRouter)

Son olarak:
  - Algoritmalar arasi karsilastirma grafigi
  - Karsilastirma CSV

Kullanim:
  export OPENROUTER_API_KEY="sk-or-..."
  python3 xai_all_algos.py

Cikti klasoru: xai_results/<algo>/
"""

import os, json, time, pickle, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings('ignore')

# ─── CONFIG ───────────────────────────────────────────────────────────────────
ALGORITHMS = {
    'ippo':  'checkpoints_ippo_new/ippo_new_final.pth',
    'mappo': 'checkpoints_mappo_new/mappo_new_final.pth',
    'dqn':   'checkpoints_dqn_new/dqn_new_final.pth',
    'a2c':   'checkpoints_a2c_new/a2c_new_final.pth',
}

DATA_PATH    = 'rl_dataset_sar_normalized.csv'
REWARD_CFG   = 'reward_config.json'
SCALER_PATH  = 'scaler.pkl'
N_EPISODES   = 200
N_BACKGROUND = 100
N_EXPLAIN    = 200
SEED         = 42
OPENROUTER_MODEL = 'openai/gpt-oss-20b:free'
MAX_RETRIES  = 5
RETRY_DELAY  = 15

OUTPUT_BASE  = 'xai_results'
os.makedirs(OUTPUT_BASE, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)

# ─── SABITLER ─────────────────────────────────────────────────────────────────
SERVICES = ['cart', 'catalogue', 'payment', 'shipping', 'ratings', 'user']
METRICS  = ['num_pods', 'cpu_usage', 'mem_usage', 'request_rate', 'latency']
AGENT_NEIGHBORS = {
    'cart':      ['catalogue', 'shipping'],
    'catalogue': ['ratings',   'cart'],
    'payment':   ['shipping'],
    'shipping':  ['cart',      'payment'],
    'ratings':   ['catalogue'],
    'user':      ['cart'],
}
ACTION_LABELS  = {0: 'scale_down', 1: 'keep', 2: 'scale_up'}
METRIC_LABELS  = {'num_pods':'Pods','cpu_usage':'CPU','mem_usage':'MEM',
                  'request_rate':'RPS','latency':'LAT'}
SERVICE_IDX    = {svc: i * len(METRICS) for i, svc in enumerate(SERVICES)}
SERVICE_COLORS = {'cart':'#4C72B0','catalogue':'#DD8452','payment':'#55A868',
                  'shipping':'#C44E52','ratings':'#8172B2','user':'#937860'}
ALGO_COLORS    = {'ippo':'#4C72B0','mappo':'#DD8452','dqn':'#55A868','a2c':'#C44E52'}
DIAG_LABELS = {
    'marl_optimized':             'MARL Optimized',
    'reward_excluded':            'Reward Excluded (Option C)',
    'pod_insensitive_bottleneck': 'Pod-Insensitive Bottleneck',
    'under_provisioned':          'Under-Provisioned',
    'over_provisioned':           'Over-Provisioned',
    'healthy':                    'Healthy',
    'monitoring_required':        'Monitoring Required',
}
SEV_COLORS = {'ok':'#2ecc71','info':'#3498db','warning':'#f39c12','critical':'#e74c3c'}

# ─── EVAL SONUCLARI (evaluation_summary_final.csv'den) ───────────────────────
EVAL_RESULTS = {
    'ippo':  {'rb':{'cart':0.1136,'catalogue':0.1034,'payment':0.0188,
                    'shipping':0.2912,'ratings':0.2876,'user':0.0663},
              'algo':{'cart':0.0619,'catalogue':0.0629,'payment':0.0162,
                      'shipping':0.2201,'ratings':0.2501,'user':0.0435}},
    'mappo': {'rb':{'cart':0.1136,'catalogue':0.1034,'payment':0.0188,
                    'shipping':0.2912,'ratings':0.2876,'user':0.0663},
              'algo':{'cart':0.0619,'catalogue':0.0629,'payment':0.0162,
                      'shipping':0.2201,'ratings':0.2501,'user':0.0435}},
    'dqn':   {'rb':{'cart':0.1136,'catalogue':0.1034,'payment':0.0188,
                    'shipping':0.2912,'ratings':0.2876,'user':0.0663},
              'algo':{'cart':0.0898,'catalogue':0.0752,'payment':0.0196,
                      'shipping':0.2905,'ratings':0.2748,'user':0.0660}},
    'a2c':   {'rb':{'cart':0.1136,'catalogue':0.1034,'payment':0.0188,
                    'shipping':0.2912,'ratings':0.2876,'user':0.0663},
              'algo':{'cart':0.0619,'catalogue':0.0629,'payment':0.0162,
                      'shipping':0.2201,'ratings':0.2501,'user':0.0435}},
}

# ─── YARDIMCI FONKSIYONLAR ────────────────────────────────────────────────────
def get_feature_names(svc):
    names = [f'{svc[:3].upper()}_{METRIC_LABELS[m]}' for m in METRICS]
    for nb in AGENT_NEIGHBORS[svc]:
        names += [f'{nb[:3].upper()}_{METRIC_LABELS[m]}' for m in METRICS]
    return names

def get_obs_dim(svc):
    return (1 + len(AGENT_NEIGHBORS[svc])) * len(METRICS)

def get_obs(gs, svc):
    parts = [gs[SERVICE_IDX[svc]:SERVICE_IDX[svc]+len(METRICS)]]
    for nb in AGENT_NEIGHBORS[svc]:
        parts.append(gs[SERVICE_IDX[nb]:SERVICE_IDX[nb]+len(METRICS)])
    return np.concatenate(parts)

HIDDEN = [128, 64]

class ActorNet(nn.Module):
    def __init__(self, obs_dim, n_actions=3):
        super().__init__()
        layers, prev = [], obs_dim
        for h in HIDDEN:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)
    def forward(self, obs):
        return self.net(obs)

class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions=3):
        super().__init__()
        layers, prev = [], obs_dim
        for h in HIDDEN:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)
    def forward(self, obs):
        return self.net(obs)

def load_actors(algo, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    actors = {}
    for svc in SERVICES:
        obs_dim = get_obs_dim(svc)
        if algo == 'dqn':
            net = QNet(obs_dim)
            net.load_state_dict(ckpt['agents'][svc]['q_net'])
        else:
            net = ActorNet(obs_dim)
            net.load_state_dict(ckpt['agents'][svc]['actor'])
        net.eval()
        actors[svc] = net
    return actors

# ─── VERI YUKLE ───────────────────────────────────────────────────────────────
print("="*65)
print("XAI Pipeline — Tum Algoritmalar")
print("="*65)

df_sar      = pd.read_csv(DATA_PATH)
state_cols  = [f'{s}_{m}' for s in SERVICES for m in METRICS if f'{s}_{m}' in df_sar.columns]
global_states = df_sar[state_cols].values.astype(np.float32)
split       = int(len(global_states) * 0.8)
test_states = global_states[split:]
obs_per_svc = {svc: np.array([get_obs(gs, svc) for gs in global_states])
               for svc in SERVICES}

with open(REWARD_CFG) as f:
    reward_cfg = json.load(f)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

def denorm(svc, metric, val):
    idx  = state_cols.index(f'{svc}_{metric}')
    dmin = scaler.data_min_[idx]
    dmax = scaler.data_max_[idx]
    return float(val * (dmax - dmin) + dmin)

print(f"  Dataset: {len(df_sar)} satir | Test: {len(test_states)} satir")
print(f"  Algoritmalar: {list(ALGORITHMS.keys())}")
print()

# ─── HER ALGORITMA ICIN XAI ───────────────────────────────────────────────────
all_algo_results = {}

for algo, ckpt_path in ALGORITHMS.items():
    print("="*65)
    print(f"ALGORITMA: {algo.upper()}")
    print("="*65)

    out_dir = f'{OUTPUT_BASE}/{algo}'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f'{out_dir}/shap_values', exist_ok=True)
    os.makedirs(f'{out_dir}/per_service', exist_ok=True)

    # ── STEP 1: Behavior Dataset ──────────────────────────────────────────────
    print(f"\n[{algo}] STEP 1: Behavior Dataset...")
    actors = load_actors(algo, ckpt_path)

    from marl_env import MarlEnvironment
    env = MarlEnvironment(seed=SEED)
    rng = np.random.default_rng(SEED)
    records = []

    for ep in range(N_EPISODES):
        idx = rng.integers(0, len(test_states))
        obs = env.reset(initial_state=test_states[idx])
        step = 0
        while True:
            global_state = env.global_state.copy()
            actions_dict = {}
            for svc in SERVICES:
                obs_t = torch.tensor(obs[svc], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits = actors[svc](obs_t)
                    probs  = torch.softmax(logits, dim=-1).squeeze().numpy()
                    action = int(probs.argmax())
                actions_dict[svc] = action
                record = {
                    'episode': ep, 'step': step, 'service': svc,
                    'action': action, 'action_label': ACTION_LABELS[action],
                    'prob_scale_down': float(probs[0]),
                    'prob_keep':       float(probs[1]),
                    'prob_scale_up':   float(probs[2]),
                    'confidence':      float(probs.max()),
                }
                for i, val in enumerate(obs[svc]):
                    record[f'obs_{i}'] = float(val)
                start = SERVICE_IDX[svc]
                for j, metric in enumerate(METRICS):
                    record[f'gs_{metric}'] = float(global_state[start + j])
                records.append(record)

            next_obs, rewards, done, info = env.step(actions_dict)
            for i, svc in enumerate(SERVICES):
                records[-(len(SERVICES) - i)]['reward'] = info['global_reward']
            obs = next_obs
            step += 1
            if done:
                break
        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1}/{N_EPISODES}")

    df_behavior = pd.DataFrame(records)
    df_behavior.to_csv(f'{out_dir}/xai_behavior_dataset.csv', index=False)

    obs_map = {}
    for svc in SERVICES:
        neighbors = AGENT_NEIGHBORS[svc]
        all_svcs  = [svc] + neighbors
        obs_map[svc] = {}
        for i, s in enumerate(all_svcs):
            for j, metric in enumerate(METRICS):
                col_idx = i * len(METRICS) + j
                obs_map[svc][f'obs_{col_idx}'] = {
                    'service': s, 'metric': metric,
                    'role': 'self' if s == svc else 'neighbor'
                }
    with open(f'{out_dir}/xai_obs_map.json', 'w') as f:
        json.dump(obs_map, f, indent=2)

    print(f"  OK {len(df_behavior)} kayit | obs_map kaydedildi")

    # Aksiyon dagilimi
    print(f"  {'Servis':<12} {'scale_down':>10} {'keep':>6} {'scale_up':>9} {'Guven':>7}")
    for svc in SERVICES:
        sdf   = df_behavior[df_behavior['service'] == svc]
        cnts  = sdf['action_label'].value_counts()
        total = len(sdf)
        conf  = sdf['confidence'].mean()
        sd = cnts.get('scale_down', 0)
        k  = cnts.get('keep', 0)
        su = cnts.get('scale_up', 0)
        print(f"  {svc:<12} {sd:>6}({sd/total*100:.0f}%) "
              f"{k:>3}({k/total*100:.0f}%) "
              f"{su:>5}({su/total*100:.0f}%)  {conf:.3f}")

    # ── STEP 2: KernelSHAP ────────────────────────────────────────────────────
    print(f"\n[{algo}] STEP 2: KernelSHAP...")
    all_shap_values = {}
    all_mean_abs    = {}

    for svc in SERVICES:
        obs_dim  = get_obs_dim(svc)
        npy_path = f'{out_dir}/shap_values/{svc}_shap.npy'

        if os.path.exists(npy_path):
            arr = np.load(npy_path)
            if arr.shape == (N_EXPLAIN, obs_dim, 3):
                print(f"  [{svc}] cache {arr.shape} OK")
                all_shap_values[svc] = arr
                all_mean_abs[svc]    = np.mean(np.abs(arr), axis=(0, 2))
                continue
            else:
                os.remove(npy_path)

        print(f"  [{svc}] hesaplaniyor...", flush=True)
        obs_data   = obs_per_svc[svc]
        idx_all    = np.random.permutation(len(obs_data))
        background = obs_data[idx_all[:N_BACKGROUND]]
        explain    = obs_data[idx_all[N_BACKGROUND:N_BACKGROUND+N_EXPLAIN]]

        def predict_proba(x, _svc=svc, _actors=actors):
            with torch.no_grad():
                probs = torch.softmax(
                    _actors[_svc](torch.tensor(x.astype(np.float32))), dim=-1)
            return probs.numpy()

        explainer = shap.KernelExplainer(predict_proba, background)
        shap_vals = explainer.shap_values(explain, nsamples=100, silent=True)

        if isinstance(shap_vals, list):
            shap_arr = np.stack([np.array(sv) for sv in shap_vals], axis=2)
        else:
            shap_arr = np.array(shap_vals)
            if shap_arr.ndim == 3 and shap_arr.shape[0] == 3:
                shap_arr = shap_arr.transpose(1, 2, 0)
        if shap_arr.shape != (N_EXPLAIN, obs_dim, 3):
            if shap_arr.shape == (obs_dim, 3, N_EXPLAIN):
                shap_arr = shap_arr.transpose(2, 0, 1)

        all_shap_values[svc] = shap_arr
        all_mean_abs[svc]    = np.mean(np.abs(shap_arr), axis=(0, 2))
        np.save(npy_path, shap_arr)

        feat_names = get_feature_names(svc)
        top3 = np.argsort(all_mean_abs[svc])[::-1][:3]
        print(f"    Top-3: {[feat_names[i] for i in top3]}")

    # Per-service SHAP grafigi
    for svc in SERVICES:
        feat_names = get_feature_names(svc)
        mean_abs   = all_mean_abs[svc]
        shap_arr   = all_shap_values[svc]
        top_n      = min(len(feat_names), 10)
        top_idx    = np.argsort(mean_abs)[::-1][:top_n]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'SHAP — {svc.upper()} ({algo.upper()})', fontsize=13, fontweight='bold')

        ax = axes[0]
        y_pos = np.arange(top_n)
        ax.barh(y_pos, mean_abs[top_idx[::-1]],
                color=SERVICE_COLORS[svc], alpha=0.8, edgecolor='black', lw=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feat_names[i] for i in top_idx[::-1]], fontsize=9)
        ax.set_xlabel('Mean |SHAP|')
        ax.set_title('Feature Importance')
        ax.grid(True, alpha=0.3, axis='x')

        ax = axes[1]
        x_pos  = np.arange(top_n)
        bottom = np.zeros(top_n)
        for a_idx, (a_label, a_color) in enumerate(
                zip(ACTION_LABELS.values(), ['#d62728','#aec7e8','#2ca02c'])):
            vals_a = np.mean(np.abs(shap_arr[:, top_idx, a_idx]), axis=0)
            ax.bar(x_pos, vals_a, 0.6, bottom=bottom,
                   label=a_label, color=a_color, alpha=0.8)
            bottom += vals_a
        ax.set_xticks(x_pos)
        ax.set_xticklabels([feat_names[i] for i in top_idx],
                           rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Mean |SHAP|')
        ax.set_title('Aksiyon Bazinda SHAP')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(f'{out_dir}/per_service/{svc}_shap.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Summary SHAP grafigi
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'SHAP Feature Importance — {algo.upper()}',
                 fontsize=15, fontweight='bold', y=0.98)
    gs_layout = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)
    for idx, svc in enumerate(SERVICES):
        ax         = fig.add_subplot(gs_layout[idx // 3, idx % 3])
        feat_names = get_feature_names(svc)
        mean_abs   = all_mean_abs[svc]
        top_n      = min(8, len(feat_names))
        top_idx    = np.argsort(mean_abs)[::-1][:top_n]
        plot_order = top_idx[::-1]
        ax.barh(np.arange(top_n), mean_abs[plot_order],
                color=SERVICE_COLORS[svc], alpha=0.85, edgecolor='black', lw=0.4)
        ax.set_yticks(np.arange(top_n))
        ax.set_yticklabels([feat_names[i] for i in plot_order], fontsize=7.5)
        ax.set_title(f'{svc.upper()}', fontsize=11, fontweight='bold',
                     color=SERVICE_COLORS[svc])
        ax.set_xlabel('Mean |SHAP|', fontsize=8)
        ax.grid(True, alpha=0.3, axis='x')
    plt.savefig(f'{out_dir}/xai_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    # CSV
    rows = []
    for svc in SERVICES:
        feat_names = get_feature_names(svc)
        mean_abs   = all_mean_abs[svc]
        for rank, fi in enumerate(np.argsort(mean_abs)[::-1], start=1):
            rows.append({'algo': algo, 'service': svc,
                         'feature': feat_names[fi],
                         'mean_abs_shap': round(float(mean_abs[fi]), 6),
                         'rank': rank})
    pd.DataFrame(rows).to_csv(f'{out_dir}/xai_results.csv', index=False)
    print(f"  OK SHAP grafikleri ve CSV kaydedildi")

    # ── STEP 3: Diagnosis ─────────────────────────────────────────────────────
    print(f"\n[{algo}] STEP 3: Diagnosis...")

    marl_imp = {}
    for svc in SERVICES:
        rb_lat   = denorm(svc, 'latency', EVAL_RESULTS[algo]['rb'][svc])
        algo_lat = denorm(svc, 'latency', EVAL_RESULTS[algo]['algo'][svc])
        marl_imp[svc] = {'rb_lat': rb_lat, 'algo_lat': algo_lat,
                         'improved': algo_lat < rb_lat}

    def get_svc_metrics(svc):
        svc_df = df_behavior[df_behavior['service'] == svc]
        result = {}
        for key, info in obs_map[svc].items():
            if info['role'] == 'self':
                col = key
                if col in svc_df.columns:
                    norm_val = svc_df[col].mean()
                    result[info['metric']] = {
                        'norm': norm_val,
                        'real': denorm(svc, info['metric'], norm_val)
                    }
        result['action']     = svc_df['action_label'].mode()[0]
        result['confidence'] = svc_df['confidence'].mean()
        return result

    def diagnose(svc, metrics, action):
        evidence, details = [], {}
        lat_real = metrics.get('latency',   {}).get('real', 0)
        cpu_norm = metrics.get('cpu_usage', {}).get('norm', 0)
        p50_real = reward_cfg.get(svc, {}).get('p50', 1000)
        in_reward= reward_cfg.get(svc, {}).get('in_reward', True)
        improved = marl_imp[svc]['improved']
        algo_lat = marl_imp[svc]['algo_lat']
        rb_lat   = marl_imp[svc]['rb_lat']

        details.update({
            'latency_ms':     round(lat_real, 1),
            'p50_ms':         p50_real,
            'latency_ratio':  round(lat_real / p50_real, 2) if p50_real > 0 else 0,
            'cpu_norm':       round(cpu_norm, 3),
            'action':         action,
            'in_reward':      in_reward,
            'algo_lat_ms':    round(algo_lat, 1),
            'rb_lat_ms':      round(rb_lat, 1),
            'pct_improvement':round((rb_lat - algo_lat) / rb_lat * 100, 1) if rb_lat > 0 else 0,
            'confidence':     round(metrics.get('confidence', 0), 3),
            'pod_count':      round(metrics.get('num_pods', {}).get('real', 0), 1),
        })

        if not in_reward:
            evidence += ['Reward disi servis (Option C)', f'Latency: {lat_real:.0f}ms']
            return 'reward_excluded', 'info', evidence, details

        pct = details['pct_improvement']
        if improved and pct > 10.0:
            evidence += [f'{algo.upper()} iyilestirme: {rb_lat:.0f}ms -> {algo_lat:.0f}ms ({pct:+.1f}%)',
                         f'SLO ihlal: %13.2 (Rule-based: %34.9)',
                         f'Politika: {action} (guven: {details["confidence"]:.2f})']
            return 'marl_optimized', 'ok', evidence, details

        if svc == 'shipping' and action == 'scale_up' and cpu_norm < 0.65:
            evidence += [f'Latency {lat_real:.0f}ms > p50 ({p50_real:.0f}ms)',
                         'I/O-bound bottleneck']
            return 'pod_insensitive_bottleneck', 'critical', evidence, details

        if lat_real > 1.2 * p50_real and action in ('keep', 'scale_down'):
            evidence.append(f'Latency {lat_real:.0f}ms > 1.2xp50 — reaktif degil')
            return 'under_provisioned', 'warning', evidence, details

        evidence.append(f'Latency {lat_real:.0f}ms, p50 {p50_real:.0f}ms')
        return 'monitoring_required', 'info', evidence, details

    diag_report = {}
    diag_rows   = []
    for svc in SERVICES:
        m       = get_svc_metrics(svc)
        action  = m['action']
        diag, sev, ev, det = diagnose(svc, m, action)
        diag_report[svc] = {'diagnosis': diag, 'severity': sev,
                             'action': action, 'confidence': round(m['confidence'], 3),
                             'evidence': ev, 'details': det}
        pct = det['pct_improvement']
        print(f"  {svc:<12} {DIAG_LABELS.get(diag, diag):<28} {sev:<6} "
              f"{det['latency_ms']:>9.1f}ms  {pct:>+7.1f}%")
        diag_rows.append({'algo': algo, 'service': svc, 'diagnosis': diag,
                          'severity': sev, 'action': action,
                          'latency_ms': det['latency_ms'],
                          'p50_ms': det['p50_ms'],
                          'pct_improvement': det['pct_improvement'],
                          'confidence': det['confidence']})

    with open(f'{out_dir}/xai_diagnosis_report.json', 'w', encoding='utf-8') as f:
        json.dump(diag_report, f, indent=2, ensure_ascii=False)
    pd.DataFrame(diag_rows).to_csv(f'{out_dir}/xai_diagnosis_summary.csv', index=False)

    # Diagnosis grafigi
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'Diagnosis Engine — {algo.upper()} (Yeni DT)',
                 fontsize=13, fontweight='bold')
    df_d = pd.DataFrame(diag_rows)
    ax = axes[0]
    x  = np.arange(len(SERVICES))
    w  = 0.35
    clrs = [SEV_COLORS[diag_report[s]['severity']] for s in SERVICES]
    ax.bar(x - w/2, df_d['latency_ms'], w, color=clrs, alpha=0.85,
           label='Latency (ms)', edgecolor='black', lw=0.8)
    ax.bar(x + w/2, df_d['p50_ms'],     w, color='lightgray', alpha=0.85,
           label='p50 (ms)', edgecolor='black', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(SERVICES, rotation=15)
    ax.set_yscale('log')
    ax.set_ylabel('Latency (ms, log)')
    ax.set_title(f'{algo.upper()} Latency vs p50')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    conf_vals   = [diag_report[s]['confidence'] for s in SERVICES]
    diag_labels = [DIAG_LABELS.get(diag_report[s]['diagnosis'], '?') for s in SERVICES]
    bars = ax.barh(SERVICES, conf_vals, color=clrs, alpha=0.85, edgecolor='black', lw=0.8)
    ax.set_xlabel('Karar Guveni')
    ax.set_title('Diagnosis & Guven')
    ax.set_xlim(0, 1.35)
    ax.grid(True, alpha=0.3, axis='x')
    for bar, label, val in zip(bars, diag_labels, conf_vals):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{label}\n({val:.2f})', va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/xai_diagnosis_plot.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── STEP 4: LLM ───────────────────────────────────────────────────────────
    print(f"\n[{algo}] STEP 4: LLM Verbalization...")
    api_key = os.getenv("OPENROUTER_API_KEY")

    shap_rows = pd.read_csv(f'{out_dir}/xai_results.csv')
    def load_feats(svc, top_k=5):
        sdf = shap_rows[shap_rows['service']==svc].sort_values('mean_abs_shap', ascending=False).head(top_k)
        return [{'feature': r['feature'], 'score': round(float(r['mean_abs_shap']),6),
                 'rank': int(r['rank'])} for _, r in sdf.iterrows()]

    def build_prompt(svc, ev):
        top_feats     = "\n".join([f"- {f['feature']} (SHAP: {f['score']:.6f}, rank {f['rank']})"
                                   for f in ev['top_features']])
        rule_ev       = "\n".join([f"- {e}" for e in ev['rule_evidence']])
        return f"""Explain this XAI result for {algo.upper()} MARL autoscaling (Kubernetes Robot Shop).
Only interpret provided evidence. Do not invent diagnoses.

Service: {svc} | Diagnosis: {ev['diagnosis_label']} | Action: {ev['action']} (conf: {ev['confidence']})
Latency: {ev['latency_ms']} ms | p50: {ev['p50_ms']} ms | Improvement vs rule-based: {ev['pct_improvement']:+.1f}%

Rule Evidence:
{rule_ev}

Top KernelSHAP Features:
{top_feats}

Write 3-4 sentences for an academic paper.""".strip()

    def call_openrouter(prompt):
        if not api_key:
            return None, "no_api_key"
        headers = {"Authorization": f"Bearer {api_key}",
                   "Content-Type":  "application/json",
                   "HTTP-Referer":  "http://localhost"}
        payload = {"model": OPENROUTER_MODEL,
                   "messages": [
                       {"role":"system","content":"Academic technical writer. Only interpret evidence."},
                       {"role":"user","content":prompt}],
                   "temperature": 0.2, "max_tokens": 500}
        for attempt in range(1, MAX_RETRIES+1):
            try:
                r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                                  headers=headers, json=payload, timeout=60)
                if r.status_code == 200:
                    return r.json()['choices'][0]['message']['content'].strip(), f"openrouter/{OPENROUTER_MODEL}"
                elif r.status_code == 429:
                    wait = RETRY_DELAY * attempt
                    print(f"    429 — {wait}s bekleniyor ({attempt}/{MAX_RETRIES})...")
                    time.sleep(wait)
                else:
                    return None, f"error_{r.status_code}"
            except Exception as e:
                time.sleep(RETRY_DELAY)
        return None, "max_retries"

    def template(svc, ev):
        top = ', '.join([f["feature"] for f in ev['top_features'][:3]])
        if ev['diagnosis'] == 'marl_optimized':
            return (f"The {svc} service is classified as {ev['diagnosis_label']}. "
                    f"The {algo.upper()} policy selected '{ev['action']}' with confidence {ev['confidence']:.2f}, "
                    f"improving latency by {ev['pct_improvement']:.1f}% ({ev['rb_lat_ms']:.0f}->{ev['algo_lat_ms']:.0f} ms). "
                    f"Top KernelSHAP features: {top}.")
        if ev['diagnosis'] == 'reward_excluded':
            return (f"The {svc} service is excluded from the reward function (Option C). "
                    f"The {algo.upper()} agent selected '{ev['action']}' with confidence {ev['confidence']:.2f}; "
                    f"latency was {ev['latency_ms']:.0f} ms.")
        return (f"The {svc} service: {ev['diagnosis_label']}. "
                f"Action: '{ev['action']}' (conf: {ev['confidence']:.2f}). "
                f"Latency: {ev['latency_ms']:.0f}/{ev['p50_ms']:.0f} ms. Features: {top}.")

    explanations = {}
    for svc in SERVICES:
        det = diag_report[svc]['details']
        ev  = {
            'service':         svc,
            'diagnosis':       diag_report[svc]['diagnosis'],
            'diagnosis_label': DIAG_LABELS.get(diag_report[svc]['diagnosis'], ''),
            'severity':        diag_report[svc]['severity'],
            'action':          diag_report[svc]['action'],
            'confidence':      diag_report[svc]['confidence'],
            'latency_ms':      det.get('latency_ms', 0),
            'p50_ms':          det.get('p50_ms', 0),
            'pct_improvement': det.get('pct_improvement', 0),
            'rb_lat_ms':       det.get('rb_lat_ms', 0),
            'algo_lat_ms':     det.get('algo_lat_ms', 0),
            'rule_evidence':   diag_report[svc]['evidence'],
            'top_features':    load_feats(svc),
        }
        print(f"  [{svc}] isleniyor...")
        if api_key:
            llm_text, method = call_openrouter(build_prompt(svc, ev))
        else:
            llm_text, method = None, "no_api_key"

        explanation = llm_text if llm_text else template(svc, ev)
        if not llm_text:
            method = "template_fallback"

        explanations[svc] = {'diagnosis': ev['diagnosis_label'], 'severity': ev['severity'],
                             'method': method, 'explanation': explanation,
                             'top_features': ev['top_features']}
        print(f"    method: {method}")
        if api_key and llm_text:
            time.sleep(3)

    with open(f'{out_dir}/xai_evidence_guided_explanations.json', 'w', encoding='utf-8') as f:
        json.dump(explanations, f, indent=2, ensure_ascii=False)

    lines = [
        "="*80, f"XAI EVIDENCE-GUIDED REPORT — {algo.upper()}",
        f"Model: {OPENROUTER_MODEL}", "="*80, "",
        "Diagnoses: rule-based engine. LLM: interprets evidence only.", "",
    ]
    icon_map = {'ok':'[OK]','info':'[INFO]','warning':'[WARN]','critical':'[CRIT]'}
    for svc, item in explanations.items():
        lines += [f"{icon_map.get(item['severity'],'[?]')} {svc.upper()} — {item['diagnosis']}",
                  "-"*60, f"Method: {item['method']}", "", item['explanation'], "",
                  "Top SHAP:"]
        for f2 in item['top_features']:
            lines.append(f"  {f2['rank']}. {f2['feature']} (score: {f2['score']:.6f})")
        lines.append("")
    lines += ["="*80]
    with open(f'{out_dir}/xai_evidence_guided_report.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"  OK Rapor kaydedildi")

    all_algo_results[algo] = {
        'behavior':     df_behavior,
        'shap':         all_mean_abs,
        'diagnosis':    diag_report,
        'explanations': explanations,
    }

# ─── KARSILASTIRMALI GRAFIKLER ────────────────────────────────────────────────
print("\n"+"="*65)
print("KARSILASTIRMA GRAFIKLERI")
print("="*65)

# 1. Top feature per algo per service
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Top SHAP Feature — Algoritma & Servis Karsilastirmasi',
             fontsize=14, fontweight='bold')
for idx, svc in enumerate(SERVICES):
    ax = axes[idx // 3, idx % 3]
    feat_names = get_feature_names(svc)
    data = {}
    for algo in ALGORITHMS:
        top_fi = np.argmax(all_algo_results[algo]['shap'][svc])
        data[algo] = {
            'feature': feat_names[top_fi],
            'score':   all_algo_results[algo]['shap'][svc][top_fi]
        }
    algos  = list(data.keys())
    scores = [data[a]['score'] for a in algos]
    feats  = [data[a]['feature'] for a in algos]
    colors = [ALGO_COLORS[a] for a in algos]
    bars   = ax.bar(algos, scores, color=colors, alpha=0.85, edgecolor='black', lw=0.5)
    for bar, feat, score in zip(bars, feats, scores):
        ax.text(bar.get_x() + bar.get_width()/2, score + max(scores)*0.01,
                f'{feat}\n{score:.4f}', ha='center', va='bottom', fontsize=7)
    ax.set_title(f'{svc.upper()}', fontsize=11, fontweight='bold',
                 color=SERVICE_COLORS[svc])
    ax.set_ylabel('Max SHAP Score')
    ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{OUTPUT_BASE}/comparison_top_features.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  OK {OUTPUT_BASE}/comparison_top_features.png")

# 2. Karar guveni karsilastirmasi
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(SERVICES))
w = 0.18
for i, algo in enumerate(ALGORITHMS):
    confs = [all_algo_results[algo]['diagnosis'][svc]['confidence'] for svc in SERVICES]
    ax.bar(x + i*w - 1.5*w, confs, w,
           label=algo.upper(), color=ALGO_COLORS[algo], alpha=0.85, edgecolor='black', lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels(SERVICES)
ax.set_ylabel('Karar Guveni')
ax.set_title('Algoritma Bazinda Karar Guveni Karsilastirmasi')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{OUTPUT_BASE}/comparison_confidence.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  OK {OUTPUT_BASE}/comparison_confidence.png")

# 3. Pct improvement karsilastirmasi
fig, ax = plt.subplots(figsize=(14, 6))
for i, algo in enumerate(ALGORITHMS):
    pcts = [all_algo_results[algo]['diagnosis'][svc]['details']['pct_improvement']
            for svc in SERVICES]
    ax.bar(x + i*w - 1.5*w, pcts, w,
           label=algo.upper(), color=ALGO_COLORS[algo], alpha=0.85, edgecolor='black', lw=0.5)
ax.axhline(0, color='gray', ls='--', alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(SERVICES)
ax.set_ylabel('Latency Iyilesme (%)')
ax.set_title('Algoritma Bazinda Latency Iyilesmesi vs Rule-based')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{OUTPUT_BASE}/comparison_improvement.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  OK {OUTPUT_BASE}/comparison_improvement.png")

# 4. Karsilastirma CSV
comp_rows = []
for algo in ALGORITHMS:
    for svc in SERVICES:
        diag = all_algo_results[algo]['diagnosis'][svc]
        det  = diag['details']
        top_fi = np.argmax(all_algo_results[algo]['shap'][svc])
        feat_names = get_feature_names(svc)
        comp_rows.append({
            'algo':            algo,
            'service':         svc,
            'diagnosis':       diag['diagnosis'],
            'severity':        diag['severity'],
            'action':          diag['action'],
            'confidence':      diag['confidence'],
            'latency_ms':      det['latency_ms'],
            'p50_ms':          det['p50_ms'],
            'pct_improvement': det['pct_improvement'],
            'top1_feature':    feat_names[top_fi],
            'top1_shap':       round(all_algo_results[algo]['shap'][svc][top_fi], 6),
        })
pd.DataFrame(comp_rows).to_csv(f'{OUTPUT_BASE}/comparison_all.csv', index=False)
print(f"  OK {OUTPUT_BASE}/comparison_all.csv")

# ─── FINAL OZET ───────────────────────────────────────────────────────────────
print("\n"+"="*65)
print("XAI PIPELINE TAMAMLANDI")
print("="*65)
print(f"\nKlasör yapisi:")
print(f"  {OUTPUT_BASE}/")
for algo in ALGORITHMS:
    print(f"    {algo}/")
    print(f"      xai_behavior_dataset.csv")
    print(f"      xai_results.csv")
    print(f"      xai_summary.png")
    print(f"      xai_diagnosis_report.json")
    print(f"      xai_diagnosis_plot.png")
    print(f"      xai_evidence_guided_report.txt")
    print(f"      per_service/*.png")
    print(f"      shap_values/*.npy")
print(f"    comparison_top_features.png")
print(f"    comparison_confidence.png")
print(f"    comparison_improvement.png")
print(f"    comparison_all.csv")
print("\nDone!")
