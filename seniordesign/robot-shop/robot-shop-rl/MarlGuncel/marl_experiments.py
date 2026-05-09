"""
marl_experiments.py
-------------------
MARL Ek Deneyler — Mevcut çalışmaları BOZMAZ

Yaptığı şeyler:
  1. Stabilite Testi     : IPPO'yu 5 farklı seed ile eğit → mean ± std
  2. Convergence Analizi : Mevcut training log'larından plateau tespiti
  3. Hyperparameter Test : LR ve episodes_per_update grid search (IPPO)
  4. Reward Ablation     : 3 farklı ağırlık konfigürasyonu

Dokunmadığı şeyler:
  - checkpoints_v2/     (IPPO checkpoint'leri)
  - checkpoints_mappo/  (MAPPO checkpoint'leri)
  - checkpoints_dqn/    (DQN checkpoint'leri)
  - checkpoints_a2c/    (A2C checkpoint'leri)
  - checkpoints_qmix/   (QMIX checkpoint'leri)
  - marl_env.py, ippo_agent.py, vb. (kaynak dosyalar)

Yeni dosyalar (sadece experiments/ klasörüne yazar):
  experiments/stability/     → 5 seed sonuçları
  experiments/convergence/   → convergence analizi
  experiments/hyperparam/    → grid search sonuçları
  experiments/ablation/      → reward ablation sonuçları
  experiments/summary.csv    → tüm sonuçlar
  experiments/figures/       → grafikler

Kullanım:
  python marl_experiments.py --all           # hepsini çalıştır
  python marl_experiments.py --stability     # sadece stabilite
  python marl_experiments.py --convergence   # sadece convergence
  python marl_experiments.py --hyperparam    # sadece hyperparameter
  python marl_experiments.py --ablation      # sadece ablation
"""

import os
import sys
import json
import argparse
import random
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List

# Mevcut modülleri import et
from marl_env import MarlEnvironment, SERVICES
from ippo_agent import IPPOAgent, build_agent_pool, compute_gae
from ippo_agent import HIDDEN, CLIP_EPS, VALUE_C, MAX_GRAD

# ─── Klasör yapısı ───────────────────────────────────────────────────────────

EXP_DIR = 'experiments'
for d in ['stability', 'convergence', 'hyperparam', 'ablation', 'figures']:
    os.makedirs(os.path.join(EXP_DIR, d), exist_ok=True)

# ─── SAR dataset ─────────────────────────────────────────────────────────────

METRICS    = ['num_pods', 'cpu_usage', 'mem_usage', 'request_rate', 'latency']
state_cols = [f'{s}_{m}' for s in SERVICES for m in METRICS]

df_sar        = pd.read_csv('rl_dataset_sar_normalized.csv')
initial_states = df_sar[state_cols].values.astype(np.float32)

# ─── Running Normalizer ───────────────────────────────────────────────────────

class RunningNormalizer:
    def __init__(self):
        self.mean = 0.0; self.var = 1.0; self.count = 0
    def update(self, v):
        self.count += 1
        d = v - self.mean; self.mean += d / self.count
        self.var += (v - self.mean) * d / self.count - self.var / self.count
    def normalize(self, v):
        return (v - self.mean) / (np.sqrt(max(self.var, 1e-8)) + 1e-8)

# ─── IPPO Eğitim Fonksiyonu ───────────────────────────────────────────────────

def train_ippo(
    n_episodes: int = 3000,
    episodes_per_update: int = 5,
    lr_actor: float = 1e-4,
    lr_critic: float = 5e-4,
    k_epochs: int = 8,
    entropy_c: float = 0.02,
    reward_weights: Dict[str, float] = None,
    seed: int = 42,
    verbose: bool = True,
    tag: str = "",
) -> Dict:
    """
    IPPO eğitimi — parametrize edilmiş versiyon.
    Mevcut train_ippo.py'a dokunmadan bağımsız çalışır.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env        = MarlEnvironment(seed=seed)
    normalizer = RunningNormalizer()

    # Reward config'i geçici olarak değiştir (ablation için)
    if reward_weights is not None:
        original_cfg = deepcopy(env.reward_cfg)
        for svc, w in reward_weights.items():
            if svc in env.reward_cfg:
                env.reward_cfg[svc]['weight'] = w

    # Agent pool — yeni parametrelerle
    agents = {}
    for svc in SERVICES:
        agent = IPPOAgent(
            obs_dim=env.obs_dim(svc),
            n_actions=env.action_dim(),
            service=svc,
        )
        # LR'yi güncelle
        for pg in agent.actor_opt.param_groups:
            pg['lr'] = lr_actor
        for pg in agent.critic_opt.param_groups:
            pg['lr'] = lr_critic
        agents[svc] = agent

    log = defaultdict(list)
    train_start = time.time()

    for episode in range(1, n_episodes + 1):
        idx  = random.randint(0, len(initial_states) - 1)
        obs  = env.reset(initial_state=initial_states[idx])
        ep_raw = ep_norm = 0.0

        while True:
            actions = {}
            for svc in SERVICES:
                action, log_prob, value = agents[svc].select_action(obs[svc])
                actions[svc] = action
                agents[svc].buffer.push(
                    obs=obs[svc], action=action, log_prob=log_prob,
                    reward=0.0, done=False, value=value,
                )
            next_obs, rewards, done, info = env.step(actions)
            raw_r = info['global_reward']
            normalizer.update(raw_r)
            norm_r = normalizer.normalize(raw_r)
            for svc in SERVICES:
                agents[svc].buffer.rewards[-1] = norm_r
                agents[svc].buffer.dones[-1]   = done
            ep_raw  += raw_r
            ep_norm += norm_r
            obs      = next_obs
            if done:
                break

        log['episode'].append(episode)
        log['reward_raw'].append(ep_raw)

        if episode % episodes_per_update == 0:
            for svc in SERVICES:
                agents[svc].update(last_obs=obs[svc])
                agents[svc].buffer.clear()

        if verbose and (episode % 500 == 0 or episode == 1):
            w    = min(500, episode)
            avg  = np.mean(log['reward_raw'][-w:])
            elapsed = time.time() - train_start
            print(f"    [{tag}] Ep {episode:>5}/{n_episodes} | "
                  f"Reward: {avg:>8.3f} | {elapsed:.0f}s")

    # Reward config'i geri yükle
    if reward_weights is not None:
        env.reward_cfg = original_cfg

    result = {
        'rewards':       log['reward_raw'],
        'first100_mean': float(np.mean(log['reward_raw'][:100])),
        'last100_mean':  float(np.mean(log['reward_raw'][-100:])),
        'improvement':   float(np.mean(log['reward_raw'][-100:]) -
                               np.mean(log['reward_raw'][:100])),
        'seed':          seed,
        'n_episodes':    n_episodes,
        'lr_actor':      lr_actor,
        'episodes_per_update': episodes_per_update,
        'tag':           tag,
        'time_sec':      time.time() - train_start,
    }
    return result

# ═════════════════════════════════════════════════════════════════════════════
# DENEY 1: STABİLİTE TESTİ
# ═════════════════════════════════════════════════════════════════════════════

def run_stability_test():
    print("\n" + "=" * 65)
    print("DENEY 1: STABİLİTE TESTİ (5 Farklı Seed)")
    print("=" * 65)
    print("  IPPO 5 farklı seed ile eğitiliyor.")
    print("  Amaç: Sonuçların tekrarlanabilirliğini kanıtlamak.\n")

    seeds   = [42, 123, 456, 789, 1024]
    results = []

    for seed in seeds:
        print(f"  Seed {seed} eğitiliyor...")
        r = train_ippo(
            n_episodes=3000,
            seed=seed,
            tag=f"seed_{seed}",
            verbose=True,
        )
        results.append(r)
        print(f"    → Son 100 ep: {r['last100_mean']:+.3f}\n")

    # İstatistik
    last100s = [r['last100_mean'] for r in results]
    mean_r   = np.mean(last100s)
    std_r    = np.std(last100s)
    min_r    = np.min(last100s)
    max_r    = np.max(last100s)

    print(f"\n  Stabilite Sonuçları:")
    print(f"  {'Seed':<8} {'Son 100 ep':>12} {'İyileşme':>12}")
    print(f"  {'-'*36}")
    for r in results:
        print(f"  {r['seed']:<8} {r['last100_mean']:>+12.3f} "
              f"{r['improvement']:>+12.3f}")
    print(f"  {'-'*36}")
    print(f"  {'Ortalama':<8} {mean_r:>+12.3f}")
    print(f"  {'Std':<8} {std_r:>12.3f}")
    print(f"  {'Min':<8} {min_r:>+12.3f}")
    print(f"  {'Max':<8} {max_r:>+12.3f}")
    print(f"\n  Coefficient of Variation: {std_r/abs(mean_r)*100:.1f}%")
    if std_r / abs(mean_r) < 0.20:
        print("  → Düşük CV: Sonuçlar kararlı ✓")
    else:
        print("  → Yüksek CV: Sonuçlar seed'e duyarlı △")

    # Grafik
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Stabilite Testi — IPPO (5 Seed)', fontweight='bold')

    colors = ['#2ecc71','#3498db','#e74c3c','#f39c12','#9b59b6']
    ax = axes[0]
    for r, clr in zip(results, colors):
        eps = np.array(range(1, len(r['rewards'])+1))
        # Smooth
        w   = 100
        smoothed = np.convolve(r['rewards'], np.ones(w)/w, 'valid')
        ax.plot(eps[w-1:], smoothed, color=clr, lw=1.5,
                label=f"seed={r['seed']}")
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_title('Reward Eğrileri (Smoothed, w=100)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.bar(range(len(seeds)), last100s, color=colors,
           alpha=0.8, edgecolor='black', lw=0.8)
    ax.errorbar([len(seeds)//2], [mean_r], yerr=[std_r],
                fmt='k^', capsize=8, ms=8, label=f'Mean±Std')
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f'seed={s}' for s in seeds], rotation=15)
    ax.set_title('Son 100 Episode Ortalama Reward')
    ax.set_ylabel('Reward')
    ax.axhline(mean_r, color='black', ls='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{EXP_DIR}/figures/stability_test.png', dpi=150)
    plt.close()

    # CSV kaydet
    df = pd.DataFrame(results)[['seed','first100_mean','last100_mean',
                                 'improvement','time_sec']]
    df.loc[len(df)] = {'seed': 'MEAN', 'first100_mean': np.mean([r['first100_mean'] for r in results]),
                       'last100_mean': mean_r, 'improvement': np.mean([r['improvement'] for r in results]),
                       'time_sec': np.mean([r['time_sec'] for r in results])}
    df.loc[len(df)] = {'seed': 'STD',  'first100_mean': np.std([r['first100_mean'] for r in results]),
                       'last100_mean': std_r, 'improvement': np.std([r['improvement'] for r in results]),
                       'time_sec': np.std([r['time_sec'] for r in results])}
    df.to_csv(f'{EXP_DIR}/stability/results.csv', index=False)

    print(f"\n  ✓ {EXP_DIR}/stability/results.csv")
    print(f"  ✓ {EXP_DIR}/figures/stability_test.png")
    return results

# ═════════════════════════════════════════════════════════════════════════════
# DENEY 2: CONVERGENCE ANALİZİ
# ═════════════════════════════════════════════════════════════════════════════

def run_convergence_analysis():
    print("\n" + "=" * 65)
    print("DENEY 2: CONVERGENCE ANALİZİ")
    print("=" * 65)
    print("  Mevcut training log'larından plateau tespiti.")
    print("  Yeni eğitim yapılmaz — sadece log analizi.\n")

    # Mevcut log'ları yükle
    algo_logs = {}
    log_files = {
        'IPPO':  'training_log_v2_episodes.csv',
        'DQN':   'training_log_dqn.csv',
        'A2C':   'training_log_a2c.csv',
        'MAPPO': 'training_log_mappo_episodes.csv',
        'QMIX':  'training_log_qmix.csv',
    }

    for algo, fname in log_files.items():
        if os.path.exists(fname):
            df = pd.read_csv(fname)
            if 'reward_raw' in df.columns:
                algo_logs[algo] = df['reward_raw'].values
            elif 'reward' in df.columns:
                algo_logs[algo] = df['reward'].values
            print(f"  ✓ {algo}: {len(algo_logs[algo])} episode yüklendi")
        else:
            print(f"  ✗ {algo}: {fname} bulunamadı, atlanıyor")

    if not algo_logs:
        print("  Hiç log dosyası bulunamadı!")
        return {}

    # Plateau tespiti
    def find_plateau(rewards, window=200, threshold=0.5):
        """
        Son window episode'da std < threshold ise plateau.
        Plateau'nun başladığı episode'u döndür.
        """
        rewards = np.array(rewards)
        for i in range(window, len(rewards)):
            segment = rewards[i-window:i]
            if np.std(segment) < threshold:
                return i - window
        return len(rewards)  # plateau bulunamadı

    print(f"\n  Convergence Sonuçları:")
    print(f"  {'Algoritma':<12} {'Toplam Ep':>10} {'Plateau Ep':>12} "
          f"{'İlk 100':>10} {'Son 100':>10} {'İyileşme':>10}")
    print(f"  {'-'*68}")

    conv_results = {}
    for algo, rewards in algo_logs.items():
        plateau_ep = find_plateau(rewards)
        first100   = np.mean(rewards[:100])
        last100    = np.mean(rewards[-100:])
        improvement = last100 - first100
        conv_results[algo] = {
            'total_episodes': len(rewards),
            'plateau_episode': plateau_ep,
            'first100': first100,
            'last100':  last100,
            'improvement': improvement,
            'converged': plateau_ep < len(rewards),
        }
        status = "✓ Yakınsadı" if plateau_ep < len(rewards) else "△ Yakınsamadı"
        print(f"  {algo:<12} {len(rewards):>10} {plateau_ep:>12} "
              f"{first100:>+10.2f} {last100:>+10.2f} "
              f"{improvement:>+10.2f}  {status}")

    # Grafik 1: Tüm algoritmalar smoothed reward
    colors = {'IPPO':'#2ecc71','MAPPO':'#f1c40f','DQN':'#e67e22',
               'A2C':'#9b59b6','QMIX':'#1abc9c'}

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Convergence Analizi — Tüm Algoritmalar', fontweight='bold')

    ax = axes[0]
    w  = 100
    for algo, rewards in algo_logs.items():
        eps      = np.arange(1, len(rewards)+1)
        smoothed = np.convolve(rewards, np.ones(w)/w, 'valid')
        ax.plot(eps[w-1:], smoothed, lw=2,
                color=colors.get(algo,'gray'), label=algo)
        # Plateau işaretle
        pe = conv_results[algo]['plateau_episode']
        if pe < len(rewards):
            ax.axvline(pe, color=colors.get(algo,'gray'),
                       ls=':', alpha=0.5, lw=1)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_title(f'Reward Eğrileri (Smoothed, w={w})\nDikeyler: plateau başlangıcı')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Grafik 2: Improvement bar
    ax = axes[1]
    algos = list(conv_results.keys())
    imps  = [conv_results[a]['improvement'] for a in algos]
    clrs  = [colors.get(a,'gray') for a in algos]
    bars  = ax.bar(algos, imps, color=clrs, alpha=0.8,
                   edgecolor='black', lw=0.8)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_title('Toplam İyileşme (Son 100 - İlk 100 Episode)')
    ax.set_ylabel('Reward Değişimi')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, imps):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f'{val:+.1f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{EXP_DIR}/figures/convergence_analysis.png', dpi=150)
    plt.close()

    # CSV
    pd.DataFrame(conv_results).T.to_csv(
        f'{EXP_DIR}/convergence/results.csv')

    print(f"\n  ✓ {EXP_DIR}/convergence/results.csv")
    print(f"  ✓ {EXP_DIR}/figures/convergence_analysis.png")
    return conv_results

# ═════════════════════════════════════════════════════════════════════════════
# DENEY 3: HYPERPARAMETER SENSITIVITY
# ═════════════════════════════════════════════════════════════════════════════

def run_hyperparam_sensitivity():
    print("\n" + "=" * 65)
    print("DENEY 3: HYPERPARAMETER SENSITIVITY (IPPO)")
    print("=" * 65)
    print("  LR_ACTOR ve EPISODES_PER_UPDATE grid search.")
    print("  Her kombinasyon 2000 episode ile eğitilir.\n")

    lr_values  = [5e-5, 1e-4, 3e-4]
    epu_values = [3, 5, 10]
    n_episodes = 2000   # hız için kısaltılmış

    grid_results = []
    total = len(lr_values) * len(epu_values)
    count = 0

    for lr in lr_values:
        for epu in epu_values:
            count += 1
            tag = f"lr={lr:.0e}_epu={epu}"
            print(f"  [{count}/{total}] {tag} eğitiliyor...")
            r = train_ippo(
                n_episodes=n_episodes,
                episodes_per_update=epu,
                lr_actor=lr,
                lr_critic=lr * 5,
                seed=42,
                tag=tag,
                verbose=False,
            )
            r['lr_actor'] = lr
            r['episodes_per_update'] = epu
            grid_results.append(r)
            print(f"    → Son 100 ep: {r['last100_mean']:+.3f}")

    # En iyi kombinasyon
    best = max(grid_results, key=lambda x: x['last100_mean'])
    print(f"\n  En iyi kombinasyon:")
    print(f"    LR_ACTOR           : {best['lr_actor']:.0e}")
    print(f"    EPISODES_PER_UPDATE: {best['episodes_per_update']}")
    print(f"    Son 100 ep reward  : {best['last100_mean']:+.3f}")

    # Heatmap
    matrix = np.zeros((len(lr_values), len(epu_values)))
    for r in grid_results:
        i = lr_values.index(r['lr_actor'])
        j = epu_values.index(r['episodes_per_update'])
        matrix[i, j] = r['last100_mean']

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(epu_values)))
    ax.set_yticks(range(len(lr_values)))
    ax.set_xticklabels([f'epu={e}' for e in epu_values])
    ax.set_yticklabels([f'lr={l:.0e}' for l in lr_values])
    ax.set_title('Hyperparameter Sensitivity — Son 100 Ep Reward\n(Yeşil = iyi)', fontweight='bold')
    ax.set_xlabel('Episodes per Update')
    ax.set_ylabel('Learning Rate (Actor)')
    plt.colorbar(im, label='Reward')
    for i in range(len(lr_values)):
        for j in range(len(epu_values)):
            ax.text(j, i, f'{matrix[i,j]:+.1f}',
                    ha='center', va='center', fontsize=10,
                    color='black', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{EXP_DIR}/figures/hyperparam_heatmap.png', dpi=150)
    plt.close()

    # CSV
    rows = [{'lr_actor': r['lr_actor'],
             'episodes_per_update': r['episodes_per_update'],
             'last100_mean': r['last100_mean'],
             'improvement':  r['improvement']}
            for r in grid_results]
    pd.DataFrame(rows).to_csv(f'{EXP_DIR}/hyperparam/results.csv', index=False)

    print(f"\n  ✓ {EXP_DIR}/hyperparam/results.csv")
    print(f"  ✓ {EXP_DIR}/figures/hyperparam_heatmap.png")
    return grid_results

# ═════════════════════════════════════════════════════════════════════════════
# DENEY 4: REWARD ABLATION
# ═════════════════════════════════════════════════════════════════════════════

def run_reward_ablation():
    print("\n" + "=" * 65)
    print("DENEY 4: REWARD ABLATION")
    print("=" * 65)
    print("  3 farklı reward ağırlığı konfigürasyonu ile IPPO eğitimi.")
    print("  Mevcut reward_config.json DEĞİŞTİRİLMEZ.\n")

    configs = {
        'Mevcut (baseline)': {
            'cart': 1.5, 'catalogue': 1.5, 'shipping': 2.0,
            'ratings': 0.5, 'user': 1.0, 'payment': 0.0
        },
        'Shipping düşük (0.5)': {
            'cart': 1.5, 'catalogue': 1.5, 'shipping': 0.5,
            'ratings': 0.5, 'user': 1.0, 'payment': 0.0
        },
        'Eşit ağırlık (1.0)': {
            'cart': 1.0, 'catalogue': 1.0, 'shipping': 1.0,
            'ratings': 1.0, 'user': 1.0, 'payment': 0.0
        },
    }

    abl_results = []
    for name, weights in configs.items():
        print(f"  '{name}' eğitiliyor...")
        r = train_ippo(
            n_episodes=3000,
            reward_weights=weights,
            seed=42,
            tag=name[:20],
            verbose=True,
        )
        r['config_name']   = name
        r['reward_weights'] = weights
        abl_results.append(r)
        print(f"    → Son 100 ep: {r['last100_mean']:+.3f}\n")

    # Karşılaştırma
    print(f"\n  Ablation Sonuçları:")
    print(f"  {'Konfigürasyon':<28} {'İlk 100':>10} "
          f"{'Son 100':>10} {'İyileşme':>10}")
    print(f"  {'-'*62}")
    for r in abl_results:
        print(f"  {r['config_name']:<28} "
              f"{r['first100_mean']:>+10.3f} "
              f"{r['last100_mean']:>+10.3f} "
              f"{r['improvement']:>+10.3f}")

    # Grafik
    colors = ['#2ecc71', '#e67e22', '#3498db']
    fig, ax = plt.subplots(figsize=(12, 6))
    w = 100
    for r, clr in zip(abl_results, colors):
        eps      = np.arange(1, len(r['rewards'])+1)
        smoothed = np.convolve(r['rewards'], np.ones(w)/w, 'valid')
        ax.plot(eps[w-1:], smoothed, lw=2, color=clr,
                label=r['config_name'])
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_title('Reward Ablation — Farklı Ağırlık Konfigürasyonları\n(Smoothed, w=100)',
                 fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{EXP_DIR}/figures/reward_ablation.png', dpi=150)
    plt.close()

    # CSV
    rows = [{'config': r['config_name'],
             'first100': r['first100_mean'],
             'last100':  r['last100_mean'],
             'improvement': r['improvement']}
            for r in abl_results]
    pd.DataFrame(rows).to_csv(f'{EXP_DIR}/ablation/results.csv', index=False)

    print(f"\n  ✓ {EXP_DIR}/ablation/results.csv")
    print(f"  ✓ {EXP_DIR}/figures/reward_ablation.png")
    return abl_results

# ═════════════════════════════════════════════════════════════════════════════
# ÖZET RAPOR
# ═════════════════════════════════════════════════════════════════════════════

def generate_summary(results: Dict):
    print("\n" + "=" * 65)
    print("ÖZET RAPOR")
    print("=" * 65)

    rows = []

    if 'stability' in results:
        last100s = [r['last100_mean'] for r in results['stability']]
        rows.append({
            'deney': 'Stabilite (5 seed)',
            'bulgu': f"Mean={np.mean(last100s):+.2f}, "
                     f"Std={np.std(last100s):.2f}, "
                     f"CV={np.std(last100s)/abs(np.mean(last100s))*100:.1f}%",
            'akademik_deger': 'Tekrarlanabilirlik kanıtı'
        })

    if 'convergence' in results:
        conv = results['convergence']
        converged = [a for a, r in conv.items() if r['converged']]
        rows.append({
            'deney': 'Convergence Analizi',
            'bulgu': f"Yakınsayan: {converged}",
            'akademik_deger': 'Episode yeterliliği kanıtı'
        })

    if 'hyperparam' in results:
        best = max(results['hyperparam'], key=lambda x: x['last100_mean'])
        rows.append({
            'deney': 'Hyperparameter Search',
            'bulgu': f"En iyi: lr={best['lr_actor']:.0e}, "
                     f"epu={best['episodes_per_update']}",
            'akademik_deger': 'Parametre seçimi justification'
        })

    if 'ablation' in results:
        best_abl = max(results['ablation'], key=lambda x: x['last100_mean'])
        rows.append({
            'deney': 'Reward Ablation',
            'bulgu': f"En iyi config: '{best_abl['config_name']}' "
                     f"({best_abl['last100_mean']:+.2f})",
            'akademik_deger': 'Reward tasarımı justification'
        })

    df = pd.DataFrame(rows)
    df.to_csv(f'{EXP_DIR}/summary.csv', index=False)
    print(df.to_string(index=False))
    print(f"\n  ✓ {EXP_DIR}/summary.csv")

# ═════════════════════════════════════════════════════════════════════════════
# ANA FONKSİYON
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='MARL Ek Deneyler — Mevcut dosyalara dokunmaz'
    )
    parser.add_argument('--all',         action='store_true', help='Tüm deneyleri çalıştır')
    parser.add_argument('--stability',   action='store_true', help='Stabilite testi')
    parser.add_argument('--convergence', action='store_true', help='Convergence analizi')
    parser.add_argument('--hyperparam',  action='store_true', help='Hyperparameter search')
    parser.add_argument('--ablation',    action='store_true', help='Reward ablation')
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        print("\n  Örnek: python marl_experiments.py --all")
        return

    results = {}

    total_start = time.time()

    if args.all or args.convergence:
        results['convergence'] = run_convergence_analysis()

    if args.all or args.stability:
        results['stability'] = run_stability_test()

    if args.all or args.hyperparam:
        results['hyperparam'] = run_hyperparam_sensitivity()

    if args.all or args.ablation:
        results['ablation'] = run_reward_ablation()

    if results:
        generate_summary(results)

    total_time = time.time() - total_start
    print(f"\n{'=' * 65}")
    print(f"Tüm deneyler tamamlandı: {total_time/60:.1f} dakika")
    print(f"Sonuçlar: {EXP_DIR}/ klasöründe")
    print(f"{'=' * 65}")

if __name__ == '__main__':
    main()
