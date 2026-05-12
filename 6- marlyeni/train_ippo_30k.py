"""
train_ippo.py  (v2)
-------------------
Geliştirilmiş IPPO Training Loop

v1'deki sorunlar ve çözümleri:
─────────────────────────────────────────────────────────────────
SORUN 1: Buffer çok küçük (10 transition)
  ÇÖZÜM: EPISODES_PER_UPDATE = 5 → 50 transition/update

SORUN 2: LR çok yüksek (3e-4)
  ÇÖZÜM: LR_ACTOR = 1e-4, LR_CRITIC = 5e-4

SORUN 3: Reward skalası (ham ~35-38)
  ÇÖZÜM: Running mean/std ile reward normalizasyonu

SORUN 4: K_EPOCHS = 4
  ÇÖZÜM: K_EPOCHS = 8 (ippo_agent.py'da güncellendi)

SORUN 5: Entropy salınımı
  ÇÖZÜM: ENTROPY_C = 0.02 (ippo_agent.py'da güncellendi)
─────────────────────────────────────────────────────────────────
"""

import os
import random
import time
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

from marl_env import MarlEnvironment, SERVICES
from ippo_agent import build_agent_pool

# ─── Hiperparametreler ────────────────────────────────────────────────────────

N_EPISODES          = 3000  # yeni DT (MAE=0.032, R2=0.945)
EPISODES_PER_UPDATE = 5       # 5 ep × 10 adım = 50 transition/update
SAVE_INTERVAL       = 500
LOG_INTERVAL        = 50
SEED                = 42
CHECKPOINT_DIR      = 'checkpoints_ippo_new'
REWARD_NORM_EPS     = 1e-8

# ─── Running Reward Normalizer ────────────────────────────────────────────────

class RunningNormalizer:
    """
    Welford online algoritması ile reward normalizasyonu.

    Neden gerekli?
      Ham reward ~35 → CriticLoss ~170 → unstable gradient.
      Normalize edilmiş reward ~[-3,+3] → CriticLoss hedef <5.
    """
    def __init__(self):
        self.mean  = 0.0
        self.var   = 1.0
        self.count = 0

    def update(self, value):
        self.count += 1
        delta       = value - self.mean
        self.mean  += delta / self.count
        delta2      = value - self.mean
        self.var   += (delta * delta2 - self.var) / self.count

    def normalize(self, value):
        std = np.sqrt(max(self.var, REWARD_NORM_EPS))
        return (value - self.mean) / (std + REWARD_NORM_EPS)

# ─── Setup ───────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

env        = MarlEnvironment(seed=SEED)
agents     = build_agent_pool(env)
normalizer = RunningNormalizer()

print("=" * 65)
print("IPPO Training v2")
print("=" * 65)
print(f"  Episodes/update : {EPISODES_PER_UPDATE} ep = {EPISODES_PER_UPDATE*10} transition")
print(f"  Reward norm     : RunningNormalizer (Welford)")
print(f"  LR actor/critic : 1e-4 / 5e-4")
print(f"  K_epochs        : 8  |  Entropy coeff : 0.02")
print("=" * 65)

# ─── SAR Dataset ─────────────────────────────────────────────────────────────

df_sar     = pd.read_csv('rl_dataset_sar_normalized.csv')
state_cols = [f'{s}_{m}' for s in SERVICES
              for m in ['num_pods','cpu_usage','mem_usage','request_rate','latency']]
initial_states = df_sar[state_cols].values.astype(np.float32)
print(f"\n  {len(initial_states)} başlangıç state'i hazır.\n")

# ─── Log ─────────────────────────────────────────────────────────────────────

log = defaultdict(list)

# ─── Training Loop ────────────────────────────────────────────────────────────

print("Eğitim başlıyor...\n")
train_start  = time.time()
update_count = 0

for episode in range(1, N_EPISODES + 1):

    # 1. Reset
    idx  = random.randint(0, len(initial_states) - 1)
    obs  = env.reset(initial_state=initial_states[idx])

    ep_raw  = 0.0
    ep_norm = 0.0

    # 2. Episode
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

        raw_r  = info['global_reward']
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
    log['reward_norm'].append(ep_norm)

    # 3. Update (her EPISODES_PER_UPDATE episode'da bir)
    if episode % EPISODES_PER_UPDATE == 0:
        update_count += 1
        ms = {}
        for svc in SERVICES:
            ms[svc] = agents[svc].update(last_obs=obs[svc])
            agents[svc].buffer.clear()

        log['update_ep'].append(episode)
        log['avg_actor_loss'].append(np.mean([ms[s]['actor_loss']  for s in SERVICES]))
        log['avg_critic_loss'].append(np.mean([ms[s]['critic_loss'] for s in SERVICES]))
        log['avg_entropy'].append(np.mean([ms[s]['entropy']     for s in SERVICES]))
        log['avg_kl'].append(np.mean([ms[s]['approx_kl']  for s in SERVICES]))

    # 4. Log
    if episode % LOG_INTERVAL == 0 or episode == 1:
        elapsed = time.time() - train_start
        eta     = (N_EPISODES - episode) / (episode / elapsed) / 60
        w       = min(LOG_INTERVAL, episode)
        avg_r   = np.mean(log['reward_raw'][-w:])
        al = log['avg_actor_loss'][-1]  if log['avg_actor_loss']  else 0
        cl = log['avg_critic_loss'][-1] if log['avg_critic_loss'] else 0
        en = log['avg_entropy'][-1]     if log['avg_entropy']     else 0
        print(f"  Ep {episode:>5}/{N_EPISODES} | "
              f"Reward: {avg_r:>8.3f} | "
              f"ActorL: {al:>8.4f} | "
              f"CriticL: {cl:>7.3f} | "
              f"Entropy: {en:>6.4f} | "
              f"ETA: {eta:.0f}dk")

    # 5. Checkpoint
    if episode % SAVE_INTERVAL == 0:
        path = os.path.join(CHECKPOINT_DIR, f'ippo_new_{episode}.pth')
        torch.save({
            'episode':    episode,
            'agents':     {svc: {
                'actor':      agents[svc].actor.state_dict(),
                'critic':     agents[svc].critic.state_dict(),
                'actor_opt':  agents[svc].actor_opt.state_dict(),
                'critic_opt': agents[svc].critic_opt.state_dict(),
            } for svc in SERVICES},
            'normalizer': vars(normalizer),
            'log':        dict(log),
        }, path)
        print(f"\n  ✓ Checkpoint: {path}\n")

# ─── Eğitim Sonu ─────────────────────────────────────────────────────────────

elapsed   = time.time() - train_start
first100  = np.mean(log['reward_raw'][:100])
last100   = np.mean(log['reward_raw'][-100:])

print("\n" + "=" * 65)
print("Eğitim tamamlandı.")
print(f"  Süre       : {elapsed/60:.1f} dk  |  Updates: {update_count}")
print(f"  İlk 100 ep : {first100:.3f}  →  Son 100 ep: {last100:.3f}  "
      f"({last100-first100:+.3f})")
if log['avg_critic_loss']:
    print(f"  CriticLoss : {log['avg_critic_loss'][0]:.3f} → {log['avg_critic_loss'][-1]:.3f}")
    print(f"  Entropy    : {log['avg_entropy'][0]:.4f} → {log['avg_entropy'][-1]:.4f}")
print("=" * 65)

# ─── CSV ─────────────────────────────────────────────────────────────────────

pd.DataFrame({'episode': log['episode'],
              'reward_raw':  log['reward_raw'],
              'reward_norm': log['reward_norm'],
              }).to_csv('training_log_ippo_new_episodes.csv', index=False)

if log['update_ep']:
    pd.DataFrame({'episode':         log['update_ep'],
                  'avg_actor_loss':  log['avg_actor_loss'],
                  'avg_critic_loss': log['avg_critic_loss'],
                  'avg_entropy':     log['avg_entropy'],
                  'avg_kl':          log['avg_kl'],
                  }).to_csv('training_log_ippo_new_updates.csv', index=False)

print("  ✓ CSV'ler kaydedildi.")

# ─── Grafikler ───────────────────────────────────────────────────────────────

def smooth(data, w=50):
    if len(data) < w:
        return np.array(data)
    return np.convolve(data, np.ones(w)/w, mode='valid')

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('IPPO — Robot Shop MARL (DT v2, MAX_STEPS=6)', fontsize=14, fontweight='bold')

eps    = np.array(log['episode'])
upd_e  = np.array(log['update_ep']) if log['update_ep'] else np.array([])

# Reward
ax = axes[0, 0]
ax.plot(eps, log['reward_raw'], alpha=0.15, color='steelblue', lw=0.8)
if len(log['reward_raw']) >= 50:
    ax.plot(eps[49:], smooth(log['reward_raw']), color='steelblue', lw=2, label='Ort-50')
ax.axhline(0, color='gray', ls='--', alpha=0.5)
ax.set_title('Episode Reward (Ham)')
ax.set_xlabel('Episode')
ax.legend()
ax.grid(True, alpha=0.3)

# Critic Loss
ax = axes[0, 1]
if len(upd_e):
    ax.plot(upd_e, log['avg_critic_loss'], alpha=0.3, color='crimson', lw=0.8)
    if len(log['avg_critic_loss']) >= 20:
        ax.plot(upd_e[19:], smooth(log['avg_critic_loss'], 20), color='crimson', lw=2)
ax.set_title('Critic Loss (normalize → düşmeli)')
ax.set_xlabel('Episode')
ax.grid(True, alpha=0.3)

# Actor Loss
ax = axes[1, 0]
if len(upd_e):
    ax.plot(upd_e, log['avg_actor_loss'], alpha=0.3, color='darkorange', lw=0.8)
    if len(log['avg_actor_loss']) >= 20:
        ax.plot(upd_e[19:], smooth(log['avg_actor_loss'], 20), color='darkorange', lw=2)
ax.set_title('Actor Loss')
ax.set_xlabel('Episode')
ax.grid(True, alpha=0.3)

# Entropy
ax = axes[1, 1]
if len(upd_e):
    ax.plot(upd_e, log['avg_entropy'], alpha=0.3, color='seagreen', lw=0.8)
    if len(log['avg_entropy']) >= 20:
        ax.plot(upd_e[19:], smooth(log['avg_entropy'], 20), color='seagreen', lw=2)
ax.axhline(np.log(3), color='gray', ls='--', alpha=0.7,
           label=f'Uniform {np.log(3):.3f}')
ax.set_title('Policy Entropy (kademeli düşüş)')
ax.set_xlabel('Episode')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_plot_ippo_new.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ training_plot_v2.png kaydedildi.")

# ─── Final Model ─────────────────────────────────────────────────────────────

torch.save({
    'episode':    N_EPISODES,
    'agents':     {svc: {
        'actor':      agents[svc].actor.state_dict(),
        'critic':     agents[svc].critic.state_dict(),
        'actor_opt':  agents[svc].actor_opt.state_dict(),
        'critic_opt': agents[svc].critic_opt.state_dict(),
    } for svc in SERVICES},
    'normalizer': vars(normalizer),
    'log':        dict(log),
}, os.path.join(CHECKPOINT_DIR, 'ippo_new_final.pth'))

print(f"  ✓ Final model: {CHECKPOINT_DIR}/ippo_new_final.pth")
print("\nTamamlandı!")
