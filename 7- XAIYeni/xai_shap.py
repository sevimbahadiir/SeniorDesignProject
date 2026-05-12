"""
xai_shap.py - XAI Pipeline SHAP Feature Importance
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

CHECKPOINT   = 'checkpoints_ippo_new/ippo_new_final.pth'
DATA_PATH    = 'rl_dataset_sar_normalized.csv'
N_BACKGROUND = 100
N_EXPLAIN    = 200
RANDOM_SEED  = 42
OUTPUT_DIR   = 'xai_output'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/shap_values', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/per_service', exist_ok=True)

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

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
ACTION_LABELS  = {0: 'Scale Down', 1: 'Hold', 2: 'Scale Up'}
METRIC_LABELS  = {'num_pods':'Pods','cpu_usage':'CPU','mem_usage':'MEM',
                  'request_rate':'RPS','latency':'LAT'}
SERVICE_COLORS = {'cart':'#4C72B0','catalogue':'#DD8452','payment':'#55A868',
                  'shipping':'#C44E52','ratings':'#8172B2','user':'#937860'}
SERVICE_IDX    = {svc: i * len(METRICS) for i, svc in enumerate(SERVICES)}

def get_feature_names(svc):
    names = [f'{svc[:3].upper()}_{METRIC_LABELS[m]}' for m in METRICS]
    for nb in AGENT_NEIGHBORS[svc]:
        names += [f'{nb[:3].upper()}_{METRIC_LABELS[m]}' for m in METRICS]
    return names

def get_obs_dim(svc):
    return (1 + len(AGENT_NEIGHBORS[svc])) * len(METRICS)

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

# ─── Step 1 ───────────────────────────────────────────────────────────────────
print("=" * 65)
print("XAI Pipeline — SHAP Feature Importance")
print("=" * 65)
print("\nSTEP 1: Model yukleniyor...")
ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
actors = {}
for svc in SERVICES:
    actor = ActorNet(get_obs_dim(svc))
    actor.load_state_dict(ckpt['agents'][svc]['actor'])
    actor.eval()
    actors[svc] = actor
    print(f"  {svc:<12} obs_dim={get_obs_dim(svc)}  OK")

# ─── Step 2 ───────────────────────────────────────────────────────────────────
print("\nSTEP 2: Veri hazirlaniyor...")
df = pd.read_csv(DATA_PATH)
state_cols    = [f'{s}_{m}' for s in SERVICES for m in METRICS if f'{s}_{m}' in df.columns]
global_states = df[state_cols].values.astype(np.float32)

def get_obs(gs, svc):
    parts = [gs[SERVICE_IDX[svc]:SERVICE_IDX[svc]+len(METRICS)]]
    for nb in AGENT_NEIGHBORS[svc]:
        parts.append(gs[SERVICE_IDX[nb]:SERVICE_IDX[nb]+len(METRICS)])
    return np.concatenate(parts)

obs_per_svc = {}
for svc in SERVICES:
    obs_per_svc[svc] = np.array([get_obs(gs, svc) for gs in global_states])
    print(f"  {svc:<12} {obs_per_svc[svc].shape}  OK")

# ─── Step 3: SHAP ─────────────────────────────────────────────────────────────
print(f"\nSTEP 3: SHAP hesaplaniyor...")

all_shap_values = {}  # svc -> (N_EXPLAIN, obs_dim, 3)
all_mean_abs    = {}  # svc -> (obs_dim,)

for svc in SERVICES:
    obs_dim  = get_obs_dim(svc)
    npy_path = f'{OUTPUT_DIR}/shap_values/{svc}_shap.npy'

    # Gecerli cache var mi?
    if os.path.exists(npy_path):
        arr = np.load(npy_path)
        if arr.shape == (N_EXPLAIN, obs_dim, 3):
            print(f"  [{svc}] cache {arr.shape} yuklendi OK")
            all_shap_values[svc] = arr
            all_mean_abs[svc]    = np.mean(np.abs(arr), axis=(0, 2))
            continue
        else:
            print(f"  [{svc}] eski format {arr.shape} siliniyor...")
            os.remove(npy_path)

    print(f"  [{svc}] hesaplaniyor...", flush=True)
    obs_data   = obs_per_svc[svc]
    idx_all    = np.random.permutation(len(obs_data))
    background = obs_data[idx_all[:N_BACKGROUND]]
    explain    = obs_data[idx_all[N_BACKGROUND:N_BACKGROUND+N_EXPLAIN]]

    # KernelSHAP
    _svc = svc  # closure icin
    def predict_proba(x, _svc=_svc):
        with torch.no_grad():
            probs = torch.softmax(
                actors[_svc](torch.tensor(x.astype(np.float32))), dim=-1)
        return probs.numpy()  # (batch, 3)

    explainer = shap.KernelExplainer(predict_proba, background)
    shap_vals = explainer.shap_values(explain, nsamples=100, silent=True)

    # shap_vals tipini kontrol et
    print(f"    shap_vals type: {type(shap_vals)}")
    if isinstance(shap_vals, list):
        print(f"    list len={len(shap_vals)}, elem shape={np.array(shap_vals[0]).shape}")
        # Her eleman: (N_EXPLAIN, obs_dim)
        # Stack axis=2 -> (N_EXPLAIN, obs_dim, 3)
        arr_list = [np.array(sv) for sv in shap_vals]
        shap_arr = np.stack(arr_list, axis=2)
    else:
        print(f"    array shape={np.array(shap_vals).shape}")
        shap_arr = np.array(shap_vals)
        # Eger (3, N_EXPLAIN, obs_dim) ise transpoz al
        if shap_arr.ndim == 3 and shap_arr.shape[0] == 3:
            shap_arr = shap_arr.transpose(1, 2, 0)

    print(f"    final shap_arr: {shap_arr.shape}  beklenen: ({N_EXPLAIN}, {obs_dim}, 3)")

    # Boyut dogrulama
    if shap_arr.shape != (N_EXPLAIN, obs_dim, 3):
        # Son care: manuel transpose dene
        if shap_arr.shape == (obs_dim, 3, N_EXPLAIN):
            shap_arr = shap_arr.transpose(2, 0, 1)
        elif shap_arr.shape == (3, obs_dim, N_EXPLAIN):
            shap_arr = shap_arr.transpose(2, 1, 0)
        print(f"    duzeltildi: {shap_arr.shape}")

    all_shap_values[svc] = shap_arr
    all_mean_abs[svc]    = np.mean(np.abs(shap_arr), axis=(0, 2))  # (obs_dim,)

    np.save(npy_path, shap_arr)
    feat_names = get_feature_names(svc)
    top3 = np.argsort(all_mean_abs[svc])[::-1][:3]
    print(f"    Top-3: {[feat_names[i] for i in top3]}")
    print(f"    Saved: {npy_path}")

# ─── Step 4: Grafikler ────────────────────────────────────────────────────────
print("\nSTEP 4: Per-service grafikler...")

for svc in SERVICES:
    feat_names = get_feature_names(svc)
    mean_abs   = all_mean_abs[svc]   # (obs_dim,)
    shap_arr   = all_shap_values[svc]  # (N, obs_dim, 3)
    obs_dim    = len(feat_names)

    top_n   = min(obs_dim, 10)
    top_idx = np.argsort(mean_abs)[::-1][:top_n]  # en buyuk -> kucuk

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'SHAP — {svc.upper()} Agent', fontsize=13, fontweight='bold')

    # Sol: barh  (en onemli en uste)
    ax         = axes[0]
    # Ters cevir: top_idx[-1] uste, top_idx[0] altta
    plot_order = top_idx[::-1]  # kucukten buyuge, uste buyuk ciksin
    y_pos      = np.arange(top_n)
    vals_plot  = mean_abs[plot_order]
    names_plot = [feat_names[i] for i in plot_order]

    ax.barh(y_pos, vals_plot,
            color=SERVICE_COLORS[svc], alpha=0.8, edgecolor='black', lw=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_plot, fontsize=9)
    ax.set_xlabel('Mean |SHAP|')
    ax.set_title('Feature Importance')
    ax.grid(True, alpha=0.3, axis='x')

    # Sag: stacked bar
    ax     = axes[1]
    x_pos  = np.arange(top_n)
    bottom = np.zeros(top_n)
    for a_idx, (a_label, a_color) in enumerate(
            zip(ACTION_LABELS.values(), ['#d62728','#aec7e8','#2ca02c'])):
        vals_a = np.mean(np.abs(shap_arr[:, top_idx, a_idx]), axis=0)  # (top_n,)
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
    plt.savefig(f'{OUTPUT_DIR}/per_service/{svc}_shap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {svc} OK")

# ─── Step 5: 6-panel summary ──────────────────────────────────────────────────
print("\nSTEP 5: Summary grafik...")
fig       = plt.figure(figsize=(18, 12))
fig.suptitle('SHAP Feature Importance — Tum Servisler (IPPO)',
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
    ax.tick_params(axis='x', labelsize=7)

plt.savefig(f'{OUTPUT_DIR}/xai_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  {OUTPUT_DIR}/xai_summary.png OK")

# ─── Step 6: Heatmap ──────────────────────────────────────────────────────────
print("\nSTEP 6: Heatmap...")
own_shap = np.zeros((len(SERVICES), len(METRICS)))
for i, svc in enumerate(SERVICES):
    feat_names = get_feature_names(svc)
    mean_abs   = all_mean_abs[svc]
    for j, m in enumerate(METRICS):
        label = f'{svc[:3].upper()}_{METRIC_LABELS[m]}'
        if label in feat_names:
            own_shap[i, j] = mean_abs[feat_names.index(label)]

row_max   = own_shap.max(axis=1, keepdims=True)
row_max[row_max == 0] = 1
norm_shap = own_shap / row_max

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(norm_shap, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(len(METRICS)))
ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=11)
ax.set_yticks(range(len(SERVICES)))
ax.set_yticklabels([s.upper() for s in SERVICES], fontsize=11)
ax.set_title('Kendi Metriklerinin Normalize SHAP Onemi', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='Normalize SHAP')
for i in range(len(SERVICES)):
    for j in range(len(METRICS)):
        val = norm_shap[i, j]
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9,
                color='black' if val < 0.7 else 'white', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/xai_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  {OUTPUT_DIR}/xai_heatmap.png OK")

# ─── Step 7: CSV ──────────────────────────────────────────────────────────────
print("\nSTEP 7: CSV...")
rows = []
for svc in SERVICES:
    feat_names = get_feature_names(svc)
    mean_abs   = all_mean_abs[svc]
    for rank, fi in enumerate(np.argsort(mean_abs)[::-1], start=1):
        rows.append({'service':svc,'feature':feat_names[fi],
                     'mean_abs_shap':round(float(mean_abs[fi]),6),'rank':rank})
pd.DataFrame(rows).to_csv(f'{OUTPUT_DIR}/xai_results.csv', index=False)
print(f"  {OUTPUT_DIR}/xai_results.csv OK")

# ─── Ozet ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("XAI TAMAMLANDI")
print("=" * 65)
for svc in SERVICES:
    feat_names = get_feature_names(svc)
    mean_abs   = all_mean_abs[svc]
    top_fi     = np.argmax(mean_abs)
    print(f"  {svc:<12} -> {feat_names[top_fi]:<20} SHAP={mean_abs[top_fi]:.4f}")
print("=" * 65)
print("Done!")