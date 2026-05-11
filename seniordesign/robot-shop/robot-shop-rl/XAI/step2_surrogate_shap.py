"""
step2_surrogate_shap.py
-----------------------
XAI Adım 2: Surrogate Model + SHAP Analizi
"""

import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap

SERVICES = ['cart', 'catalogue', 'payment', 'shipping', 'ratings', 'user']
ACTION_LABELS = {0: 'scale_down', 1: 'keep', 2: 'scale_up'}

COLORS = {
    'cart': '#2ecc71', 'catalogue': '#3498db', 'payment': '#e74c3c',
    'shipping': '#f39c12', 'ratings': '#9b59b6', 'user': '#1abc9c'
}

print("=" * 65)
print("XAI Adım 2: Surrogate Model + SHAP Analizi")
print("=" * 65)

with open('xai_obs_map.json', 'r') as f:
    obs_map = json.load(f)

def get_feature_names(svc):
    svc_map = obs_map[svc]
    names = []
    for key in sorted(svc_map.keys(), key=lambda x: int(x.split('_')[1])):
        info = svc_map[key]
        role = '★' if info['role'] == 'self' else '→'
        names.append(f"{role}{info['service']}_{info['metric']}")
    return names

# ─── Her servis için analiz ───────────────────────────────────────────────────

all_results = []
shap_data   = {}

for svc in SERVICES:
    print(f"\n  [{svc}] işleniyor...")

    df       = pd.read_csv(f'xai_obs_{svc}.csv')
    obs_cols = sorted([c for c in df.columns if c.startswith('obs_')],
                      key=lambda x: int(x.split('_')[1]))
    X        = df[obs_cols].values.astype(float)
    y        = df['action'].values
    fnames   = get_feature_names(svc)

    unique, counts = np.unique(y, return_counts=True)
    print(f"    Aksiyon dağılımı: " +
          ", ".join([f"{ACTION_LABELS[int(u)]}={c}" for u, c in zip(unique, counts)]))

    if len(unique) == 1:
        print(f"    ⚠ Tek sınıf: {ACTION_LABELS[int(unique[0])]} — variance analizi")
        feature_vars = np.var(X, axis=0)
        norm_vars    = feature_vars / (feature_vars.max() + 1e-10)

        shap_data[svc] = {
            'type':          'variance',
            'values':        norm_vars,          # 1D array, servis boyutunda
            'feature_names': fnames,
            'action':        ACTION_LABELS[int(unique[0])],
            'n_features':    len(fnames),
        }

        result = {
            'service':            svc,
            'dominant_action':    ACTION_LABELS[int(unique[0])],
            'surrogate_accuracy': 1.0,
            'n_classes':          1,
        }
        all_results.append(result)
        print(f"    ✓ Variance-based importance ({len(fnames)} feature)")

    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
        clf.fit(X_train, y_train)
        acc  = accuracy_score(y_test, clf.predict(X_test))
        print(f"    Surrogate accuracy: {acc:.3f}")

        explainer   = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test[:200])
        # shap_values: (n_samples, n_features, n_classes) veya (n_samples, n_features)
        if shap_values.ndim == 3:
            importance = np.abs(shap_values).mean(axis=(0, 2))
        else:
            importance = np.abs(shap_values).mean(axis=0)
        norm_imp = importance / (importance.max() + 1e-10)

        shap_data[svc] = {
            'type':          'shap',
            'values':        norm_imp,
            'feature_names': fnames,
            'action':        ACTION_LABELS[int(unique[np.argmax(counts)])],
            'n_features':    len(fnames),
        }
        result = {
            'service':            svc,
            'dominant_action':    ACTION_LABELS[int(unique[np.argmax(counts)])],
            'surrogate_accuracy': acc,
            'n_classes':          len(unique),
        }
        all_results.append(result)
        print(f"    ✓ SHAP importance ({len(fnames)} feature)")

# ─── Grafik 1: Servis bazlı feature importance ────────────────────────────────

print("\n  Grafik 1: Servis bazlı...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('IPPO Policy — Feature Importance per Service\n'
             '(★ = kendi servisi, → = komşu servisi)',
             fontsize=13, fontweight='bold')
axes_flat = axes.flatten()

for idx, svc in enumerate(SERVICES):
    ax     = axes_flat[idx]
    data   = shap_data[svc]
    fnames = data['feature_names']          # bu servise ait liste
    vals   = data['values']                 # bu servise ait 1D array
    n      = len(fnames)                    # bu servise ait uzunluk

    assert len(vals) == n, f"{svc}: vals={len(vals)}, fnames={n}"

    y_pos      = np.arange(n)
    bar_colors = [COLORS[svc] if f.startswith('★') else '#BDC3C7' for f in fnames]

    ax.barh(y_pos, vals, color=bar_colors, alpha=0.85, edgecolor='white', lw=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(fnames, fontsize=8)
    ax.set_xlabel('Normalize Önem (0-1)', fontsize=9)
    ax.set_title(f"{svc.upper()}\n(baskın: {data['action']})",
                 fontweight='bold', fontsize=10)
    ax.set_xlim(0, 1.2)
    ax.grid(True, alpha=0.3, axis='x')

    for i, v in enumerate(vals):
        ax.text(v + 0.02, i, f'{v:.2f}', va='center', fontsize=7)

    p1 = mpatches.Patch(color=COLORS[svc], label='Kendi (★)')
    p2 = mpatches.Patch(color='#BDC3C7',  label='Komşu (→)')
    ax.legend(handles=[p1, p2], fontsize=7, loc='lower right')

plt.tight_layout()
plt.savefig('xai_shap_per_service.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ xai_shap_per_service.png")

# ─── Grafik 2: Top 3 özet ─────────────────────────────────────────────────────

print("  Grafik 2: Özet top-3...")

y_labels, y_values, y_colors = [], [], []
for svc in SERVICES:
    data   = shap_data[svc]
    fnames = data['feature_names']
    vals   = data['values']
    top3   = np.argsort(vals)[::-1][:3]
    for i in top3:
        y_labels.append(f"[{svc}] {fnames[i]}")
        y_values.append(float(vals[i]))
        y_colors.append(COLORS[svc])

fig, ax = plt.subplots(figsize=(14, 9))
y_pos   = np.arange(len(y_labels))
ax.barh(y_pos, y_values, color=y_colors, alpha=0.85, edgecolor='white', lw=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(y_labels, fontsize=9)
ax.set_xlabel('Normalize Önem (0-1)', fontsize=10)
ax.set_title('Top-3 Feature Importance — Tüm Servisler', fontsize=13, fontweight='bold')
ax.set_xlim(0, 1.2)
ax.grid(True, alpha=0.3, axis='x')
for pos, val in zip(y_pos, y_values):
    ax.text(val + 0.02, pos, f'{val:.2f}', va='center', fontsize=8)
patches = [mpatches.Patch(color=COLORS[s], label=s) for s in SERVICES]
ax.legend(handles=patches, fontsize=9, loc='lower right')
plt.tight_layout()
plt.savefig('xai_shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ xai_shap_summary.png")

# ─── Konsol özet ─────────────────────────────────────────────────────────────

print(f"\n{'=' * 65}")
print("FEATURE IMPORTANCE ÖZETİ")
print("=" * 65)
for svc in SERVICES:
    data   = shap_data[svc]
    fnames = data['feature_names']
    vals   = data['values']
    top3   = np.argsort(vals)[::-1][:3]
    print(f"\n  {svc.upper()} → {data['action']}")
    for rank, i in enumerate(top3, 1):
        print(f"    {rank}. {fnames[i]:<38} {vals[i]:.3f}")


# ─── Feature importance CSV kaydı ─────────────────────────────────────────────

importance_rows = []

for svc in SERVICES:
    data = shap_data[svc]
    fnames = data["feature_names"]
    vals = data["values"]

    for feature_name, importance in zip(fnames, vals):
        importance_rows.append({
            "service": svc,
            "feature": feature_name,
            "importance": float(importance),
            "importance_type": data["type"],
            "dominant_action": data["action"],
        })

pd.DataFrame(importance_rows).to_csv(
    "xai_feature_importance.csv",
    index=False
)

print("  ✓ xai_feature_importance.csv")

# ─── CSV ─────────────────────────────────────────────────────────────────────

pd.DataFrame(all_results).to_csv('xai_surrogate_results.csv', index=False)
print(f"\n  ✓ xai_surrogate_results.csv")
print("\nAdım 2 tamamlandı.")
print("=" * 65)
