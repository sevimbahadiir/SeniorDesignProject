"""
sar2_augmented.py
-----------------
sar2.py'nin augmentation eklenmiş versiyonu.

Değişiklikler:
- SAR oluşturulduktan sonra Gaussian augmentation eklendi
- Gürültü HAM VERİYE eklenir, sonra normalize edilir (doğru yöntem)
- num_pods, action, reward, date, user_load değişmez
- Sadece sürekli metrikler (cpu, mem, request_rate, latency) gürültü alır
- Payment latency=0 olan satırlarda latency sıfır kalır
- Orijinalde payment>0 iken augmented'da 0 olan 140 sorunlu satır çıkarılır

Augmentation parametreleri:
    NOISE_RATIO : Her kolonun kendi std'sinin yüzdesi olarak gürültü
    AUG_COPIES  : Kaç kopya oluşturulacak (1 = 2x veri)
"""

import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ─── Configuration ────────────────────────────────────────────────────────────

SERVICES = ['cart', 'catalogue', 'payment', 'shipping', 'ratings', 'user']
METRICS  = ['num_pods', 'cpu_usage', 'mem_usage', 'request_rate', 'latency']

SERVICE_WEIGHTS = {
    'cart':      1.5,
    'catalogue': 1.5,
    'shipping':  2.0,
    'ratings':   0.5,
    'user':      1.0,
    'payment':   0.0,
}

POD_COST    = 0.05
NOISE_RATIO = 0.02   # her kolonun std'sinin %5'i kadar gürültü
AUG_COPIES  = 1      # 1 kopya = veri 2x olur

# Gürültü eklenecek metrikler (num_pods hariç)
NOISY_METRICS = ['cpu_usage', 'mem_usage', 'request_rate', 'latency']

# ─── Step 1: Load Clean Data ──────────────────────────────────────────────────
print("=" * 65)
print("STEP 1: Loading Clean Data")
print("=" * 65)

df = pd.read_csv('rl_dataset_clean.csv')
print(f"  Loaded : {len(df)} rows | {len(df.columns)} columns")
print(f"  Runs   : {sorted(df['user_load'].unique())}")

state_cols = []
for svc in SERVICES:
    for metric in METRICS:
        col = f'{svc}_{metric}'
        if col in df.columns:
            state_cols.append(col)

print(f"  State  : {len(state_cols)} columns")

# ─── Step 2: p50 Thresholds ───────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2: Auto-Computing p50 Thresholds")
print("=" * 65)

reward_config = {}
for svc in SERVICES:
    lat_col = f'{svc}_latency'
    valid   = df[df[lat_col] > 0][lat_col]
    p25 = float(valid.quantile(0.25))
    p50 = float(valid.quantile(0.50))
    p75 = float(valid.quantile(0.75))
    reward_config[svc] = {
        'p25': round(p25, 2), 'p50': round(p50, 2), 'p75': round(p75, 2),
        'weight': SERVICE_WEIGHTS[svc],
        'in_reward': svc != 'payment',
        'n_valid': int((df[lat_col] > 0).sum()),
    }
    in_reward = "YES" if svc != 'payment' else "NO (Option C)"
    print(f"  {svc:<12} p50={p50:>10.2f}ms  weight={SERVICE_WEIGHTS[svc]}  reward={in_reward}")

with open('reward_config.json', 'w') as f:
    json.dump(reward_config, f, indent=2)
print(f"  ✓ reward_config.json saved")

# ─── Step 3: Actions ──────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3: Computing Actions")
print("=" * 65)

action_cols = []
for svc in SERVICES:
    pod_col    = f'{svc}_num_pods'
    action_col = f'{svc}_action'
    diff = df.groupby('user_load')[pod_col].diff().fillna(0)
    df[action_col] = diff.apply(
        lambda x: -1 if x < 0 else (1 if x > 0 else 0)
    ).astype(int)
    action_cols.append(action_col)

print(f"  {'Service':<12} {'Scale Down':>12} {'No Change':>12} {'Scale Up':>10}")
for svc, col in zip(SERVICES, action_cols):
    counts = df[col].value_counts()
    total  = len(df)
    down = counts.get(-1, 0)
    keep = counts.get(0,  0)
    up   = counts.get(1,  0)
    print(f"  {svc:<12} {down:>8} ({down/total*100:4.1f}%)"
          f" {keep:>8} ({keep/total*100:4.1f}%)"
          f" {up:>6} ({up/total*100:4.1f}%)")

# ─── Step 4: Reward ───────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4: Computing Reward")
print("=" * 65)

def compute_reward(row):
    reward = 0.0
    for svc, cfg in reward_config.items():
        if not cfg['in_reward']:
            continue
        lat = row.get(f'{svc}_latency', 0)
        if lat <= 0:
            continue
        reward += np.tanh(1.0 - lat / cfg['p50']) * cfg['weight']
    total_pods = sum(row.get(f'{svc}_num_pods', 0) for svc in SERVICES)
    reward -= total_pods * POD_COST
    return round(reward, 4)

df['reward'] = df.apply(compute_reward, axis=1)
print(f"  Reward mean: {df['reward'].mean():.4f}  std: {df['reward'].std():.4f}")
print(f"  Positive: {(df['reward']>0).sum()} ({(df['reward']>0).mean()*100:.1f}%)")

# ─── Step 5: Next State ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5: Building Next State")
print("=" * 65)

next_state_cols = []
for col in state_cols:
    next_col = f'next_{col}'
    df[next_col] = df.groupby('user_load')[col].shift(-1)
    next_state_cols.append(next_col)

before = len(df)
df = df.dropna(subset=next_state_cols).reset_index(drop=True)
print(f"  Dropped  : {before - len(df)} rows (last row of each run)")
print(f"  Remaining: {len(df)} rows")

# ─── Step 6: Save Raw SAR ─────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 6: Saving Raw SAR")
print("=" * 65)

col_order = (['date', 'user_load'] + state_cols +
             action_cols + ['reward'] + next_state_cols)
df_raw = df[col_order].copy()
df_raw.to_csv('rl_dataset_sar_raw.csv', index=False)
print(f"  ✓ rl_dataset_sar_raw.csv ({len(df_raw)} rows)")

# ─── Step 7: Gaussian Augmentation on RAW data ───────────────────────────────
print("\n" + "=" * 65)
print("STEP 7: Gaussian Augmentation (HAM VERİYE)")
print("=" * 65)
print(f"  Noise ratio: her kolonun std'sinin %{int(NOISE_RATIO*100)}'i")
print(f"  Copies     : {AUG_COPIES} → veri {AUG_COPIES+1}x olacak")
print(f"  Sabit      : num_pods, action, reward, date, user_load")
print(f"  Gürültü    : cpu, mem, request_rate, latency")
print()

# Gürültü eklenecek kolonlar (state ve next_state için)
noisy_cols = []
for col in state_cols + next_state_cols:
    if any(col.endswith(f'_{m}') or 
           col.endswith(f'_{m}') 
           for m in NOISY_METRICS):
        # metric adını bul
        for m in NOISY_METRICS:
            if col.endswith(f'_{m}') or col == f'next_{col.split("next_")[-1]}':
                noisy_cols.append(col)
                break

# Daha temiz yaklaşım
noisy_cols = [c for c in state_cols + next_state_cols
              if any(c.endswith(f'_{m}') for m in NOISY_METRICS)]

# Her kolon için std hesapla (ham raw veride)
col_stds = {}
for col in noisy_cols:
    base_col = col.replace('next_', '')
    if base_col in df_raw.columns:
        vals = df_raw[base_col]
        nonzero = vals[vals > 0]
        col_stds[col] = nonzero.std() * NOISE_RATIO if len(nonzero) > 0 else 0
    else:
        col_stds[col] = 0

print(f"  Gürültü std değerleri (ham skala):")
for svc in SERVICES:
    lat_col = f'{svc}_latency'
    if lat_col in col_stds:
        print(f"    {lat_col:<30}: ±{col_stds[lat_col]:.2f} ms")

np.random.seed(42)
aug_frames = [df_raw]

for copy_idx in range(AUG_COPIES):
    df_aug = df_raw.copy()

    # Her kolona kendi std'siyle gürültü ekle
    for col in noisy_cols:
        if col_stds[col] > 0:
            noise = np.random.normal(0, col_stds[col], size=len(df_aug))
            df_aug[col] = df_aug[col] + noise

            # Negatife düşmesin
            df_aug[col] = df_aug[col].clip(lower=0)

    # Payment latency=0 olan satırlar sıfır kalsın
    pay_zero_mask = df_raw['payment_latency'] == 0
    next_pay_zero = df_raw['next_payment_latency'] == 0
    df_aug.loc[pay_zero_mask, 'payment_latency'] = 0.0
    df_aug.loc[next_pay_zero, 'next_payment_latency'] = 0.0

    # Orijinalde payment>0 iken augmented'da 0 olan satırları çıkar
    orig_pay_nonzero = df_raw['payment_latency'].values > 0
    aug_pay_zero     = df_aug['payment_latency'].values == 0
    problem_mask     = orig_pay_nonzero & aug_pay_zero
    df_aug = df_aug[~problem_mask].reset_index(drop=True)

    # date işareti ekle
    df_aug['date'] = df_aug['date'].astype(str) + f'_aug{copy_idx+1}'

    aug_frames.append(df_aug)
    print(f"\n  ✓ Kopya {copy_idx+1}: {len(df_aug)} satır "
          f"({problem_mask.sum()} sorunlu çıkarıldı)")

df_combined = pd.concat(aug_frames, ignore_index=True)
print(f"\n  Orijinal  : {len(df_raw):>6} satır")
print(f"  Augmented : {len(df_combined)-len(df_raw):>6} satır")
print(f"  Toplam    : {len(df_combined):>6} satır")

# ─── Step 8: Normalize ────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 8: Normalizing (train-only fit, TÜM VERİ ÜZERİNDE)")
print("=" * 65)
print("  Scaler orijinal train split'e fit edilir.")
print("  Augmented veri de aynı scaler ile transform edilir.")

normalize_cols = state_cols + next_state_cols

# Train/val split sadece orijinal veriden
orig_df = df_combined[~df_combined['date'].astype(str).str.contains('aug')]
train_idx, val_idx = train_test_split(orig_df.index, test_size=0.2, shuffle=False)

scaler = MinMaxScaler()
scaler.fit(df_combined.loc[train_idx, normalize_cols])

df_norm = df_combined.copy()
df_norm[normalize_cols] = scaler.transform(df_combined[normalize_cols])
df_norm[normalize_cols] = df_norm[normalize_cols].clip(0, 1)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"  Train: {len(train_idx)} rows  Val: {len(val_idx)} rows")
print(f"  Normalized range: [{df_norm[normalize_cols].min().min():.4f}, "
      f"{df_norm[normalize_cols].max().max():.4f}]")

# ─── Step 9: Gürültü Kalite Kontrolü ─────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 9: Gürültü Kalite Kontrolü")
print("=" * 65)

orig_norm = df_norm[~df_norm['date'].astype(str).str.contains('aug')]
aug_norm  = df_norm[df_norm['date'].astype(str).str.contains('aug')]

print(f"  {'Servis':<12} {'mean_diff':>12} {'std_diff':>12} {'max_diff':>12}")
for svc in SERVICES:
    col = f'{svc}_latency'
    min_len = min(len(orig_norm), len(aug_norm))
    o = orig_norm[col].values[:min_len]
    a = aug_norm[col].values[:min_len]
    diff = a - o
    flag = ' ⚠️' if diff.std() > 0.05 else ' ✓'
    print(f"  {svc:<12} {diff.mean():>+12.4f} {diff.std():>12.4f} "
          f"{np.abs(diff).max():>12.4f}{flag}")

# ─── Step 10: Save ────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 10: Saving")
print("=" * 65)

df_norm.to_csv('rl_dataset_sar_normalized.csv', index=False)
print(f"  ✓ rl_dataset_sar_normalized.csv ({len(df_norm)} satır)")
print(f"  ✓ scaler.pkl")
print(f"  ✓ reward_config.json")

# ─── Step 11: Verification ────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 11: Verification")
print("=" * 65)

min_val = df_norm[normalize_cols].min().min()
max_val = df_norm[normalize_cols].max().max()
print(f"  Normalized range : [{min_val:.6f}, {max_val:.6f}] "
      f"{'✓' if 0 <= min_val and max_val <= 1.0 else '✗'}")

action_ok = all(
    set(df_norm[c].unique()).issubset({-1, 0, 1}) for c in action_cols
)
print(f"  Action values    : [-1,0,1] only {'✓' if action_ok else '✗'}")

pay_orig_zero = (orig_norm['payment_latency'] == 0).sum()
pay_aug_zero  = (aug_norm['payment_latency'] == 0).sum()
print(f"  Payment zero     : orijinal={pay_orig_zero} aug={pay_aug_zero} "
      f"{'✓' if pay_orig_zero==pay_aug_zero else '⚠️'}")

print(f"  Toplam satır     : {len(df_norm):,}")

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"  Orijinal SAR     : {len(df_raw):,} satır")
print(f"  Augmented toplam : {len(df_norm):,} satır ({AUG_COPIES+1}x)")
print(f"  Noise ratio      : std'nin %{int(NOISE_RATIO*100)}'i (ham skala)")
print(f"  Yöntem           : Ham veriye gürültü → normalize")
print(f"  num_pods         : değişmedi")
print(f"  action/reward    : değişmedi")
print(f"  payment zero     : korundu")
print("=" * 65)
print("Done!")
