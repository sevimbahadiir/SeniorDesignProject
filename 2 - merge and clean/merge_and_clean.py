"""
merge_and_clean.py
------------------
Step 1: Merges and cleans raw CSV files.

Operations:
1. Merge 10 CSV files into one (40 to 500 concurrent users)
2. Drop dispatch and web columns (no agents for these services)
3. Drop rows where cart + catalogue + shipping latency are ALL zero
   (payment zero rows are kept — Option C decision)
4. Clip pod counts to per-service bounds
5. Save clean CSV

Output: rl_dataset_clean.csv
"""

import pandas as pd
import numpy as np

# ─── Configuration ────────────────────────────────────────────────────────────

CSV_FILES = {
    '40user':  'dataset_run1_40user.csv',
    '80user':  'dataset_run2_80user.csv',
    '120user': 'dataset_run3_120user.csv',
    '150user': 'dataset_run4_150user.csv',
    '200user': 'dataset_run5_200user.csv',
    '250user': 'dataset_run6_250user.csv',
    '300user': 'dataset_run7_300user.csv',
    '350user': 'dataset_run8_350user.csv',
    '400user': 'dataset_run9_400user.csv',
    '500user': 'dataset_run10_500user.csv',
}

SERVICES = ['cart', 'catalogue', 'payment', 'shipping', 'ratings', 'user']
METRICS  = ['num_pods', 'cpu_usage', 'mem_usage', 'request_rate', 'latency']

# Valid pod bounds per service (aligned with random_scaler.py SERVICE_BOUNDS)
POD_BOUNDS = {
    'cart':      (1, 8),
    'catalogue': (1, 8),
    'payment':   (2, 8),
    'shipping':  (2, 8),
    'ratings':   (1, 8),
    'user':      (1, 8),
}

# A row is invalid only when ALL three of these services report zero latency
# simultaneously (indicates missing measurement, not a real system state).
# Payment zero rows are intentionally kept (Option C).
FILTER_SERVICES = ['cart', 'catalogue', 'shipping']

# ─── Step 1: Load and Merge ───────────────────────────────────────────────────
print("=" * 65)
print("STEP 1: Loading and Merging CSV Files")
print("=" * 65)

frames = []
for user_load, filename in CSV_FILES.items():
    try:
        df_tmp = pd.read_csv(filename)
        df_tmp['user_load'] = user_load
        frames.append(df_tmp)
        print(f"  ✓ {filename:<38} → {len(df_tmp):>5} rows")
    except FileNotFoundError:
        print(f"  ✗ {filename} not found, skipping.")

if not frames:
    raise RuntimeError("No CSV files loaded. Check file paths.")

df = pd.concat(frames, ignore_index=True)
df = df.sort_values('date').reset_index(drop=True)
print(f"\n  Total merged : {len(df)} rows | {len(df.columns)} columns")
print(f"  Runs loaded  : {len(frames)}")

# ─── Step 2: Drop dispatch and web Columns ────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2: Dropping dispatch and web Columns")
print("=" * 65)

to_drop = [c for c in df.columns
           if c.startswith('dispatch') or c.startswith('web')]
df = df.drop(columns=to_drop)
print(f"  Dropped ({len(to_drop)} columns): {to_drop}")
print(f"  Remaining columns: {len(df.columns)}")

# ─── Step 3: Drop Invalid Rows ────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3: Dropping Invalid Rows")
print("=" * 65)
print(f"  Rule: drop rows where {' + '.join(FILTER_SERVICES)} latency")
print(f"        are ALL zero simultaneously.")
print(f"  Note: rows where only payment=0 are KEPT (Option C).")

before = len(df)
filter_cols = [f'{svc}_latency' for svc in FILTER_SERVICES]
all_zero    = df[filter_cols].eq(0).all(axis=1)
df = df[~all_zero].reset_index(drop=True)

dropped = before - len(df)
print(f"\n  Before : {before} rows")
print(f"  Dropped: {dropped} rows")
print(f"  After  : {len(df)} rows")

pay_zero = (df['payment_latency'] == 0).sum()
print(f"\n  Payment latency = 0 rows: {pay_zero}/{len(df)} "
      f"({pay_zero/len(df)*100:.1f}%) → KEPT (Option C)")

# ─── Step 4: Clip Pod Counts ──────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4: Clipping Pod Counts to Valid Bounds")
print("=" * 65)
print(f"  {'Service':<12} {'Before (min-max)':<20} {'After (min-max)':<20} Changed")
print(f"  {'-'*62}")

for svc, (lo, hi) in POD_BOUNDS.items():
    col = f'{svc}_num_pods'
    if col not in df.columns:
        continue
    before_min, before_max = df[col].min(), df[col].max()
    changed = ((df[col] < lo) | (df[col] > hi)).sum()
    df[col] = df[col].clip(lower=lo, upper=hi).round().astype(int)
    after_min, after_max = df[col].min(), df[col].max()
    print(f"  {svc:<12} {str(before_min)+'-'+str(before_max):<20} "
          f"{str(after_min)+'-'+str(after_max):<20} {changed} rows")

# ─── Step 5: Reorder Columns ──────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5: Reordering Columns")
print("=" * 65)

col_order = ['date', 'user_load']
for svc in SERVICES:
    for metric in METRICS:
        col = f'{svc}_{metric}'
        if col in df.columns:
            col_order.append(col)

df = df[col_order]
print(f"  {len(df.columns)} columns: date, user_load + "
      f"{len(SERVICES)} services × {len(METRICS)} metrics")

# ─── Step 6: Save ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 6: Saving Output")
print("=" * 65)

output_file = 'rl_dataset_clean.csv'
df.to_csv(output_file, index=False)
print(f"  ✓ Saved: {output_file}")

# ─── Step 7: Summary Report ───────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY REPORT")
print("=" * 65)
print(f"  Total rows    : {len(df)}")
print(f"  Total columns : {len(df.columns)}")
print(f"  Time range    : {df['date'].iloc[0]} → {df['date'].iloc[-1]}")
print()
print(f"  User load distribution:")
for load, count in df['user_load'].value_counts().sort_index().items():
    print(f"    {load:<10}: {count} rows")
print()
print(f"  {'Service':<12} {'Pods(min-max)':<15} {'MaxRPS':>8} "
      f"{'Zero-lat%':>10} {'AvgLat(ms)':>12}")
print(f"  {'-'*62}")
for svc in SERVICES:
    pod_col = f'{svc}_num_pods'
    lat_col = f'{svc}_latency'
    rps_col = f'{svc}_request_rate'
    pod_r   = f"{int(df[pod_col].min())}-{int(df[pod_col].max())}"
    zpct    = (df[lat_col] == 0).mean() * 100
    avglat  = df[df[lat_col] > 0][lat_col].mean() if (df[lat_col] > 0).any() else 0
    maxrps  = df[rps_col].max()
    flag    = " ⚠" if zpct >= 30 else ""
    print(f"  {svc:<12} {pod_r:<15} {maxrps:>8.2f} "
          f"{zpct:>9.1f}%{flag} {avglat:>12.1f}")
print()
print("  Option C note:")
print("  Payment zero rows are kept. In the SAR step, payment")
print("  latency will be included in state but excluded from reward.")
print("=" * 65)
print("Done!")
