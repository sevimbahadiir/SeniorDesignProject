"""
step3_diagnosis_engine.py
-------------------------
XAI Adim 3: Rule-based Diagnosis Engine

Evaluate sonuclarindan alinan yeni metrikler:
  - IPPO/MAPPO/A2C mean_reward = +2.059, SLO = 13.2%
  - Rule-based mean_reward = -9.206, SLO = 34.9%
  - evaluate_summary_final.csv'deki latency degerleri kullanildi

Normalize latency degerlerini denormalize etmek icin scaler kullanilir.
"""

import numpy as np
import pandas as pd
import json
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SERVICES = ['cart', 'catalogue', 'payment', 'shipping', 'ratings', 'user']
METRICS  = ['num_pods', 'cpu_usage', 'mem_usage', 'request_rate', 'latency']

# ─── Yeni evaluate sonuclarindan alinan MARL iyilestirme metrikleri ───────────
# Kaynak: evaluation_summary_final.csv
# Rule-based mean_latency → IPPO mean_latency (normalize [0,1])
# Denormalize degerleri hesaplamak icin scaler.pkl kullanilacak
# Asagidaki degerler normalize latency olup scaler ile ms'ye donusturulecek

# evaluation_summary_final.csv'den:
# Rule-based latency (normalize): cart=0.1136, catalogue=0.1034,
#   payment=0.0188, shipping=0.2912, ratings=0.2876, user=0.0663
# IPPO latency (normalize):       cart=0.0619, catalogue=0.0629,
#   payment=0.0162, shipping=0.2201, ratings=0.2501, user=0.0435

EVAL_RESULTS = {
    'cart':      {'rb_lat_norm': 0.1136, 'ippo_lat_norm': 0.0619},
    'catalogue': {'rb_lat_norm': 0.1034, 'ippo_lat_norm': 0.0629},
    'payment':   {'rb_lat_norm': 0.0188, 'ippo_lat_norm': 0.0162},
    'shipping':  {'rb_lat_norm': 0.2912, 'ippo_lat_norm': 0.2201},
    'ratings':   {'rb_lat_norm': 0.2876, 'ippo_lat_norm': 0.2501},
    'user':      {'rb_lat_norm': 0.0663, 'ippo_lat_norm': 0.0435},
}

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

state_cols = [f'{s}_{m}' for s in SERVICES for m in METRICS]

def denorm(svc, metric, val):
    idx  = state_cols.index(f'{svc}_{metric}')
    dmin = scaler.data_min_[idx]
    dmax = scaler.data_max_[idx]
    return float(val * (dmax - dmin) + dmin)

# Normalize latency degerlerini ms'ye donustur
MARL_IMPROVEMENT = {}
for svc in SERVICES:
    rb_lat_ms   = denorm(svc, 'latency', EVAL_RESULTS[svc]['rb_lat_norm'])
    ippo_lat_ms = denorm(svc, 'latency', EVAL_RESULTS[svc]['ippo_lat_norm'])
    improved    = ippo_lat_ms < rb_lat_ms
    MARL_IMPROVEMENT[svc] = {
        'rb_lat':   rb_lat_ms,
        'ippo_lat': ippo_lat_ms,
        'improved': improved,
    }

# ─── Veri Yukle ───────────────────────────────────────────────────────────────
df = pd.read_csv('xai_behavior_dataset.csv')
with open('xai_obs_map.json') as f:
    obs_map = json.load(f)
with open('reward_config.json') as f:
    reward_cfg = json.load(f)

print("=" * 65)
print("XAI Adim 3: Rule-based Diagnosis Engine")
print("=" * 65)

print(f"\n  Denormalize edilmis latency degerleri (ms):")
print(f"  {'Servis':<12} {'RB Latency':>12} {'IPPO Latency':>14} {'Iyilesme':>10}")
print(f"  {'-'*52}")
for svc in SERVICES:
    rb  = MARL_IMPROVEMENT[svc]['rb_lat']
    ip  = MARL_IMPROVEMENT[svc]['ippo_lat']
    pct = (rb - ip) / rb * 100 if rb > 0 else 0
    print(f"  {svc:<12} {rb:>12.1f} {ip:>14.1f} {pct:>+10.1f}%")

# ─── Metrik Hesaplama ─────────────────────────────────────────────────────────
def get_service_metrics(svc):
    svc_df = df[df['service'] == svc]
    self_metrics = {
        info['metric']: f"obs_{key.split('_')[1]}"
        for key, info in obs_map[svc].items()
        if info['role'] == 'self'
    }
    result = {}
    for metric, col in self_metrics.items():
        if col in svc_df.columns:
            norm_val = svc_df[col].mean()
            real_val = denorm(svc, metric, norm_val)
            result[metric] = {'norm': norm_val, 'real': real_val}
    result['action']     = svc_df['action_label'].mode()[0]
    result['confidence'] = svc_df['confidence'].mean()
    return result

# ─── Diagnosis Kurallari ──────────────────────────────────────────────────────
def diagnose(svc, metrics, action):
    evidence = []
    details  = {}

    lat_norm = metrics.get('latency',      {}).get('norm', 0)
    cpu_norm = metrics.get('cpu_usage',    {}).get('norm', 0)
    pod_norm = metrics.get('num_pods',     {}).get('norm', 0)
    lat_real = metrics.get('latency',      {}).get('real', 0)
    pod_real = metrics.get('num_pods',     {}).get('real', 0)

    p50_real  = reward_cfg.get(svc, {}).get('p50', 1000)
    in_reward = reward_cfg.get(svc, {}).get('in_reward', True)
    improved  = MARL_IMPROVEMENT[svc]['improved']
    ippo_lat  = MARL_IMPROVEMENT[svc]['ippo_lat']
    rb_lat    = MARL_IMPROVEMENT[svc]['rb_lat']

    details['latency_ms']     = round(lat_real, 1)
    details['p50_ms']         = p50_real
    details['latency_ratio']  = round(lat_real / p50_real, 2) if p50_real > 0 else 0
    details['pod_count']      = round(pod_real, 1)
    details['cpu_norm']       = round(cpu_norm, 3)
    details['action']         = action
    details['in_reward']      = in_reward
    details['ippo_lat_ms']    = round(ippo_lat, 1)
    details['rb_lat_ms']      = round(rb_lat, 1)
    details['pct_improvement']= round((rb_lat - ippo_lat) / rb_lat * 100, 1) if rb_lat > 0 else 0
    details['confidence']     = round(metrics.get('confidence', 0), 3)

    # Kural 0: Reward disi
    if not in_reward:
        evidence.append("Reward disi servis (Option C — payment)")
        evidence.append(f"Latency: {lat_real:.0f}ms — izleniyor ama optimize edilmiyor")
        return 'reward_excluded', 'info', evidence, details

    pct = details['pct_improvement']

    # Kural 1: MARL iyilestirdi (>%10)
    if improved and pct > 10.0:
        evidence.append(f"MARL latency iyilestirmesi: {rb_lat:.0f}ms -> {ippo_lat:.0f}ms ({pct:+.1f}%)")
        evidence.append(f"SLO ihlal orani: %13.2 (Rule-based: %34.9)")
        evidence.append(f"Mevcut politika: {action} (guven: {details.get('confidence', 0):.2f})")
        return 'marl_optimized', 'ok', evidence, details

    # Kural 2: Shipping — pod-insensitive bottleneck
    if svc == 'shipping' and action == 'scale_up' and cpu_norm < 0.65:
        evidence.append(f"Latency {lat_real:.0f}ms > p50 ({p50_real:.0f}ms)")
        evidence.append(f"CPU normalize: {cpu_norm:.2f} — I/O-bound bottleneck")
        evidence.append("Scale-up etkisiz: DB/I/O bottleneck (action space siniri)")
        evidence.append(f"MARL vs Rule-based: {ippo_lat:.0f}ms vs {rb_lat:.0f}ms")
        return 'pod_insensitive_bottleneck', 'critical', evidence, details

    # Kural 3: Genel pod-insensitive
    if lat_real > 1.5 * p50_real and action == 'scale_up' and cpu_norm < 0.4:
        evidence.append(f"Latency {lat_real:.0f}ms > 1.5xp50 ({1.5*p50_real:.0f}ms)")
        evidence.append(f"CPU dusuk ({cpu_norm:.2f}) — CPU-bound degil")
        return 'pod_insensitive_bottleneck', 'warning', evidence, details

    # Kural 4: Under-provisioned
    if lat_real > 1.2 * p50_real and action in ('keep', 'scale_down'):
        evidence.append(f"Latency {lat_real:.0f}ms > 1.2xp50 ({1.2*p50_real:.0f}ms)")
        evidence.append(f"Agent {action} seciyor — reaktif olmayan politika")
        return 'under_provisioned', 'warning', evidence, details

    # Kural 5: Over-provisioned
    if lat_real < 0.5 * p50_real and action == 'scale_down':
        evidence.append(f"Latency {lat_real:.0f}ms < 0.5xp50 ({0.5*p50_real:.0f}ms)")
        evidence.append("Verimli kaynak kullanimi — scale_down uygun")
        return 'over_provisioned', 'ok', evidence, details

    # Kural 6: Healthy
    if lat_real <= p50_real:
        evidence.append(f"Latency {lat_real:.0f}ms <= p50 ({p50_real:.0f}ms)")
        evidence.append(f"Agent {action} — normal calisma")
        return 'healthy', 'ok', evidence, details

    evidence.append(f"Latency {lat_real:.0f}ms, p50 {p50_real:.0f}ms, aksiyon {action}")
    return 'monitoring_required', 'info', evidence, details

# ─── Ana Analiz ───────────────────────────────────────────────────────────────
SEV_COLORS = {'critical':'#e74c3c','warning':'#f39c12','ok':'#2ecc71','info':'#3498db'}
DIAG_TR = {
    'pod_insensitive_bottleneck': 'Pod-Insensitive Bottleneck',
    'under_provisioned':          'Under-Provisioned',
    'over_provisioned':           'Over-Provisioned',
    'marl_optimized':             'MARL Optimize Etti',
    'healthy':                    'Saglikli',
    'reward_excluded':            'Reward Disi (Option C)',
    'monitoring_required':        'Izleme Gerekli',
}

print(f"\n  {'Servis':<12} {'Diagnosis':<30} {'Sev':<8} {'Latency':>10} {'p50':>8} {'Iyilesme':>10}")
print(f"  {'-'*82}")

diagnosis_report = {}
rows = []

for svc in SERVICES:
    metrics  = get_service_metrics(svc)
    action   = metrics['action']
    diagnosis, severity, evidence, details = diagnose(svc, metrics, action)
    diagnosis_report[svc] = {
        'diagnosis': diagnosis, 'severity': severity,
        'action': action, 'confidence': round(metrics['confidence'], 3),
        'evidence': evidence, 'details': details,
    }
    pct     = details['pct_improvement']
    pct_str = f"{pct:+.1f}%" if pct != 0 else "--"
    print(f"  {svc:<12} {DIAG_TR.get(diagnosis,diagnosis):<30} "
          f"{severity:<8} {details['latency_ms']:>10.1f} "
          f"{details['p50_ms']:>8.1f} {pct_str:>10}")
    rows.append({
        'service': svc, 'diagnosis': diagnosis, 'severity': severity,
        'action': action, 'latency_ms': details['latency_ms'],
        'p50_ms': details['p50_ms'], 'latency_ratio': details['latency_ratio'],
        'pod_count': details['pod_count'], 'cpu_norm': details['cpu_norm'],
        'confidence': round(metrics['confidence'], 3),
        'pct_improvement': details['pct_improvement'],
    })

# ─── Detayli Cikti ────────────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("DETAYLI DIAGNOSIS")
print("=" * 65)
for svc, report in diagnosis_report.items():
    d = report['details']
    print(f"\n  -- {svc.upper()} --")
    print(f"  Diagnosis  : {DIAG_TR.get(report['diagnosis'], report['diagnosis'])}")
    print(f"  Severity   : {report['severity']}")
    print(f"  Aksiyon    : {report['action']} (guven: {report['confidence']:.3f})")
    print(f"  Latency    : {d['latency_ms']:.1f}ms (p50: {d['p50_ms']}ms, {d['latency_ratio']:.2f}x)")
    if d['pct_improvement'] != 0:
        print(f"  MARL Iyi.  : {d['rb_lat_ms']:.0f}ms -> {d['ippo_lat_ms']:.0f}ms ({d['pct_improvement']:+.1f}%)")
    for ev in report['evidence']:
        print(f"    * {ev}")

# ─── Grafik ───────────────────────────────────────────────────────────────────
df_rows = pd.DataFrame(rows)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Rule-based Diagnosis Engine — Servis Bazli Analiz (Yeni DT)',
             fontsize=13, fontweight='bold')

ax = axes[0]
x  = np.arange(len(SERVICES))
w  = 0.35
clrs = [SEV_COLORS[diagnosis_report[s]['severity']] for s in SERVICES]
ax.bar(x - w/2, df_rows['latency_ms'], w, label='IPPO Latency (ms)',
       color=clrs, alpha=0.85, edgecolor='black', lw=0.8)
ax.bar(x + w/2, df_rows['p50_ms'],     w, label='p50 esigi (ms)',
       color='lightgray', alpha=0.85, edgecolor='black', lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(SERVICES, rotation=15)
ax.set_yscale('log')
ax.set_ylabel('Latency (ms, log scale)')
ax.set_title('IPPO Latency vs p50 Esigi')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1]
diag_labels = [DIAG_TR.get(diagnosis_report[s]['diagnosis'], '?') for s in SERVICES]
conf_vals   = [diagnosis_report[s]['confidence'] for s in SERVICES]
clrs2       = [SEV_COLORS[diagnosis_report[s]['severity']] for s in SERVICES]
bars = ax.barh(SERVICES, conf_vals, color=clrs2, alpha=0.85, edgecolor='black', lw=0.8)
ax.set_xlabel('Karar Guveni')
ax.set_title('Diagnosis & Karar Guveni')
ax.set_xlim(0, 1.4)
ax.grid(True, alpha=0.3, axis='x')
for bar, label, val in zip(bars, diag_labels, conf_vals):
    ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
            f'{label}\n({val:.2f})', va='center', fontsize=8)
patches = [mpatches.Patch(color=c, label=l)
           for c, l in [('#e74c3c','Critical'),('#f39c12','Warning'),
                        ('#2ecc71','OK'),('#3498db','Info')]]
ax.legend(handles=patches, fontsize=8, loc='lower right')

plt.tight_layout()
plt.savefig('xai_diagnosis_plot.png', dpi=150, bbox_inches='tight')
plt.close()

with open('xai_diagnosis_report.json', 'w', encoding='utf-8') as f:
    json.dump(diagnosis_report, f, indent=2, ensure_ascii=False)
df_rows.to_csv('xai_diagnosis_summary.csv', index=False)

print(f"\n  OK xai_diagnosis_report.json")
print(f"  OK xai_diagnosis_summary.csv")
print(f"  OK xai_diagnosis_plot.png")
print(f"\n{'=' * 65}")
print("Adim 3 tamamlandi. Siradaki: step4_evidence_guided_llm.py")
print("=" * 65)
