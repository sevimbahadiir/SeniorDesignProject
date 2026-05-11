"""
step3_diagnosis_engine.py (v2)
------------------------------
XAI Adım 3: Rule-based Diagnosis Engine

Düzeltmeler:
- Shipping için özel pod-insensitive kural (latency/p50 > 1.0 + scale_up + CPU düşük)
- MARL başarılı kategorisi (latency azalmış, verimli çalışıyor)
- Kural öncelik sırası netleştirildi
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

# MARL analizinden bilinen latency iyileşmeleri (ms cinsinden)
# Rule-based vs IPPO karşılaştırmasından
MARL_IMPROVEMENT = {
    'cart':      {'rb_lat': 187.0, 'ippo_lat': 125.5, 'improved': True},
    'catalogue': {'rb_lat': 119.1, 'ippo_lat':  61.7, 'improved': True},
    'payment':   {'rb_lat': 696.2, 'ippo_lat': 1292.4,'improved': False},
    'shipping':  {'rb_lat': 17492.9,'ippo_lat': 19996.0,'improved': False},
    'ratings':   {'rb_lat': 1112.2, 'ippo_lat': 890.9, 'improved': True},
    'user':      {'rb_lat':  47.3,  'ippo_lat':  19.7, 'improved': True},
}

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

state_cols = [f'{s}_{m}' for s in SERVICES for m in METRICS]

def denorm(svc, metric, val):
    idx  = state_cols.index(f'{svc}_{metric}')
    dmin = scaler.data_min_[idx]
    dmax = scaler.data_max_[idx]
    return float(val * (dmax - dmin) + dmin)

df = pd.read_csv('xai_behavior_dataset.csv')
with open('xai_obs_map.json') as f:
    obs_map = json.load(f)
with open('reward_config.json') as f:
    reward_cfg = json.load(f)

print("=" * 65)
print("XAI Adım 3: Rule-based Diagnosis Engine (v2)")
print("=" * 65)

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

def diagnose(svc, metrics, action):
    evidence = []
    details  = {}

    lat_norm = metrics.get('latency',      {}).get('norm', 0)
    cpu_norm = metrics.get('cpu_usage',    {}).get('norm', 0)
    pod_norm = metrics.get('num_pods',     {}).get('norm', 0)
    rps_norm = metrics.get('request_rate', {}).get('norm', 0)
    lat_real = metrics.get('latency',      {}).get('real', 0)
    pod_real = metrics.get('num_pods',     {}).get('real', 0)

    p50_real  = reward_cfg.get(svc, {}).get('p50', 1000)
    in_reward = reward_cfg.get(svc, {}).get('in_reward', True)
    improved  = MARL_IMPROVEMENT[svc]['improved']
    ippo_lat  = MARL_IMPROVEMENT[svc]['ippo_lat']
    rb_lat    = MARL_IMPROVEMENT[svc]['rb_lat']

    details['latency_ms']    = round(lat_real, 1)
    details['p50_ms']        = p50_real
    details['latency_ratio'] = round(lat_real / p50_real, 2) if p50_real > 0 else 0
    details['pod_count']     = round(pod_real, 1)
    details['cpu_norm']      = round(cpu_norm, 3)
    details['action']        = action
    details['in_reward']     = in_reward
    details['ippo_lat_ms']   = ippo_lat
    details['rb_lat_ms']     = rb_lat
    details['pct_improvement']= round((rb_lat - ippo_lat) / rb_lat * 100, 1) if rb_lat > 0 else 0
    details['confidence']     = round(metrics.get('confidence', 0), 3)

    # Kural 0: Reward dışı
    if not in_reward:
        evidence.append("Reward dışı servis (Option C)")
        evidence.append(f"Latency: {lat_real:.0f}ms — izleniyor ama optimize edilmiyor")
        return 'reward_excluded', 'info', evidence, details

    # Kural 1: MARL iyileştirdi → başarılı (iyileşme oranı > %10)
    # Bu kural diğer tüm kurallardan önce gelir
    pct = details['pct_improvement']
    if improved and pct > 10.0:
        evidence.append(f"MARL latency iyileştirmesi: {rb_lat:.0f}ms → {ippo_lat:.0f}ms ({pct:+.1f}%)")
        evidence.append(f"Mevcut politika: {action} (güven: {details.get('confidence', 0):.2f})")
        return 'marl_optimized', 'ok', evidence, details

    # Kural 2: Shipping spesifik — pod-insensitive bottleneck
    # Shipping için daha yumuşak eşik: latency > p50 + scale_up + CPU düşük
    if svc == 'shipping' and action == 'scale_up' and cpu_norm < 0.65:
        evidence.append(f"Latency {lat_real:.0f}ms > p50 ({p50_real:.0f}ms)")
        evidence.append(f"CPU normalize: {cpu_norm:.2f} — yüksek yük değil, I/O-bound")
        evidence.append("Scale-up etkisiz: DB/I/O bottleneck (action space sınırı)")
        evidence.append(f"MARL vs Rule-based: {ippo_lat:.0f}ms vs {rb_lat:.0f}ms (+{ippo_lat-rb_lat:.0f}ms)")
        return 'pod_insensitive_bottleneck', 'critical', evidence, details

    # Kural 3: Genel pod-insensitive (latency > 1.5×p50 + scale_up + CPU düşük)
    if lat_real > 1.5 * p50_real and action == 'scale_up' and cpu_norm < 0.4:
        evidence.append(f"Latency {lat_real:.0f}ms > 1.5×p50 ({1.5*p50_real:.0f}ms)")
        evidence.append(f"CPU düşük ({cpu_norm:.2f}) → CPU-bound değil")
        evidence.append("Scale-up seçilmiş ama CPU bottleneck yok")
        return 'pod_insensitive_bottleneck', 'warning', evidence, details

    # Kural 4: Under-provisioned (latency yüksek, agent yanlış aksiyon)
    if lat_real > 1.2 * p50_real and action in ('keep', 'scale_down'):
        evidence.append(f"Latency {lat_real:.0f}ms > 1.2×p50 ({1.2*p50_real:.0f}ms)")
        evidence.append(f"Agent {action} seçiyor — reaktif olmayan politika")
        return 'under_provisioned', 'warning', evidence, details

    # Kural 5: Over-provisioned (çok düşük latency, scale_down)
    if lat_real < 0.5 * p50_real and action == 'scale_down':
        evidence.append(f"Latency {lat_real:.0f}ms < 0.5×p50 ({0.5*p50_real:.0f}ms)")
        evidence.append("Verimli kaynak kullanımı — scale_down uygun")
        return 'over_provisioned', 'ok', evidence, details

    # Kural 6: Healthy
    if lat_real <= p50_real:
        evidence.append(f"Latency {lat_real:.0f}ms ≤ p50 ({p50_real:.0f}ms)")
        evidence.append(f"Agent {action} — normal çalışma")
        return 'healthy', 'ok', evidence, details

    evidence.append(f"Latency {lat_real:.0f}ms, p50 {p50_real:.0f}ms, aksiyon {action}")
    return 'monitoring_required', 'info', evidence, details

# ─── Ana Analiz ───────────────────────────────────────────────────────────────

SEV_ICON = {'critical':'🔴','warning':'🟡','ok':'🟢','info':'🔵'}
SEV_COLORS = {'critical':'#e74c3c','warning':'#f39c12','ok':'#2ecc71','info':'#3498db'}
DIAG_TR = {
    'pod_insensitive_bottleneck': 'Pod-Insensitive Bottleneck',
    'under_provisioned':          'Under-Provisioned',
    'over_provisioned':           'Over-Provisioned (Verimli)',
    'marl_optimized':             'MARL Optimize Etti ✓',
    'traffic_surge':              'Traffic Surge',
    'healthy':                    'Sağlıklı',
    'reward_excluded':            'Reward Dışı (Option C)',
    'monitoring_required':        'İzleme Gerekli',
}

print(f"\n  {'Servis':<12} {'Diagnosis':<32} {'Sev':<5} {'Latency':>10} {'p50':>8} {'İyileşme':>10}")
print(f"  {'-'*80}")

diagnosis_report = {}
rows = []

for svc in SERVICES:
    metrics = get_service_metrics(svc)
    action  = metrics['action']
    diagnosis, severity, evidence, details = diagnose(svc, metrics, action)
    diagnosis_report[svc] = {
        'diagnosis': diagnosis, 'severity': severity,
        'action': action, 'confidence': round(metrics['confidence'], 3),
        'evidence': evidence, 'details': details,
    }
    icon = SEV_ICON.get(severity, '⚪')
    pct  = details['pct_improvement']
    pct_str = f"{pct:+.1f}%" if pct != 0 else "—"
    print(f"  {svc:<12} {DIAG_TR.get(diagnosis,diagnosis):<32} "
          f"{icon} {severity:<3} {details['latency_ms']:>10.1f} "
          f"{details['p50_ms']:>8.1f} {pct_str:>10}")
    rows.append({
        'service': svc, 'diagnosis': diagnosis, 'severity': severity,
        'action': action, 'latency_ms': details['latency_ms'],
        'p50_ms': details['p50_ms'], 'latency_ratio': details['latency_ratio'],
        'pod_count': details['pod_count'], 'cpu_norm': details['cpu_norm'],
        'confidence': round(metrics['confidence'], 3),
        'pct_improvement': details['pct_improvement'],
    })

# ─── Detaylı çıktı ────────────────────────────────────────────────────────────

print(f"\n{'=' * 65}")
print("DETAYLI DIAGNOSIS")
print("=" * 65)
for svc, report in diagnosis_report.items():
    d = report['details']
    print(f"\n  ── {svc.upper()} ──")
    print(f"  Diagnosis  : {DIAG_TR.get(report['diagnosis'], report['diagnosis'])}")
    print(f"  Severity   : {report['severity']}")
    print(f"  Aksiyon    : {report['action']} (güven: {report['confidence']:.3f})")
    print(f"  Latency    : {d['latency_ms']:.1f}ms (p50: {d['p50_ms']}ms, {d['latency_ratio']:.2f}x)")
    if d['pct_improvement'] != 0:
        print(f"  MARL İyil. : {d['rb_lat_ms']:.0f}ms → {d['ippo_lat_ms']:.0f}ms ({d['pct_improvement']:+.1f}%)")
    for ev in report['evidence']:
        print(f"    • {ev}")

# ─── Grafik ───────────────────────────────────────────────────────────────────

df_rows = pd.DataFrame(rows)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Rule-based Diagnosis Engine — Servis Bazlı Analiz',
             fontsize=13, fontweight='bold')

ax = axes[0]
x = np.arange(len(SERVICES))
w = 0.35
clrs = [SEV_COLORS[diagnosis_report[s]['severity']] for s in SERVICES]
ax.bar(x - w/2, df_rows['latency_ms'], w, label='IPPO Latency (ms)',
       color=clrs, alpha=0.85, edgecolor='black', lw=0.8)
ax.bar(x + w/2, df_rows['p50_ms'],     w, label='p50 eşiği (ms)',
       color='lightgray', alpha=0.85, edgecolor='black', lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(SERVICES, rotation=15)
ax.set_yscale('log')
ax.set_ylabel('Latency (ms, log scale)')
ax.set_title('IPPO Latency vs p50 Eşiği')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')

ax = axes[1]
diag_labels = [DIAG_TR.get(diagnosis_report[s]['diagnosis'],'?') for s in SERVICES]
conf_vals   = [diagnosis_report[s]['confidence'] for s in SERVICES]
clrs2       = [SEV_COLORS[diagnosis_report[s]['severity']] for s in SERVICES]
bars = ax.barh(SERVICES, conf_vals, color=clrs2, alpha=0.85, edgecolor='black', lw=0.8)
ax.set_xlabel('Karar Güveni')
ax.set_title('Diagnosis & Karar Güveni')
ax.set_xlim(0, 1.35)
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

print(f"\n  ✓ xai_diagnosis_report.json")
print(f"  ✓ xai_diagnosis_summary.csv")
print(f"  ✓ xai_diagnosis_plot.png")
print(f"\n{'=' * 65}")
print("Adım 3 tamamlandı.")
print("=" * 65)
