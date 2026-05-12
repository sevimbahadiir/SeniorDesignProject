"""
step4_evidence_guided_llm.py
----------------------------
XAI Adim 4: Evidence-Guided LLM Verbalization
Rate limit icin retry mekanizmasi eklendi.

Kullanim:
    export OPENROUTER_API_KEY="sk-or-..."
    python3 step4_evidence_guided_llm.py
"""

import os, json, time
import pandas as pd
import requests

SERVICES       = ['cart', 'catalogue', 'payment', 'shipping', 'ratings', 'user']
OPENROUTER_MODEL = 'openai/gpt-oss-20b:free'
MAX_RETRIES    = 5
RETRY_DELAY    = 15  # saniye

DIAG_LABELS = {
    'pod_insensitive_bottleneck': 'Pod-Insensitive Bottleneck',
    'marl_optimized':             'MARL Optimized',
    'reward_excluded':            'Reward Excluded (Option C)',
    'under_provisioned':          'Under-Provisioned',
    'over_provisioned':           'Over-Provisioned',
    'healthy':                    'Healthy',
    'monitoring_required':        'Monitoring Required',
}

def load_shap_features(service, top_k=5):
    try:
        df = pd.read_csv('xai_output/xai_results.csv')
        svc_df = (df[df['service'] == service]
                  .sort_values('mean_abs_shap', ascending=False)
                  .head(top_k))
        return [{'feature': row['feature'],
                 'score':   round(float(row['mean_abs_shap']), 6),
                 'rank':    int(row['rank']),
                 'importance_type': 'KernelSHAP'}
                for _, row in svc_df.iterrows()]
    except:
        return []

def build_evidence_pack():
    with open('xai_diagnosis_report.json', 'r', encoding='utf-8') as f:
        diagnosis_report = json.load(f)
    evidence_pack = {}
    for service in SERVICES:
        report  = diagnosis_report[service]
        details = report['details']
        evidence_pack[service] = {
            'service':         service,
            'diagnosis':       report['diagnosis'],
            'diagnosis_label': DIAG_LABELS.get(report['diagnosis'], report['diagnosis']),
            'severity':        report['severity'],
            'action':          report['action'],
            'confidence':      report['confidence'],
            'latency_ms':      details.get('latency_ms'),
            'p50_ms':          details.get('p50_ms'),
            'latency_ratio':   details.get('latency_ratio'),
            'cpu_norm':        details.get('cpu_norm'),
            'pod_count':       details.get('pod_count'),
            'pct_improvement': details.get('pct_improvement'),
            'rb_lat_ms':       details.get('rb_lat_ms'),
            'ippo_lat_ms':     details.get('ippo_lat_ms'),
            'rule_evidence':   report.get('evidence', []),
            'top_features':    load_shap_features(service, top_k=5),
        }
    with open('xai_evidence_pack.json', 'w', encoding='utf-8') as f:
        json.dump(evidence_pack, f, indent=2, ensure_ascii=False)
    print("  OK xai_evidence_pack.json")
    return evidence_pack

def template_explanation(ev):
    service    = ev['service']
    diagnosis  = ev['diagnosis_label']
    action     = ev['action']
    confidence = ev['confidence']
    latency    = ev['latency_ms']
    p50        = ev['p50_ms']
    ratio      = ev['latency_ratio']
    improvement= ev['pct_improvement']
    top_feats  = ', '.join([f["feature"] for f in ev['top_features'][:3]])

    if ev['diagnosis'] == 'pod_insensitive_bottleneck':
        return (f"The {service} service is classified as {diagnosis}. "
                f"The MARL agent selected '{action}' with confidence {confidence:.2f}, "
                f"while latency remained at {latency:.1f} ms against p50 of {p50:.1f} ms ({ratio:.2f}x). "
                f"Top SHAP features: {top_feats}. "
                f"Increasing pod count alone may not resolve the degradation.")
    if ev['diagnosis'] == 'marl_optimized':
        return (f"The {service} service is classified as {diagnosis}. "
                f"The IPPO policy selected '{action}' with confidence {confidence:.2f}. "
                f"Latency improved by {improvement:.1f}% vs rule-based "
                f"({ev['rb_lat_ms']:.0f} ms -> {ev['ippo_lat_ms']:.0f} ms). "
                f"Top SHAP features: {top_feats}.")
    if ev['diagnosis'] == 'reward_excluded':
        return (f"The {service} service is marked as {diagnosis}. "
                f"Excluded from reward function (Option C). "
                f"Agent selected '{action}' with confidence {confidence:.2f}; "
                f"latency was {latency:.1f} ms.")
    return (f"The {service} service: {diagnosis}. "
            f"Action: '{action}' (conf: {confidence:.2f}). "
            f"Latency: {latency:.1f} ms / p50: {p50:.1f} ms.")

def build_prompt(ev):
    top_features  = "\n".join([
        f"- {f['feature']} (KernelSHAP: {f['score']:.6f}, rank {f['rank']})"
        for f in ev['top_features']
    ])
    rule_evidence = "\n".join([f"- {e}" for e in ev['rule_evidence']])
    return f"""You are writing a concise XAI explanation for an academic paper on MARL-based Kubernetes autoscaling.
Only interpret the provided evidence. Do not invent new diagnoses.

Service: {ev['service']} | Diagnosis: {ev['diagnosis_label']} | Severity: {ev['severity']}
Action: {ev['action']} (confidence: {ev['confidence']})
Latency: {ev['latency_ms']} ms | p50: {ev['p50_ms']} ms | Ratio: {ev['latency_ratio']}x
Improvement vs Rule-based: {ev['pct_improvement']:+.1f}% ({ev['rb_lat_ms']:.0f}->{ev['ippo_lat_ms']:.0f} ms)

Rule Evidence:
{rule_evidence}

Top KernelSHAP Features:
{top_features}

Write 3-4 sentences for an academic paper.""".strip()

def call_openrouter(prompt):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None, "no_api_key"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "http://localhost",
        "X-Title":       "XAI MARL",
    }
    payload = {
        "model":       OPENROUTER_MODEL,
        "messages":    [
            {"role": "system",
             "content": "Academic technical writer. Only interpret evidence provided."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens":  500,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers, json=payload, timeout=60
            )
            if r.status_code == 200:
                text = r.json()['choices'][0]['message']['content']
                return text.strip(), f"openrouter/{OPENROUTER_MODEL}"
            elif r.status_code == 429:
                wait = RETRY_DELAY * attempt
                print(f"    429 rate limit — {wait}s bekleniyor (deneme {attempt}/{MAX_RETRIES})...")
                time.sleep(wait)
            else:
                print(f"    Hata {r.status_code}: {r.text[:150]}")
                return None, f"error_{r.status_code}"
        except Exception as e:
            print(f"    Baglanti hatasi: {e}")
            time.sleep(RETRY_DELAY)

    return None, "max_retries_exceeded"

def main():
    print("=" * 65)
    print("XAI Adim 4: Evidence-Guided LLM Verbalization")
    print("=" * 65)

    evidence_pack = build_evidence_pack()
    explanations  = {}

    api_key = os.getenv("OPENROUTER_API_KEY")
    use_api = api_key is not None
    print(f"\n  Model   : {OPENROUTER_MODEL}")
    print(f"  API     : {'AKTIF' if use_api else 'YOK — template fallback'}")
    print(f"  Retry   : maks {MAX_RETRIES}x, {RETRY_DELAY}s aralikla\n")

    for service, ev in evidence_pack.items():
        print(f"  [{service}] isleniyor...")
        prompt = build_prompt(ev)

        if use_api:
            llm_text, method = call_openrouter(prompt)
        else:
            llm_text, method = None, "no_api_key"

        if llm_text:
            explanation = llm_text
        else:
            explanation = template_explanation(ev)
            method      = "template_fallback"

        explanations[service] = {
            'diagnosis':    ev['diagnosis_label'],
            'severity':     ev['severity'],
            'method':       method,
            'explanation':  explanation,
            'top_features': ev['top_features'],
        }
        print(f"    method: {method}")
        print(f"    {explanation[:180]}...\n")

        # Servisler arasi kisa bekleme (rate limit onlemi)
        if use_api and llm_text:
            time.sleep(3)

    with open('xai_evidence_guided_explanations.json', 'w', encoding='utf-8') as f:
        json.dump(explanations, f, indent=2, ensure_ascii=False)

    icon_map = {'ok':'[OK]','info':'[INFO]','warning':'[WARN]','critical':'[CRIT]'}
    lines = [
        "=" * 80,
        "XAI EVIDENCE-GUIDED EXPLANATION REPORT",
        "IPPO Policy Analysis — Robot Shop Microservice Autoscaling",
        f"Model: {OPENROUTER_MODEL}",
        "=" * 80, "",
        "Diagnoses: deterministic rule-based engine.",
        "LLM: interprets pre-computed evidence only.",
        "",
    ]
    for service, item in explanations.items():
        icon = icon_map.get(item['severity'], '[?]')
        lines += [
            f"{icon} {service.upper()} — {item['diagnosis']}",
            "-" * 60,
            f"Method: {item['method']}",
            "",
            item['explanation'],
            "",
            "Top SHAP Features:",
        ]
        for f in item['top_features']:
            lines.append(f"  {f['rank']}. {f['feature']} (score: {f['score']:.6f})")
        lines.append("")
    lines += ["=" * 80, "End of Report", "=" * 80]

    with open('xai_evidence_guided_report.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    print("  OK xai_evidence_guided_explanations.json")
    print("  OK xai_evidence_guided_report.txt")
    print(f"\n{'=' * 65}")
    print("Adim 4 tamamlandi.")
    print("=" * 65)

if __name__ == '__main__':
    main()
