# MARL Eğitimi — Notlar ve Kararlar

## Klasördeki Dosyalar

```
marl_env.py                          → MARL ortamı (Digital Twin üzerinde çalışır)
ippo_agent.py                        → IPPO ajan implementasyonu
mappo_agent.py                       → MAPPO ajan implementasyonu
dqn_agent.py                         → DQN ajan implementasyonu
a2c_agent.py                         → A2C ajan implementasyonu
qmix_agent.py                        → QMIX ajan implementasyonu
train_ippo_30k.py                    → IPPO eğitim scripti
train_mappo_new.py                   → MAPPO eğitim scripti
train_dqn_new.py                     → DQN eğitim scripti
train_a2c_new.py                     → A2C eğitim scripti
train_qmix_new.py                    → QMIX eğitim scripti
evaluate_new.py                      → Karşılaştırmalı evaluate scripti (kullanılan)
evaluate.py                          → Eski versiyon (kullanılmadı)
evaluation_summary_final.csv         → Ana sonuç dosyası
evaluation_plot_final.png            → Karşılaştırma grafikleri
checkpoints_ippo_new/                → IPPO checkpoint'leri
checkpoints_mappo_new/               → MAPPO checkpoint'leri
checkpoints_dqn_new/                 → DQN checkpoint'leri
checkpoints_a2c_new/                 → A2C checkpoint'leri
checkpoints_qmix_new/                → QMIX checkpoint'leri
training_log_*.csv                   → Epoch bazında eğitim logları
training_plot_*.png                  → Eğitim grafikleri
digital_twin_best.pth                → DT klasöründen kopyalandı
scaler.pkl / dt_scaler.pkl           → SAR/DT klasöründen kopyalandı
```

---

## Ortam: MARL Env (marl_env.py)

Gerçek Kubernetes cluster'a dokunmadan Digital Twin üzerinde eğitim yapıldı.

**Temel ayarlar:**
```
MAX_STEPS = 6   ← Trajectory validation kararı (Step 6 MAE=0.076 < 0.08)
SEED      = 42  ← marl_env.py'de mevcut
```

**Partial Observability — her ajan sadece komşularını görüyor:**

| Ajan | Gözlemlediği Servisler | Obs Boyutu |
|---|---|---|
| cart | cart, catalogue, shipping | 15 |
| catalogue | catalogue, ratings, cart | 15 |
| payment | payment, shipping | 10 |
| shipping | shipping, cart, payment | 15 |
| ratings | ratings, catalogue | 10 |
| user | user, cart | 10 |

**Aksiyon uzayı:**
```
0: scale_down  (-1 pod)
1: hold        (değişim yok)
2: scale_up    (+1 pod)
```

---

## Eğitilen Algoritmalar

### IPPO (Independent PPO)
Her servis için ayrı Actor-Critic. Partial observation.

| Metrik | Başlangıç | Son |
|---|---|---|
| Reward (ilk 100 ep) | -3.744 | — |
| Reward (son 100 ep) | — | **+4.921** |
| Entropy | 1.097 | **0.364** |
| CriticLoss | 4.787 | **1.342** |

Checkpoint: `checkpoints_ippo_new/ippo_new_final.pth`

---

### MAPPO (Multi-Agent PPO)
Actor: partial observation. **Critic: global state (CTDE — Centralized Training Decentralized Execution).**

| Metrik | Başlangıç | Son |
|---|---|---|
| Reward (ilk 100 ep) | -4.005 | — |
| Reward (son 100 ep) | — | **+5.923** |
| Entropy | 1.097 | **0.142** |
| CriticLoss | 5.830 | **0.641** |

IPPO'dan daha yüksek reward, daha düşük entropy — global bilgiyle daha kararlı policy öğrendi.

Checkpoint: `checkpoints_mappo_new/mappo_new_final.pth`

---

### DQN (Independent DQN)
Replay buffer, ε-greedy exploration.

| Metrik | Başlangıç | Son |
|---|---|---|
| Reward (ilk 100 ep) | +1.874 | — |
| Reward (son 100 ep) | — | **+2.138** |
| Epsilon | 0.75 | **0.05** (min'e ulaştı) |

Epsilon çok hızlı düştü — exploration yetersiz kaldı, IPPO/MAPPO'nun gerisinde.

Checkpoint: `checkpoints_dqn_new/dqn_new_final.pth`

---

### A2C (Independent A2C)
Her adımda anlık TD güncelleme.

| Metrik | Başlangıç | Son |
|---|---|---|
| Reward (ilk 100 ep) | -3.894 | — |
| Reward (son 100 ep) | — | **+5.117** |
| CriticLoss | 0.39 | ~280 (unstable) |

Reward IPPO ile benzer ama CriticLoss patladı — training boyunca instability gözlemlendi.

Checkpoint: `checkpoints_a2c_new/a2c_new_final.pth`

---

### QMIX
Global Q-value mixing network.

| Metrik | Başlangıç | Son |
|---|---|---|
| Reward (ilk 100 ep) | -5.063 | — |
| Reward (son 100 ep) | — | **-7.596** |
| Loss | ~0.5 | **~8764** (diverge) |

Loss tamamen patladı. Evaluate'de en kötü sonuç. **Analizden hariç tutuldu.**

Checkpoint: `checkpoints_qmix_new/qmix_new_final.pth` (saklandı, kullanılmadı)

---

## Evaluate Sonuçları (evaluate_new.py)

200 episode, greedy policy, 7 politika karşılaştırması.

| Politika | Ort. Reward | SLO İhlal % |
|---|---|---|
| **IPPO** | **+2.059** | **%13.2** |
| **MAPPO** | **+2.059** | **%13.2** |
| **A2C** | **+2.059** | **%13.2** |
| DQN | -7.236 | %26.4 |
| Random | -11.726 | %29.4 |
| Rule-based | -9.206 | %34.9 |
| QMIX | -13.290 | %32.5 |

> IPPO/MAPPO/A2C evaluate'de aynı skoru aldı — greedy modda aynı aksiyonlara converge ettiler.

**SLO iyileşmesi:**
```
Rule-based: %34.9  →  MARL: %13.2  →  %62 iyileşme
```

**Latency iyileşmeleri:**

| Servis | Rule-based | MARL | İyileşme |
|---|---|---|---|
| cart | 0.1136 | 0.0619 | **-45.5%** |
| catalogue | 0.1034 | 0.0629 | **-39.2%** |
| user | 0.0663 | 0.0435 | **-34.4%** |
| shipping | 0.2912 | 0.2201 | **-24.4%** |
| ratings | 0.2876 | 0.2501 | **-13.0%** |
| payment | 0.0188 | 0.0162 | -13.8% (reward dışı) |

*(değerler normalize [0,1] aralığında)*

---

## Öğrenme Kanıtları

| Kanıt | Değer |
|---|---|
| Reward artışı | -3.744 → +4.921 (IPPO) |
| SLO düşüşü | %34.9 → %13.2 (%62 iyileşme) |
| Entropy düşüşü | 1.097 → 0.364 (rastgeleden kararlılığa) |
| Karar güveni | cart/catalogue: 0.977-0.979 |
| SHAP koordinasyon | user→CAR_Pods=0.0246 (emergent) |

---

## Hangi Checkpoint Kullanılacak?

**XAI ve değerlendirme için:**
```
checkpoints_ippo_new/ippo_new_final.pth   ← kullan
checkpoints_mappo_new/mappo_new_final.pth ← kullan
checkpoints_dqn_new/dqn_new_final.pth    ← kullan
checkpoints_a2c_new/a2c_new_final.pth    ← kullan
```

500/1000/1500/2000/2500/3000.pth dosyaları sadece eğitim sırasındaki anlık kaydedmeler — ihtiyaç yoksa silinebilir.

---

## Tekrar Eğitim Gerekiyor Mu?

**Hayır.** Sonuçlar iyi, tüm checkpoint'ler mevcut. QMIX diverge etti ama bu beklenen — tezde sınırlılık olarak belirtildi.

**Not:** Train script'lerinde random seed kontrolü tam değil. Tekrar çalıştırılırsa sonuçlar benzer ama birebir aynı olmayabilir.

---

## Sıradaki Adım

XAI pipeline:
```
step1_behavior_dataset.py
step2_shap.py
step3_diagnosis_engine.py
step4_evidence_guided_llm.py
```
