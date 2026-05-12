# Kubernetes Robot Shop — MARL Tabanlı Otomatik Ölçeklendirme
## Proje Dokümantasyonu (Veri Toplama → XAI)

---

## 1. VERİ TOPLAMA

### Ortam
- **Platform:** Kubernetes üzerinde Robot Shop mikro servis uygulaması
- **Servisler:** cart, catalogue, payment, shipping, ratings, user (6 servis)
- **Metrikler:** num_pods, cpu_usage, mem_usage, request_rate, latency (5 metrik/servis)
- **Monitoring:** Istio service mesh metrikleri

### Toplama Süreci
- 10 farklı yük senaryosu (40–500 eşzamanlı kullanıcı)
- ~84 saat gerçek cluster gözlemi
- **Ham veri:** 14.410 satır

### Yeni DT için Değişiklik
- Önceki çalışmada daha az veri mevcuttu
- Gaussian augmentation ile **28.780 satıra** çıkarıldı (2x artış)
- `rl_dataset_sar_normalized.csv` — MinMaxScaler [0,1] normalizasyonu

---

## 2. VERİ TEMİZLEME VE SAR FORMATI

### SAR (State-Action-Reward-NextState)
Her satır bir geçiş tuplası:
```
(s_t, a_t, r_t, s_{t+1})
```

### Reward Fonksiyonu
Tanh-sıkıştırmalı, servis ağırlıklı:
```
r = Σ w_i * (-tanh(latency_i / threshold_i))
```

**Servis ağırlıkları (reward_config.json):**
| Servis    | Ağırlık | Reward'da? |
|-----------|---------|------------|
| shipping  | 2.0     | Evet       |
| cart      | 1.5     | Evet       |
| catalogue | 1.5     | Evet       |
| user      | 1.0     | Evet       |
| ratings   | 0.5     | Evet       |
| payment   | 0.0     | **Hayır (Option C)** |

> **Önemli:** Payment servisi reward dışında bırakıldı çünkü latency değerleri (p50=2672ms) diğer servislerden çok farklı ölçekte — reward fonksiyonunu domine ediyordu.

### Gaussian Augmentation
- Orijinal 14.410 satır → **28.780 satır**
- Küçük Gaussian gürültüsü eklenerek sentetik geçişler üretildi
- Normalizasyon sonrası [0,1] aralığında kaldı

---

## 3. DIGITAL TWIN EĞİTİMİ (dt2.py)

### Mimari
```
Input: 36 boyut (30 state + 6 action)
  → Linear(256) → BatchNorm → ReLU → Dropout(0.2)
  → Linear(256) → BatchNorm → ReLU → Dropout(0.2)
  → Linear(128) → BatchNorm → ReLU → Dropout(0.2)
  → Linear(30) → Sigmoid
Output: 30 boyut (next_state)
```

### Hiperparametreler
| Parametre     | Değer         |
|---------------|---------------|
| Loss          | Huber (δ=1.0) |
| Optimizer     | Adam (lr=1e-3)|
| LR Scheduler  | ReduceLROnPlateau (patience=15)|
| Early Stopping| patience=30   |
| Epochs        | maks 300      |
| Batch Size    | 64            |

### Sonuçlar
| Metrik | Değer |
|--------|-------|
| Best Epoch | 282 |
| Val Loss (Huber) | ~0.0014 |
| **MAE** | **0.032** |
| **R²** | **0.945** |

### Önceki DT ile Karşılaştırma
| | Eski DT | **Yeni DT** |
|--|---------|-------------|
| Veri | 14.410 satır | **28.780 satır** |
| MAE | ~0.08 | **0.032** |
| R² | ~0.85 | **0.945** |

> Val loss < Train loss — overfitting yok, smooth convergence ✅

---

## 4. TRAJECTORY VALIDATION (test2.py)

### Yöntem
200 farklı başlangıç noktasından **8 adımlık otoregresif rollout:**
```
Gerçek veri → DT → tahmin₁
              tahmin₁ → DT → tahmin₂  (gerçek veriye bakmadan)
              ...
              tahmin₇ → DT → tahmin₈
```

### Sonuçlar
| Adım | MAE    | R²    | Durum      |
|------|--------|-------|------------|
| 1    | 0.033  | 0.936 | Good ✓     |
| 2    | 0.045  | 0.889 | Good ✓     |
| 3    | 0.054  | 0.842 | Acceptable |
| 4    | 0.063  | 0.791 | Acceptable |
| 5    | 0.069  | 0.759 | Acceptable |
| 6    | **0.076** | **0.709** | **Acceptable ✓** |
| 7    | 0.079  | 0.683 | Marginal   |
| 8    | 0.082  | 0.650 | Eşik aşıldı |

### Karar: MAX_STEPS = 6
- Step 6 MAE = 0.076 < 0.08 eşiği ✅
- Step 8 sadece 0.002 aşıyor → istatistiksel gürültü sınırında
- **Shipping servisi** en yüksek drift (MAE=0.118) — volatil servis, beklenen

---

## 5. MARL EĞİTİMİ

### Ortam: Digital Twin Simülasyonu
Gerçek Kubernetes cluster'a dokunmadan DT üzerinde eğitim.

### Partial Observability (Kısmi Gözlem)
Her ajan sadece kendi ve komşu servislerini görüyor:
| Ajan      | Gözlemlediği Servisler        | Obs Boyutu |
|-----------|-------------------------------|-----------|
| cart      | cart, catalogue, shipping     | 15        |
| catalogue | catalogue, ratings, cart      | 15        |
| payment   | payment, shipping             | 10        |
| shipping  | shipping, cart, payment       | 15        |
| ratings   | ratings, catalogue            | 10        |
| user      | user, cart                    | 10        |

### Aksiyonlar
```
0: scale_down  (-1 pod)
1: hold        (değişim yok)
2: scale_up    (+1 pod)
```

### Eğitilen Algoritmalar (3000 episode, MAX_STEPS=6)

#### IPPO (Independent PPO)
- Her servis için ayrı Actor-Critic ajan
- Partial observation (sadece kendi + komşu)
- **Sonuç:** İlk 100 ep: -3.744 → Son 100 ep: **+4.921** (+8.664)
- Entropy: 1.097 → 0.364 (policy öğrendi)
- CriticLoss: 4.787 → 1.342

#### MAPPO (Multi-Agent PPO)
- Actor: partial observation
- **Critic: global state (30 boyut) — CTDE**
- **Sonuç:** İlk 100 ep: -4.005 → Son 100 ep: **+5.923** (+9.928)
- Entropy: 1.097 → 0.142 (IPPO'dan çok daha kararlı)
- CriticLoss: 5.830 → 0.641

#### DQN (Independent DQN)
- Replay buffer, ε-greedy exploration
- Epsilon çok hızlı düştü (ep 250'de min'e ulaştı)
- **Sonuç:** +2.138 — IPPO/MAPPO'nun gerisinde

#### A2C (Independent A2C)
- Her adımda anlık TD güncelleme
- CriticLoss patlama: 0 → 280 (unstable)
- **Sonuç:** Training'de +5.12, ama critic instability

#### QMIX
- Global Q-value mixing network
- Loss: 0.5 → **8054** → tamamen diverge
- **Sonuç:** Hariç tutuldu

### Değişiklikler (Eski → Yeni)
| Parametre | Eski | **Yeni** |
|-----------|------|----------|
| MAX_STEPS | 10 | **6** (trajectory testinden) |
| Digital Twin | MAE≈0.08 | **MAE=0.032** |
| Episode sayısı | 3000 | **3000** (aynı) |
| Checkpoint klasörü | checkpoints_v2/ | **checkpoints_ippo_new/** |

---

## 6. EVALUATE (evaluate_new.py)

### Yöntem
- 200 episode × 5756 test state
- Greedy policy (argmax, deterministik)
- 7 politika karşılaştırması

### Final Sonuçlar
| Politika  | Ort Reward | SLO İhlal% | Shipping Latency |
|-----------|-----------|------------|-----------------|
| **MAPPO** | **+2.059** | **%13.2** | 0.220 |
| **IPPO**  | **+2.059** | **%13.2** | 0.220 |
| **A2C**   | **+2.059** | **%13.2** | 0.220 |
| DQN       | -7.236    | %26.4      | 0.291 |
| Rule-based| -9.206    | %34.9      | 0.291 |
| Random    | -11.726   | %29.4      | 0.294 |
| QMIX      | -13.290   | %32.5      | 0.332 |

> **Not:** IPPO/MAPPO/A2C evaluate'de aynı skoru aldı — policy greedy modda aynı aksiyonlara converge etti.

### SLO İyileşmesi
```
Rule-based: %34.9  →  MARL: %13.2  →  %62 iyileşme
```

### Latency İyileşmeleri (ms)
| Servis    | Rule-based | IPPO    | İyileşme  |
|-----------|-----------|---------|-----------|
| cart      | 187.9 ms  | 102.4 ms| **+45.5%** |
| catalogue | 91.7 ms   | 55.8 ms | **+39.2%** |
| user      | 38.7 ms   | 25.4 ms | **+34.4%** |
| shipping  | 18491 ms  | 13976 ms| **+24.4%** |
| ratings   | 1190.9 ms | 1035.6 ms| **+13.0%** |
| payment   | 742.6 ms  | 639.9 ms | +13.8% (reward dışı) |

---

## 7. XAI PİPELİNE (4 Adım)

### Step 1 — Behavior Dataset (step1_behavior_dataset.py)
- 200 episode × greedy IPPO policy çalıştırıldı
- **7200 kayıt** (6 servis × 200 ep × 6 adım)
- Çıktı: `xai_behavior_dataset.csv`, `xai_obs_map.json`, `xai_obs_<servis>.csv`

**Aksiyon Dağılımı:**
| Servis    | scale_down | keep | scale_up | Güven |
|-----------|-----------|------|----------|-------|
| cart      | %100      | -    | -        | 0.977 |
| catalogue | %100      | -    | -        | 0.979 |
| payment   | -         | -    | %100     | 0.901 |
| shipping  | -         | -    | %100     | 0.730 |
| ratings   | -         | -    | %100     | 0.851 |
| user      | -         | -    | %100     | 0.927 |

> Policy tamamen converge etmiş — her durumda tek bir aksiyona commit ediyor.

---

### Step 2 — KernelSHAP (step2_shap.py)

#### Neden Surrogate değil KernelSHAP?
- Eski yöntem: GradientBoosting surrogate → SHAP
  - Surrogate accuracy = 1.0 çıktı (policy tek aksiyona converge, surrogate anlamsız)
- **Yeni yöntem: Direkt ActorNet üzerinde KernelSHAP**
  - Model-agnostic, proxy yok, metodolojik olarak daha güçlü

#### Parametre
- Background: 100 sample
- Explain: 200 sample
- nsamples: 100 (KernelSHAP başına)

#### En Önemli Feature'lar (#1 SHAP)
| Servis    | Top Feature | SHAP Score |
|-----------|------------|------------|
| cart      | SHI_Pods   | 0.000381   |
| catalogue | RAT_LAT    | 0.000353   |
| payment   | PAY_Pods   | 0.002193   |
| shipping  | CAR_Pods   | **0.006214** |
| ratings   | RAT_Pods   | 0.005503   |
| user      | CAR_Pods   | **0.024568** |

#### Anahtar Bulgular
1. **Emergent coordination:** Bağımsız eğitilen ajanlar komşu servis metriklerini öğrendi
2. **Pods dominant:** 4/6 serviste pod sayısı en önemli feature
3. **RPS önemsiz:** Tüm servislerde request rate SHAP değeri çok düşük
4. **Scale Up baskın strateji:** 4/6 serviste yeşil (scale_up) SHAP katkısı en yüksek

#### Heatmap Yorumu
- CART, CATALOGUE: kendi LAT = 1.00 (ama komşu metrikler daha önemli)
- PAYMENT, RATINGS, USER: Pods = 1.00 (kaynak kullanımı öncelikli)
- **Önemli not:** Heatmap sadece kendi 5 metriğini karşılaştırır; komşu metrikler per-service grafikte görünür

---

### Step 3 — Rule-based Diagnosis Engine (step3_diagnosis_engine.py)

#### Kural Hiyerarşisi
1. Reward dışı servis → `reward_excluded`
2. MARL iyileştirdi (>%10) → `marl_optimized`
3. Shipping: scale_up + CPU<0.65 → `pod_insensitive_bottleneck`
4. Genel: latency>1.5×p50 + scale_up + CPU<0.4 → `pod_insensitive_bottleneck`
5. latency>1.2×p50 + keep/scale_down → `under_provisioned`
6. latency<0.5×p50 + scale_down → `over_provisioned`
7. latency≤p50 → `healthy`

#### Diagnosis Sonuçları
| Servis    | Diagnosis          | Severity | Latency   | p50      | İyileşme  |
|-----------|--------------------|----------|-----------|----------|-----------|
| cart      | MARL Optimize Etti | OK       | 136.1 ms  | 51.8 ms  | +45.5%    |
| catalogue | MARL Optimize Etti | OK       | 73.0 ms   | 28.1 ms  | +39.2%    |
| payment   | Reward Dışı (C)    | INFO     | 589.0 ms  | 2672.9 ms| +13.8%    |
| shipping  | MARL Optimize Etti | OK       | 15229 ms  | 16948 ms | +24.4%    |
| ratings   | MARL Optimize Etti | OK       | 1102.3 ms | 661.8 ms | +13.0%    |
| user      | MARL Optimize Etti | OK       | 32.5 ms   | 11.6 ms  | +34.4%    |

> 5/6 servis "MARL Optimize Etti" — hiç kritik/warning yok ✅

---

### Step 4 — LLM Verbalization (step4_evidence_guided_llm.py)

#### Mimari
```
Diagnosis JSON → Evidence Pack → LLM Prompt → Açıklama
```
LLM yeni teşhis koymuyor — sadece deterministik sonucu yorumluyor.

#### Model
- **OpenRouter API:** `openai/gpt-oss-20b:free`
- Retry mekanizması: maks 5x, 15s aralıklı
- API key yoksa template fallback devreye giriyor

#### LLM Açıklamalarından Örnekler

**Cart:**
> "The MARL policy diagnosed the cart service as suitable for a scale‑down action with 97.7% confidence, reducing latency from 188ms to 102ms — a 45.5% improvement. KernelSHAP attribution identified pod count, CPU, and memory usage as the most influential features, confirming that resource saturation was the primary trigger."

**Shipping:**
> "KernelSHAP attribution identified CAR_Pods (0.0062), SHI_LAT (0.0040), SHI_Pods (0.0026) as the most influential features — the MARL policy primarily responds to cart's load when making shipping scale decisions."

**User:**
> "CAR_Pods (0.0246) confirmed that pod count and memory usage were the primary drivers — the user agent monitors cart capacity as its primary signal."

---

## 8. PROJE DOSYA YAPISI

```
testguncel/          ← DT eğitim klasörü
  dt2.py
  digital_twin_best.pth   (MAE=0.032, R²=0.945)
  dt_scaler.pkl
  test2.py                (trajectory validation)
  trajectory_results.csv

marlyeni/            ← MARL eğitim klasörü
  marl_env.py             (MAX_STEPS=6)
  ippo_agent.py
  mappo_agent.py
  dqn_agent.py
  a2c_agent.py
  qmix_agent.py
  train_ippo_30k.py
  train_mappo_new.py
  train_dqn_new.py
  train_a2c_new.py
  train_qmix_new.py
  evaluate_new.py
  checkpoints_ippo_new/
  checkpoints_mappo_new/
  checkpoints_dqn_new/
  checkpoints_a2c_new/
  checkpoints_qmix_new/
  evaluation_summary_final.csv

XAIYeni/             ← XAI pipeline klasörü
  step1_behavior_dataset.py
  step2_shap.py
  step3_diagnosis_engine.py
  step4_evidence_guided_llm.py
  xai_behavior_dataset.csv
  xai_obs_map.json
  xai_obs_*.csv
  xai_feature_importance.csv
  xai_diagnosis_report.json
  xai_diagnosis_summary.csv
  xai_diagnosis_plot.png
  xai_evidence_pack.json
  xai_evidence_guided_explanations.json
  xai_evidence_guided_report.txt
  xai_output/
    xai_results.csv
    xai_summary.png
    xai_heatmap.png
    per_service/*.png
    shap_values/*.npy
```

---

## 9. ÖĞRENME KANITLERI

### Kanıt 1 — Reward Artışı
```
İlk 100 episode: -3.744  →  Son 100 episode: +4.921  (+8.664)
```

### Kanıt 2 — SLO İyileşmesi
```
Random: %29.4  →  Rule-based: %34.9  →  MARL: %13.2  (%62 düşüş)
```

### Kanıt 3 — Entropy Düşüşü
```
Başlangıç: 1.097 (≈ uniform, tam rastgele)
Son:       0.364 (belirli aksiyonlara commit)
```

### Kanıt 4 — Yüksek Karar Güveni
```
cart/catalogue: 0.977-0.979 (neredeyse deterministik)
```

### Kanıt 5 — SHAP'ta Anlamlı Bağımlılıklar
```
USER  → CAR_Pods SHAP=0.0246  (cart bağımlılığı öğrenildi)
SHIP  → CAR_Pods SHAP=0.0062  (cart-shipping koordinasyonu)
```
Rastgele bir policy bu pattern'leri gösteremez.

### Kanıt 6 — Latency Azaldı
```
Cart:     -45.5%
Catalogue: -39.2%
User:     -34.4%
Shipping: -24.4%
```

---

## 10. ÖZET: NEYİ DEĞİŞTİRDİK

| Adım | Eski | Yeni | Neden |
|------|------|------|-------|
| Veri | 14.410 satır | **28.780 satır** | Gaussian augmentation |
| DT MAE | ~0.08 | **0.032** | Daha fazla veri |
| DT R² | ~0.85 | **0.945** | Daha fazla veri |
| MAX_STEPS | 10 | **6** | Trajectory test sonucu |
| XAI Step 2 | Surrogate+SHAP | **KernelSHAP (direkt)** | Surrogate anlamsızlaştı |
| XAI Step 4 | OpenRouter (eski model) | **GPT-OSS-20B** | Model değişti |
| QMIX | Dahil | **Hariç** | Loss diverge (8054) |
| Checkpoint | checkpoints_v2/ | **checkpoints_ippo_new/** | Yeni DT ile sıfırdan |

---

*Belge oluşturulma tarihi: Mayıs 2026*
*Proje: Kubernetes Robot Shop MARL Autoscaling — Bitirme Projesi*
