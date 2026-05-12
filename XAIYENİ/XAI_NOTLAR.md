# XAI Pipeline — Notlar ve Kararlar

## Klasör Yapısı

```
xai_all_algos.py              → Tüm algoritmalar için XAI pipeline (tek script)
xai_results/
  ippo/                       → IPPO XAI çıktıları
  mappo/                      → MAPPO XAI çıktıları
  dqn/                        → DQN XAI çıktıları
  a2c/                        → A2C XAI çıktıları
  comparison_all.csv          → Tüm algoritmalar karşılaştırma tablosu
  comparison_top_features.png → Top feature karşılaştırma grafikleri
  comparison_confidence.png   → Karar güveni karşılaştırması
  comparison_improvement.png  → Latency iyileşme karşılaştırması
```

Her algoritma klasöründe:
```
xai_behavior_dataset.csv          → 7200 kayıt (200 ep × 6 adım × 6 servis)
xai_obs_map.json                  → Observation index haritası
xai_results.csv                   → SHAP feature importance tablosu
xai_summary.png                   → 6-panel SHAP özet grafiği
xai_diagnosis_report.json         → Servis bazında diagnosis
xai_diagnosis_summary.csv         → Diagnosis özet tablosu
xai_diagnosis_plot.png            → Diagnosis grafikleri
xai_evidence_guided_report.txt    → LLM açıklamaları (GPT-OSS-20B)
xai_evidence_guided_explanations.json → JSON formatında açıklamalar
per_service/*.png                 → Her servis için detaylı SHAP grafikleri
shap_values/*.npy                 → Ham SHAP değerleri (numpy)
```

---

## Pipeline Mimarisi (4 Adım)

```
Step 1: Behavior Dataset  →  Policy greedy modda çalıştırılır
Step 2: KernelSHAP        →  Direkt ActorNet üzerinde SHAP hesaplanır
Step 3: Diagnosis Engine  →  Rule-based kural hiyerarşisi uygulanır
Step 4: LLM Verbalization →  GPT-OSS-20B kanıtları yorumlar
```

**Neden KernelSHAP (surrogate değil)?**
Eski yöntemde GradientBoosting surrogate kullanılıyordu. Surrogate accuracy=1.0 çıktı — policy tek aksiyona converge etmişti, surrogate anlamsızlaştı. Yeni yöntemde KernelSHAP direkt actor network üzerinde çalışıyor — proxy yok, metodolojik olarak daha güçlü.

**LLM rolü nedir?**
LLM yeni teşhis koymaz. Sadece deterministik rule-based engine'in ürettiği kanıtları (diagnosis, latency, confidence, SHAP features) doğal dile çevirir. Hallucination riski minimize edildi.

---

## Step 1: Behavior Dataset

200 episode, greedy policy (argmax), 6 adım/episode → 7200 kayıt/algoritma.

**Aksiyon Dağılımı ve Karar Güveni:**

| Servis | IPPO | MAPPO | DQN | A2C |
|---|---|---|---|---|
| cart | scale_down %100 (0.977) | scale_down %100 (0.993) | karma (0.515) | scale_down %100 (0.909) |
| catalogue | scale_down %100 (0.979) | scale_down %100 (0.942) | karma (0.467) | scale_down %100 (0.925) |
| payment | scale_up %100 (0.901) | scale_up %100 (0.995) | karma (0.449) | scale_up %100 (0.913) |
| shipping | scale_up %100 (0.730) | scale_up %100 (0.961) | karma (0.461) | scale_up %100 (0.844) |
| ratings | scale_up %100 (0.851) | scale_up %100 (0.945) | karma (0.644) | scale_up %100 (0.785) |
| user | scale_up %100 (0.927) | scale_up %100 (0.985) | karma (0.424) | scale_up %100 (0.862) |

**Kritik Gözlem:** IPPO, MAPPO, A2C tamamen converge etmiş — her durumda tek bir aksiyon seçiyor. DQN ise karma kararlar veriyor, convergence yetersiz.

**MAPPO en yüksek güvene sahip** — global state bilgisiyle daha kararlı policy öğrendi (shipping: 0.961 vs IPPO: 0.730).

---

## Step 2: KernelSHAP

**Parametreler:** background=100, explain=200, nsamples=100

**Top 1 Feature (Algoritma × Servis):**

| Servis | IPPO | MAPPO | DQN | A2C |
|---|---|---|---|---|
| cart | SHI_Pods (0.000381) | CAT_Pods (0.000170) | SHI_LAT (0.110642) | CAT_Pods (0.000378) |
| catalogue | RAT_LAT (0.000353) | RAT_Pods (0.000973) | CAT_LAT (0.039553) | RAT_Pods (0.002459) |
| payment | PAY_Pods (0.002193) | PAY_Pods (0.000169) | PAY_RPS (0.082283) | PAY_Pods (0.003151) |
| shipping | CAR_Pods (0.006214) | SHI_Pods (0.000332) | PAY_RPS (0.039669) | PAY_Pods (0.002770) |
| ratings | RAT_Pods (0.005503) | CAT_Pods (0.001722) | RAT_LAT (0.105177) | RAT_Pods (0.006696) |
| user | CAR_Pods (0.024568) | USE_MEM (0.000318) | CAR_MEM (0.030029) | CAR_Pods (0.004287) |

**Anahtar Bulgular:**

**1. IPPO ve A2C — Pods dominant:**
Her iki algoritmada da Pods (pod sayısı) en önemli feature. Policy kapasite sinyalini latency'den önce öğrendi.

**2. DQN — Latency dominant, çok yüksek SHAP değerleri:**
DQN'de LAT ve RPS baskın, SHAP değerleri diğerlerinden 10-100x büyük (0.11 vs 0.0004). Convergence yetersiz, anlık latency'ye reaktif tepki veriyor.

**3. MAPPO — Çok düşük SHAP değerleri:**
MAPPO'da SHAP değerleri en küçük (0.000169-0.001722). Global state ile eğitilen critic daha smooth bir policy öğretti — hiçbir feature dominant değil, kararlar dağıtık. Bu yorumlanabilirliği zorlaştırıyor.

**4. Emergent Coordination (IPPO/A2C):**
- USER → CAR_Pods = 0.0246 (cart kapasitesini izliyor)
- SHIPPING → CAR_Pods = 0.006214 (cart yüküne göre scale ediyor)
- Bağımsız eğitilmelerine rağmen servisler arası koordinasyon öğrenildi.

**5. RPS (request rate) tüm başarılı algoritmalarda önemsiz:**
IPPO/MAPPO/A2C'de RPS en altta. DQN'de ise RPS önemli — bu DQN'in anlık trafik sinyaline takılı kaldığını gösteriyor.

---

## Step 3: Rule-based Diagnosis

**Kural Hiyerarşisi:**
1. Reward dışı servis → `reward_excluded`
2. MARL iyileştirdi (>%10) → `marl_optimized`
3. Shipping: scale_up + CPU<0.65 → `pod_insensitive_bottleneck`
4. Latency>1.2×p50 + keep/scale_down → `under_provisioned`
5. Diğer → `monitoring_required`

**Diagnosis Sonuçları:**

| Servis | IPPO | MAPPO | DQN | A2C |
|---|---|---|---|---|
| cart | ✅ MARL Optimized | ✅ MARL Optimized | ✅ MARL Optimized | ✅ MARL Optimized |
| catalogue | ✅ MARL Optimized | ✅ MARL Optimized | ✅ MARL Optimized | ✅ MARL Optimized |
| payment | ℹ️ Reward Excluded | ℹ️ Reward Excluded | ℹ️ Reward Excluded | ℹ️ Reward Excluded |
| shipping | ✅ MARL Optimized | ✅ MARL Optimized | 🔴 Pod-Insensitive | ✅ MARL Optimized |
| ratings | ✅ MARL Optimized | ✅ MARL Optimized | ⚠️ Under-Provisioned | ✅ MARL Optimized |
| user | ✅ MARL Optimized | ✅ MARL Optimized | ⚠️ Under-Provisioned | ✅ MARL Optimized |

**DQN neden farklı?**
DQN shipping için "Pod-Insensitive Bottleneck" aldı — shipping'de yalnızca %0.2 iyileşme (IPPO %24.4 vs). Ratings ve user için "Under-Provisioned" — DQN bu servislerde latency yüksekken scale_down seçiyor, yanlış yönde aksiyon. DQN convergence sorunu XAI ile açıkça görünür hale geldi.

---

## Step 4: LLM Verbalization

**Model:** `openai/gpt-oss-20b:free` via OpenRouter
**Yöntem:** Evidence-guided — LLM sadece kanıtları yorumlar, yeni teşhis koymaz.

**Öne Çıkan LLM Açıklamaları:**

**IPPO — Shipping:**
> "KernelSHAP analysis identified the number of CAR pods as the most influential feature (SHAP = 0.006214), followed by SHI latency, SHI pods, CAR latency, and PAY pods, suggesting that pod allocation for customer-related and shipping-related workloads drives the observed performance gains."

**DQN — Shipping (Critical):**
> "The XAI analysis indicates that the shipping service is experiencing a pod-insensitive bottleneck. PAY_RPS, SHI_LAT, and CAR_Pods are the most influential features, suggesting that high request rates, service-level interface latency are primary contributors."

**DQN — User (Warning):**
> "These findings suggest that the policy's confidence (0.424) is based on resource-intensive features that are not adequately addressed by a simple scale-down, highlighting a mismatch between the learned policy and the rule-based baseline."

---

## Algoritmalar Arası Karşılaştırma Özeti

### Karar Güveni
| Algoritma | Ort. Güven | Yorum |
|---|---|---|
| MAPPO | 0.970 | En yüksek — global state avantajı |
| IPPO | 0.894 | Yüksek |
| A2C | 0.873 | İyi |
| DQN | 0.493 | Çok düşük — convergence yok |

### Latency İyileşmesi vs Rule-based
| Servis | IPPO | MAPPO | DQN | A2C |
|---|---|---|---|---|
| cart | +45.5% | +45.5% | +21.0% | +45.5% |
| catalogue | +39.2% | +39.2% | +27.3% | +39.2% |
| payment | +13.8% | +13.8% | -4.3% | +13.8% |
| shipping | +24.4% | +24.4% | +0.2% | +24.4% |
| ratings | +13.0% | +13.0% | +4.5% | +13.0% |
| user | +34.4% | +34.4% | +0.5% | +34.4% |

### SHAP Feature Profili
| Algoritma | Dominant Feature Tipi | SHAP Büyüklüğü |
|---|---|---|
| IPPO | Pods (pod sayısı) | Orta (0.001-0.025) |
| MAPPO | Dağıtık | Çok küçük (0.0001-0.002) |
| DQN | LAT + RPS | Çok büyük (0.03-0.11) |
| A2C | Pods (pod sayısı) | Orta (0.001-0.007) |

---

## Tez İçin Anahtar Bulgular

**1. XAI, DQN'in başarısızlığını açıklıyor**
DQN evaluate'de kötü sonuç vermişti (%26.4 SLO). XAI bunu somutlaştırıyor: düşük güven, yanlış aksiyon (under-provisioned servislere scale_down), latency'ye reaktif ve anlık tepkisel policy.

**2. Emergent coordination kanıtlandı**
IPPO ve A2C'de USER → CAR_Pods, SHIPPING → CAR_Pods bağımlılıkları SHAP ile görünür hale geldi. Ajanlar bağımsız eğitilmesine rağmen koordinasyon öğrendi.

**3. MAPPO yorumlanabilirlik paradoksu**
MAPPO en yüksek performansı gösteriyor (güven: 0.970) ama en düşük SHAP değerlerine sahip. Global state bilgisi smooth policy üretiyor — XAI perspektifinden bakıldığında bu "black box" eğilimi artıyor.

**4. Pods > Latency > RPS**
Başarılı algoritmalarda önem sırası: Pods en önemli, Latency ikinci, RPS neredeyse önemsiz. Bu Kubernetes'te pod kapasitesinin birincil kontrol sinyali olduğunu doğruluyor.

---

## Sınırlılıklar

**1. SHAP hesaplama maliyeti**
KernelSHAP yavaş — background=100, explain=200 ile her servis ~2-5 dakika. Cache mekanizması eklendi, tekrar çalıştırmada atlanıyor.

**2. Behavior dataset dağılımı**
Test state'leri SAR normalized veriden rastgele seçildi — gerçek MARL eğitimindeki state dağılımından farklı olabilir.

**3. LLM deterministik değil**
Aynı prompt'a her çalıştırmada biraz farklı yanıt üretebilir. Template fallback mekanizması eklendi — API olmadığında deterministik çıktı garantisi var.

---

## Çalıştırma

```bash
# API key olmadan (template fallback)
python3 xai_all_algos.py

# API key ile (LLM açıklamaları)
export OPENROUTER_API_KEY="sk-or-..."
python3 xai_all_algos.py
```

SHAP cache varsa (`shap_values/*.npy`) Step 2 atlanır, hızlı geçer.
