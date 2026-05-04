# Veri Ön İşleme — Adım 2: sar_create.py

## Genel Amaç

Temizlenmiş veriyi (`rl_dataset_clean.csv`) Digital Twin ve MARL modellerinin
kullanabileceği SAR formatına dönüştürür.

Her satır bir **transition** (geçiş) temsil eder:
```
(state, action, reward, next_state)
```

Çıktı dosyaları:
- `rl_dataset_sar_raw.csv` — normalize edilmemiş, analiz ve debug için
- `rl_dataset_sar_normalized.csv` — normalize edilmiş, DT modeline girdi
- `scaler.pkl` — normalizasyon için kullanılan MinMaxScaler nesnesi

---

## Sütun Yapısı

```
date | user_load | [30 state] | [6 action] | reward | [30 next_state]
```

| Grup | Sütun Sayısı | İçerik |
|------|-------------|--------|
| Meta | 2 | date, user_load |
| State | 30 | 6 servis × 5 metrik |
| Action | 6 | her servis için -1/0/+1 |
| Reward | 1 | skaler değer |
| Next State | 30 | bir sonraki adımın state'i |
| **Toplam** | **69** | |

---

## STEP 1 — Veriyi Yükle

`rl_dataset_clean.csv` okunur. State sütunları tanımlanır:

```
[servis]_num_pods, [servis]_cpu_usage, [servis]_mem_usage,
[servis]_request_rate, [servis]_latency
```

6 servis × 5 metrik = **30 boyutlu state uzayı.**

---

## STEP 2 — Action Hesapla

Her servis için pod sayısının bir önceki adıma göre değişimini hesaplar.

```python
diff = df.groupby('user_load')[f'{svc}_num_pods'].diff()
action = sign(diff)  # -1, 0, veya +1
```

**groupby('user_load') neden kritik?**
Run geçişlerinde (örn. 40user → 80user) pod sayısı sıfırlanır.
`groupby` olmadan bu geçişte hatalı action hesaplanır.
Her run'ın ilk satırında action = 0 (NaN → 0) olarak atanır.

**Action değerleri:**
| Değer | Anlam |
|-------|-------|
| -1 | Scale down (pod azaldı) |
| 0 | No change |
| +1 | Scale up (pod arttı) |

Farkın büyüklüğü önemli değildir — sadece yön kullanılır.
Random scaler zaten en fazla birkaç pod değiştirdiğinden
bu ayrım yeterli bilgiyi taşımaktadır.

**Gözlemlenen dağılım:**
| Servis | Scale Down | No Change | Scale Up |
|--------|-----------|-----------|----------|
| cart | ~%6 | ~%88 | ~%6 |
| catalogue | ~%6 | ~%88 | ~%6 |
| shipping | ~%5 | ~%91 | ~%4 |
| ratings | ~%5 | ~%90 | ~%5 |
| payment | ~%6 | ~%89 | ~%5 |
| user | ~%6 | ~%89 | ~%5 |

No change ~%88-91 oranında baskın — bu beklenen bir durumdur.
Random scaler 90 saniyede bir karar verir ve çoğu zaman pod sayısını
korur. RL için hem değişim hem sabit örneklerin yeterli sayıda
bulunması önemlidir.

---

## STEP 3 — Reward Hesapla

### Tasarım Kararı: Tanh-Compressed Continuous Reward

**Neden threshold değil continuous?**

Threshold tabanlı reward'da bilgi kaybı yaşanır:
```
cart_lat = 47ms  → reward = 0.0
cart_lat = 153ms → reward = 0.0
```
47ms ile 153ms aynı reward'u alır — ajan bu farkı öğrenemez.

**Neden tanh?**

Saf continuous reward (`-(lat/p50) * weight`) uç latency değerlerinde
çok büyük negatif değerler üretir. Shipping 60.000ms'e çıkabildiğinden
reward -100'ün altına düşüyor ve reward landscape'i tamamen shipping
domine ediyordu.

Tanh bu sorunu doğal olarak çözer:

```python
contribution = tanh(1 - latency / p50) * weight
```

| Durum | Formül | Sonuç |
|-------|--------|-------|
| lat = p50/4 | tanh(0.75) × w | +0.635w (çok iyi) |
| lat = p50/2 | tanh(0.50) × w | +0.462w (iyi) |
| lat = p50   | tanh(0.0)  × w | 0.0 (nötr) |
| lat = p50×2 | tanh(-1.0) × w | -0.762w (kötü) |
| lat = p50×4 | tanh(-3.0) × w | -0.995w (çok kötü, sınırlı) |

**Avantajlar:**
- Her servisin katkısı (-weight, +weight) arasında doğal olarak sınırlı
- Yapay clip gerekmez — matematiksel olarak temiz
- Uç latency değerlerinde gradient patlaması olmaz
- MARL öğrenmesini stabilize eder
- Türevlenebilir — gradient-based optimizasyon ile uyumlu

**Akademik gerekçe:**
Tanh aktivasyon RL reward fonksiyonlarında yaygın kullanılır.
Bounded reward, özellikle çok ajanlı sistemlerde gradient patlamasını
önler ve politika öğrenmesini stabilize eder.

**Rapor için:**
> *"A tanh-compressed continuous reward was adopted to prevent reward
> dominance by the shipping service, which exhibits latency values
> up to 3.5× its median. The tanh function naturally bounds each
> service's contribution to (-weight, +weight), eliminating the need
> for artificial clipping while preserving gradient information
> across the full latency range."*

### Servis Ağırlıkları

Ağırlıklar servisin checkout funnel'ındaki kritikliğine ve
sistem genelindeki etkisine göre belirlenmiştir:

| Servis | Ağırlık | Gerekçe |
|--------|---------|---------|
| cart | 1.5 | Kullanıcı direkt etkileşimde |
| catalogue | 1.5 | Kullanıcı direkt etkileşimde |
| shipping | 2.0 | Darboğaz servis, tüm sistemi etkiliyor |
| ratings | 0.5 | Auxiliary servis, kritik değil |
| user | 1.0 | Session yönetimi, orta kritik |
| payment | 0.0 | Seçenek C — reward'dan çıkarıldı |

**Rapor için:**
> *"Service weights were assigned based on their criticality in the
> checkout funnel and their observed impact on end-to-end latency.
> The shipping service received the highest weight (2.0) due to its
> role as the primary system bottleneck."*

### Latency Eşikleri

Gerçek verinin istatistiklerinden türetilmiştir (sıfır olmayan satırlar):

| Servis | p50 (ms) | p75 (ms) |
|--------|----------|----------|
| cart | 46.2 | 154.2 |
| catalogue | 25.4 | 86.5 |
| shipping | 17042.6 | 26202.7 |
| ratings | 587.9 | 1341.0 |
| user | 10.9 | 25.6 |

### Pod Maliyet Cezası

```python
reward -= total_pods * POD_COST  # POD_COST = 0.05
```

**Neden gerekli?**
Pod cezası olmadan ajan "her zaman maksimum pod aç" politikasını
öğrenir — latency düşer ama kaynak verimliliği sıfıra iner.

**Neden 0.05?**
- Max latency reward ≈ 6.5 (tüm servisler iyi latency)
- Ortalama toplam pod ≈ 30 (6 servis × 5 pod)
- Pod cezası = 30 × 0.05 = 1.5

Pod cezasının max reward'a oranı ~%23 — ajan hem latency'yi
hem verimliliği optimize etmek zorunda kalır, ancak pod cezası
baskın hale gelmez.

**0.1 yerine 0.05 seçilme gerekçesi:**
Önceki versiyonda reward ortalaması -1.83, negatif oran %74 çıkmıştı.
Continuous reward zaten daha negatif bir landscape üretir. Pod cezasını
düşürmek MARL'ın öğrenme sürecini stabilize eder.

### Payment Reward'dan Çıkarılma Gerekçesi

Payment latency tüm runlarda yapısal olarak eksiktir (%7-%91).
Eksikliğin nedeni payment servisinin arızası değil, shipping
bottleneck'inin Locust worker'larını tüketmesidir.

Payment reward'a dahil edilseydi:
- Yüksek yük senaryolarında payment her zaman 0 reward üretirdi
- Ajan "yüksek yükte payment ceza almaz" şeklinde yanlış öğrenirdi
- Reward sinyali bozulurdu

Payment **state'e dahil edilmiştir** — DT servisler arası
bağımlılığı öğrenebilsin diye.

**Rapor için:**
> *"Payment latency was excluded from the reward function due to
> structural measurement gaps caused by the shipping bottleneck.
> Under high load, Locust workers are saturated at the shipping
> step, preventing requests from reaching the payment service.
> Payment latency is retained in the state space to enable the
> Digital Twin to capture inter-service dependencies."*

---

## STEP 4 — Next State Oluştur

```python
df[f'next_{col}'] = df.groupby('user_load')[col].shift(-1)
```

Her satıra bir sonraki zaman adımının state'i eklenir.
Her run'ın son satırında next_state NaN olur — bu satırlar silinir.

**Silinen satır sayısı:** 10 run × 1 satır = 10 satır

**DT için önemi:**
Digital Twin şu ilişkiyi öğrenir:
```
f(state, action) → next_state
```
next_state olmadan bu eğitim yapılamaz.

---

## STEP 5 — Raw SAR Kaydet

Ham (normalize edilmemiş) SAR dosyası kaydedilir.
Gerçek değerleri görmek ve debug için kullanılır.

---

## STEP 6 — Normalize Et

### Neden normalizasyon gerekli?

Farklı metrikler çok farklı ölçeklerde:
```
shipping_mem_usage ≈ 6.800.000.000 (byte)
cart_latency       ≈ 50            (ms)
user_num_pods      ≈ 5             (adet)
```

Normalize edilmezse sinir ağı büyük sayılara aşırı ağırlık verir.

### MinMaxScaler

```python
normalized = (value - min) / (max - min)  → [0, 1]
```

### Data Leakage Önleme

Scaler **sadece train setinde** (%80) fit edilir:
```python
scaler.fit(train_rows)       # sadece train'den öğren
scaler.transform(all_rows)   # hepsine uygula
```

Val setinde fit edilseydi val'in min/max değerleri modele
sızmış olurdu — bu **data leakage** olur ve makale
hakem sürecinde eleştirilir.

### Out-of-Range Clip

Val setinde train'in min/max aralığını aşan değerler
0-1 dışına çıkabilir. Bu değerler `clip(0, 1)` ile sabitlenir.

```python
df_norm[normalize_cols] = df_norm[normalize_cols].clip(0, 1)
```

Önceki versiyonda 44 satır (%0.3) etkilenmişti.

**Rapor için:**
> *"A MinMaxScaler was fitted exclusively on the training split
> to prevent data leakage. Values exceeding the training range
> in the validation set were clipped to [0, 1]."*

**Normalize edilmeyen sütunlar:**
- Action: zaten -1/0/+1 ölçeğinde
- Reward: MARL aşamasında ayrıca ele alınacak

---

## Özet İstatistikler

| Metrik | Değer |
|--------|-------|
| Toplam satır (SAR) | ~14.390 |
| State boyutu | 30 |
| Action boyutu | 6 |
| Next state boyutu | 30 |
| Train / Val split | %80 / %20 |
| Normalize edilen sütun | 60 (state + next_state) |

---

## Sonraki Adım

`rl_dataset_sar_normalized.csv` → `digital_twin.py`
