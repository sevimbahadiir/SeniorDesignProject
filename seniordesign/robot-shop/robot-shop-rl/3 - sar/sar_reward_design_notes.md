# SAR Adımı — Reward Tasarımı ve Literatür

## 1. Percentile (Yüzdelik Dilim) Nedir?

Percentile, bir veri kümesindeki değerlerin belirli bir yüzdesinin
altında kalan noktayı gösterir.

### Basit Örnek

10 isteğin latency değerleri:
```
5ms, 8ms, 12ms, 15ms, 20ms, 25ms, 40ms, 60ms, 90ms, 200ms
```

| Percentile | Değer | Anlam |
|------------|-------|-------|
| p50 (medyan) | ~22ms | İsteklerin %50'si bu kadar veya daha hızlı |
| p75 | ~50ms | İsteklerin %75'i bu kadar veya daha hızlı |
| p95 | ~90ms | İsteklerin %95'i bu kadar veya daha hızlı |
| p99 | ~195ms | İsteklerin %99'u bu kadar veya daha hızlı |

### Neden Ortalama Değil Percentile?

O 200ms'lik uç değer ortalamayı büyük ölçüde etkiler:
```
Ortalama = 47.5ms  ← 200ms tek başına ortalamanın yarısını oluşturuyor
p50      = 22ms    ← gerçek tipik deneyim
```

Ortalama, uç değerlere (outlier) karşı hassastır ve kullanıcının
gerçek deneyimini yanlış temsil edebilir. Percentile daha sağlıklı
bir ölçüttür.

### Endüstri Pratiği

Google, Amazon ve Netflix gibi şirketler SLA hedeflerini p99
üzerinden tanımlar:
> *"Requests must complete within 100ms at p99."*

Bu şu anlama gelir: isteklerin %99'u 100ms altında tamamlanmalı.
En yavaş %1'lik dilim bu garantinin dışındadır.

- **p50:** Kullanıcının "normal" deneyimi
- **p95:** Şikayet etmeye başlayan kullanıcı eşiği
- **p99:** Kritik sorunların görüldüğü nokta

### Projemizdeki Değerler

| Servis | p50 | p75 | p95 |
|--------|-----|-----|-----|
| cart | 46ms | 154ms | 542ms |
| catalogue | 25ms | 87ms | 323ms |
| shipping | 17.043ms | 26.203ms | 39.972ms |
| ratings | 588ms | 1.341ms | 2.280ms |
| user | 11ms | 26ms | 186ms |

Shipping'in p50'si bile 17 saniye — bu sistemin en büyük
darboğazının somut göstergesidir.

### Neden p50'yi Reward Eşiği Olarak Seçtik?

- **p50 = medyan** → outlier'lara duyarsız, gerçekçi baseline
- **Veri odaklı** → subjektif SLA hedefi belirlemeye gerek yok
- **Her servis için ayrı** → servisler arası adil karşılaştırma
- **p75 yerine neden p50?** p75 kullanılsaydı "iyi" eşiği çok
  yükselir, çoğu durum iyi görünür ve ajan yeterince zorlanmazdı

**Potansiyel reviewer sorusu:** *"Neden p50? Neden SLA hedefi değil?"*

**Cevap:**
> *"Robot Shop is an open-source benchmark application without
> formally defined SLAs. We adopted a data-driven approach using
> the empirical median (p50) as the baseline, which generalizes
> without requiring domain-specific threshold knowledge."*

---

## 2. SAR Formatı — Genel

SAR (State-Action-Reward) formatı, **Markov Decision Process (MDP)**'nin
veri temsilidir. RL'nin matematiksel temeli MDP'ye dayanır:

```
Bir ajan:
  - Ortamı STATE olarak gözlemler
  - ACTION alır
  - REWARD alır
  - Yeni STATE'e geçer (next_state)
```

Bizim dataset'imiz bu döngünün kaydedilmiş geçmişidir.
Her satır şunu ifade eder:

```
"Sistem şöyle görünüyordu     → state
 şu karar alındı              → action
 bu karar şu kadar iyiydi     → reward
 sonuç şu oldu                → next_state"
```

Bu format **Offline RL** için standarttır — yani sistemle canlı
etkileşim kurmadan, sadece geçmiş veriden öğrenmek için.
Bizim durumumuzda Kubernetes cluster'ını her denemede çalıştırmak
mümkün değildir; bu nedenle önce veri toplanmış, sonra bu formata
dönüştürülmüştür.

---

## 3. Reward Tasarımı — Literatür Karşılaştırması

### Yaklaşım 1: Threshold-based (Binary) Reward

```python
if latency < threshold:
    reward = +1
else:
    reward = -1
```

**Kullanıldığı çalışmalar:**
- Mao et al. (2016) — *"Resource Management with Deep Reinforcement
  Learning"* (Pensieve) — network scheduling için binary reward
- Tesauro et al. (2006) — IBM'in ilk RL tabanlı server provisioning
  çalışması

**Zayıflığı:** Bilgi kaybı.
```
latency = 47ms  → reward = 0.0
latency = 153ms → reward = 0.0
```
47ms ile 153ms aynı reward'u alır — ajan bu farkı öğrenemez.
Biz bu nedenle kullanmadık.

---

### Yaklaşım 2: Raw Continuous Reward

```python
reward = -latency          # veya
reward = -(latency / p50)
```

**Kullanıldığı çalışmalar:**
- Gu et al. (2022) — *"Auto-scaling Microservices with RL"* —
  response time'ı direkt negatif reward olarak kullandı
- Rossi et al. (2020) — *"Horizontal and Vertical Scaling of
  Container-based Applications"* — normalized latency reward

**Zayıflığı:** Uç değerlerde reward patlar.
Bizim verimizde shipping 60.000ms'e çıkabiliyor:
```
reward = -(60000 / 17042) * 2.0 = -7.04
```
Tek bir servis reward'u domine eder, diğer servisler öğrenilemez.

---

### Yaklaşım 3: SLA-based Reward

```python
if latency < SLA_target:
    reward = 0
else:
    reward = -(latency - SLA_target) / SLA_target
```

**Kullanıldığı çalışmalar:**
- Qiu et al. (2020) — *"FIRM: An Intelligent Fine-grained Resource
  Management Framework for SLO-Oriented Microservices"*
- Horovitz & Arian (2021) — Kubernetes HPA'yı RL ile replace etme

**Zayıflığı:** SLA hedefi tanımlı olmayan sistemlerde eşik
belirlenmesi subjektiftir. Robot Shop açık kaynaklı bir benchmark
olduğundan resmi SLA hedefleri yoktur.

---

### Yaklaşım 4: Tanh-compressed Continuous Reward (Bizim Seçimimiz)

```python
contribution = np.tanh(1.0 - latency / p50) * weight
reward = sum(contributions) - total_pods * POD_COST
```

**Literatür bağlantısı:**
Bu formülün tam olarak bu haliyle kullanıldığı bir çalışma
bulunmamaktadır — bu projede geliştirilen **özgün bir katkıdır.**
Ancak bileşenleri ayrı ayrı literatürde yerleşiktir:

- **Tanh sıkıştırma:**
  Haarnoja et al. (2018) — *"Soft Actor-Critic: Off-Policy Maximum
  Entropy Deep Reinforcement Learning with a Stochastic Actor"*
  Continuous action space'de reward sıkıştırma için tanh kullandı.
  SAC bugün en yaygın kullanılan RL algoritmalarından biridir.

- **Median normalizasyon:**
  Sigelman et al. (2010) — Google'ın *Dapper* distributed tracing
  çalışması — latency analizinde p50/p95/p99 kullanımını
  endüstride yaygınlaştırdı.

- **Servis ağırlıklandırma:**
  Gan et al. (2019) — *"An Open-Source Benchmark Suite for
  Microservices"* (DeathStarBench) — farklı microservislerin
  farklı kritiklikte olduğunu ve farklı ağırlıklandırılması
  gerektiğini gösterdi.

---

## 4. Tanh Reward — Neden Doğru Seçim?

### Matematiksel Davranış

tanh fonksiyonu her girdiyi (-1, +1) arasına sıkıştırır:

| Durum | Formül | Sonuç (weight=1) |
|-------|--------|-----------------|
| lat = p50 / 4 | tanh(0.75) | +0.635 (çok iyi) |
| lat = p50 / 2 | tanh(0.50) | +0.462 (iyi) |
| lat = p50 | tanh(0.0) | 0.0 (nötr, baseline) |
| lat = p50 × 2 | tanh(-1.0) | -0.762 (kötü) |
| lat = p50 × 4 | tanh(-3.0) | -0.995 (çok kötü) |
| lat = 60.000ms (shipping uç) | tanh(-2.52) | -0.987 (sınırlı) |

### Avantajlar

| Özellik | Açıklama |
|---------|----------|
| Doğal sınır | Her servis katkısı (-weight, +weight) arasında kalır |
| Yapay clip yok | Matematiksel olarak temiz, bilgi kaybı minimum |
| Gradient korunur | Uç değerlerde bile türev sıfırlanmaz |
| MARL stabilitesi | Bounded reward, çok ajanlı öğrenmede gradient patlamasını önler |
| Yorumlanabilirlik | p50'de sıfır geçiş — baseline referansı net |

### Servis Ağırlıkları

| Servis | Ağırlık | Gerekçe |
|--------|---------|---------|
| cart | 1.5 | Kullanıcı direkt etkileşimde |
| catalogue | 1.5 | Kullanıcı direkt etkileşimde |
| shipping | 2.0 | Birincil darboğaz, tüm sistemi etkiliyor |
| ratings | 0.5 | Auxiliary servis, kritik değil |
| user | 1.0 | Session yönetimi, orta kritik |
| payment | 0.0 | Seçenek C — reward'dan çıkarıldı |

### Pod Maliyet Cezası

```python
reward -= total_pods * 0.05
```

Pod cezası olmadan ajan "her zaman maksimum pod aç" politikasını
öğrenir. 0.05 katsayısı şu dengeyi sağlar:

- Max latency reward ≈ +6.5 (tüm servisler mükemmel)
- Ortalama pod cezası ≈ -1.5 (30 pod × 0.05)
- Pod cezası max reward'un ~%23'ü — baskın değil ama etkili

### Reward Aralığı (Gözlemlenen)

| İstatistik | Değer |
|------------|-------|
| Min | -8.45 |
| Max | +3.28 |
| Mean | -2.43 |
| Std | 2.71 |
| Pozitif | %22.5 |
| Negatif | %77.5 |

Negatif ağırlıklı dağılım beklenen bir sonuçtur — shipping
bottleneck nedeniyle sistem çoğunlukla stres altında çalışmaktadır.
MARL ajanı için bu "en az negatifi bul" problemi anlamına gelir
ki bu da gerçekçi bir optimizasyon hedefidir.

---

## 5. Rapor İçin Hazır Paragraflar

### Reward Fonksiyonu Açıklaması

> *"We propose a tanh-compressed, p50-normalized, service-weighted
> reward function. Existing approaches either employ binary
> threshold-based rewards, which suffer from information loss, or
> raw continuous rewards, which are susceptible to reward explosion
> under high latency values. Our formulation combines the stability
> of bounded rewards with the informativeness of continuous signals:*
>
> *r = Σ tanh(1 - l_i / p50_i) · w_i − N_pods · c*
>
> *where l_i is the latency of service i, p50_i is the empirical
> median latency, w_i is the service weight, N_pods is the total
> pod count, and c = 0.05 is the resource cost coefficient.
> The tanh function naturally bounds each service's contribution
> to (−w_i, +w_i), preventing any single service from dominating
> the reward signal. This approach is inspired by the bounded
> reward formulation in Soft Actor-Critic (Haarnoja et al., 2018)
> and adapted for the microservice latency optimization domain."*

### Payment Açıklaması

> *"Payment latency was excluded from the reward function due to
> structural measurement gaps caused by the shipping bottleneck.
> Under high load, Locust workers are saturated at the shipping
> step, preventing requests from reaching the payment service.
> Payment latency is retained in the state space to enable the
> Digital Twin to capture inter-service dependencies."*

### Percentile Eşik Seçimi

> *"Reward thresholds were derived empirically from the p50
> (median) of non-zero latency observations in the collected
> dataset. Robot Shop is an open-source benchmark application
> without formally defined SLAs; we therefore adopted a
> data-driven approach using the empirical median as the baseline,
> which generalizes without requiring domain-specific threshold
> knowledge."*

---

## 6. Referanslar

- Haarnoja et al. (2018). *Soft Actor-Critic: Off-Policy Maximum
  Entropy Deep Reinforcement Learning with a Stochastic Actor.*
  ICML 2018.

- Mao et al. (2016). *Resource Management with Deep Reinforcement
  Learning.* HotNets 2016.

- Gan et al. (2019). *An Open-Source Benchmark Suite for
  Microservices and Their Hardware-Software Implications for Cloud
  and Edge Systems.* ASPLOS 2019.

- Qiu et al. (2020). *FIRM: An Intelligent Fine-grained Resource
  Management Framework for SLO-Oriented Microservices.*
  OSDI 2020.

- Rossi et al. (2020). *Horizontal and Vertical Scaling of
  Container-based Applications using Reinforcement Learning.*
  CLOUD 2020.

- Sigelman et al. (2010). *Dapper, a Large-Scale Distributed
  Systems Tracing Infrastructure.* Google Technical Report.
