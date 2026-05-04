# Veri Ön İşleme — Adım 1: merge_and_clean.py

## Genel Amaç

Ham Prometheus/Istio metriklerinden oluşan 10 ayrı CSV dosyasını birleştirerek
temizlenmiş ve düzenlenmiş tek bir dataset dosyası (`rl_dataset_clean.csv`) üretir.
Bu dosya SAR formatı oluşturma adımının (sar_create.py) girdisidir.

---

## Veri Seti Genel Bilgileri

| Özellik | Değer |
|---------|-------|
| Run sayısı | 10 |
| Kullanıcı yük aralığı | 40 – 500 eşzamanlı kullanıcı |
| Her run süresi | 4 saat |
| Ham toplam satır | 14.410 |
| Temizleme sonrası satır | 14.400 |
| Toplam sütun | 32 |
| Zaman aralığı | 2026-04-27 → 2026-04-30 |

---

## Adımlar

### STEP 1 — CSV Dosyalarını Yükle ve Birleştir

10 farklı yük senaryosuna ait CSV dosyaları okunur. Her dosyaya `user_load` sütunu
eklenir — bu sütun sonraki adımlarda run sınırlarını (hangi satırın hangi run'a ait
olduğunu) tespit etmek için kullanılır. Tüm dosyalar tarihe göre sıralanarak
tek bir DataFrame'de birleştirilir.

**Neden user_load ekleniyor?**
SAR adımında action hesaplanırken ve next_state oluşturulurken run geçişlerinde
hatalı transition üretilmemesi için `groupby('user_load')` kullanılır. Bu sütun
olmadan 40user run'ının son satırı ile 80user run'ının ilk satırı arasında
anlamlı olmayan bir diff hesaplanır.

| Dosya | Kullanıcı | Satır |
|-------|-----------|-------|
| dataset_run1_40user.csv | 40 | 1441 |
| dataset_run2_80user.csv | 80 | 1441 |
| dataset_run3_120user.csv | 120 | 1441 |
| dataset_run4_150user.csv | 150 | 1441 |
| dataset_run5_200user.csv | 200 | 1441 |
| dataset_run6_250user.csv | 250 | 1441 |
| dataset_run7_300user.csv | 300 | 1440 |
| dataset_run8_350user.csv | 350 | 1438 |
| dataset_run9_400user.csv | 400 | 1439 |
| dataset_run10_500user.csv | 500 | 1437 |

---

### STEP 2 — dispatch ve web Sütunlarını Çıkar

`dispatch_*` ve `web_*` önekli 10 sütun dataset'ten çıkarılır.

**Neden çıkarılıyor?**
- **dispatch:** RabbitMQ consumer servisi. HTTP trafiği taşımaz, Istio metrikleri
  üretmez. Projede bu servis için ajan tanımlanmamıştır.
- **web:** Nginx reverse proxy. Gelen trafiği diğer servislere yönlendirir,
  kendi başına ölçeklendirilmez. Bu servis için de ajan yoktur.

Bu sütunlar modele dahil edilseydi anlamsız gürültü olarak etki ederdi.

42 sütun → 32 sütun.

---

### STEP 3 — Geçersiz Satırları Filtrele

**Kural:** `cart_latency`, `catalogue_latency` ve `shipping_latency` sütunlarının
**üçü birden** sıfır olan satırlar silinir.

**Neden bu üçü?**
Locust'un simüle ettiği kullanıcı akışı şu sıradadır:
```
user → catalogue → cart → shipping → payment
```
Bu akış nedeniyle normal çalışma koşullarında cart, catalogue ve shipping
aynı anda sıfır latency raporlayamaz. Üçü birden sıfırsa iki durum söz konusudur:
1. Locust henüz başlamamıştır (run'ın ilk birkaç saniyesi)
2. Prometheus o pencerede veri yakalayamamıştır

Her iki durumda da bu satır gerçek bir sistem durumunu temsil etmemektedir.

**Neden payment bu filtreye dahil değil?**  
→ Aşağıda "Payment Sıfır Sorunu" başlığında açıklanmaktadır.

**Sonuç:** 14.410 satırdan 10 satır silindi. Sistemin neredeyse hiç
kesintisiz çalıştığını göstermektedir.

---

### STEP 4 — Pod Değerlerini Sınırla

`random_scaler.py`'de tanımlı servis sınırları:

| Servis | Min Pod | Max Pod |
|--------|---------|---------|
| cart | 1 | 8 |
| catalogue | 1 | 8 |
| payment | 2 | 8 |
| shipping | 2 | 8 |
| ratings | 1 | 8 |
| user | 1 | 8 |

**Neden gerekli?**
Prometheus bazı zaman dilimlerinde bu sınırların dışında pod sayıları
ölçmüştür (örn. ratings=0, cart=9, payment=10). Bunlar ölçüm artefaktlarıdır —
gerçek sistem durumunu yansıtmazlar. `clip()` ile sınırlar uygulanır.

**Etkilenen satırlar:** 84 satır (%0.6) — çok küçük bir düzeltme.

| Servis | Önce | Sonra | Değişen |
|--------|------|-------|---------|
| cart | 3-9 | 3-8 | 6 |
| catalogue | 3-9 | 3-8 | 3 |
| payment | 2-10 | 2-8 | 34 |
| shipping | 1-10 | 2-8 | 11 |
| ratings | 0-10 | 1-8 | 29 |
| user | 3-10 | 3-8 | 1 |

---

### STEP 5 — Sütun Sırasını Düzenle

Sütunlar okunabilir ve tutarlı bir sıraya getirilir:

```
date | user_load | cart_num_pods | cart_cpu_usage | cart_mem_usage |
cart_request_rate | cart_latency | catalogue_num_pods | ... | user_latency
```

6 servis × 5 metrik = 30 metrik sütunu + 2 meta sütun = 32 toplam sütun.

---

### STEP 6 — Kaydet

Çıktı: `rl_dataset_clean.csv`

---

## Özet İstatistikler

| Servis | Pod (min-max) | Max RPS | Zero-lat% | Ort. Latency (ms) |
|--------|---------------|---------|-----------|-------------------|
| cart | 3-8 | 108.46 | %0.1 | 143.3 |
| catalogue | 3-8 | 178.41 | %0.1 | 82.1 |
| payment | 2-8 | 2.60 | %58.9 ⚠ | 3704.3 |
| shipping | 2-8 | 20.36 | %1.8 | 18943.8 |
| ratings | 1-8 | 63.36 | %0.2 | 880.0 |
| user | 3-8 | 33.64 | %0.2 | 38.0 |

---

## Shipping Yavaşlama Nedeni

Shipping servisi her checkout işlemi için MySQL veritabanına **3 ayrı sorgu** atar:

1. `/api/shipping/codes` → posta kodları listesi
2. `/api/shipping/cities/{code}` → şehir listesi  
3. `/api/shipping/calc/{uuid}` → fiyat hesaplama

Her sorgu disk I/O gerektirmektedir. Yük arttıkça MySQL bağlantı havuzu
tıkanmakta ve sorgular kuyrukta beklemeye başlamaktadır.

**Kritik nokta:** Pod sayısını artırmak bu sorunu tam olarak çözmez.
Darboğaz shipping pod'u değil, tüm pod'ların paylaştığı **tek MySQL instance'ıdır.**
Daha fazla shipping pod'u, daha fazla eş zamanlı MySQL bağlantısı demektir —
bu da bağlantı havuzunu daha hızlı tüketir.

Bu nedenle 40 kullanıcıda bile ortalama **~10 saniye**, 200+ kullanıcıda
**~20 saniye** latency gözlemlenmektedir.

**Rapor için not:** Bu durum "inherent architectural bottleneck" olarak
sunulabilir. RL ajanının bu kısıtı öğrenmesi ve buna göre shipping pod
kararı vermesi projenin ilginç bulgularından biri olacaktır.

---

## Payment Sıfır Sorunu ve Seçenek C Kararı

### Neden payment sıfır geliyor?

Locust kullanıcı akışı sıralıdır:
```
user → catalogue → cart → shipping (20sn) → payment
```

Shipping'de 20 saniye beklenmektedir. Locust worker sayısı sınırlı olduğundan
yük arttıkça **tüm worker'lar shipping adımında mahsur kalır** ve payment'a
sıra gelmez. Istio, trafik almayan bir servisten metrik üretemez;
Prometheus bu durumu 0 olarak kaydeder.

| Run | Kullanıcı | Payment Zero% |
|-----|-----------|---------------|
| run1 | 40 | %7 |
| run2 | 80 | %17 |
| run3 | 120 | %46 |
| run4 | 150 | %54 |
| run5 | 200 | %69 |
| run6 | 250 | %66 |
| run7 | 300 | %75 |
| run8 | 350 | %79 |
| run9 | 400 | %86 |
| run10 | 500 | %91 |

### Seçenek C Kararı

Payment latency **state uzayına dahil edilir** (sıfır değerler de dahil),
ancak **reward hesabından çıkarılır.**

**Gerekçe:**
- Payment'ı tamamen çıkarmak: checkout funnel'ının son adımını hiç modellememiş olmak demektir.
- Payment=0 satırları filtrelemek: %59 veri kaybı ve yük dağılımı bozulması demektir.
- Seçenek C: DT modeli "yüksek yükte payment'a trafik ulaşamıyor" ilişkisini öğrenir.
  Bu, servisler arası bağımlılığın doğal bir öğrenme sinyalidir.

**Rapor için not:**
> "Payment latency was included in the state space to allow the Digital Twin
> to learn inter-service dependencies. However, due to the structural measurement
> gap caused by the shipping bottleneck, payment was excluded from the reward
> function to prevent signal corruption."

---

## Sonraki Adım

`rl_dataset_clean.csv` → `sar_create.py` (State-Action-Reward-NextState formatı)
