# SAR Veri Hazırlığı — Notlar ve Kararlar

## Klasördeki Dosyalar

```
rl_dataset_clean.csv          → Ham temizlenmiş veri (14.400 satır)
sar2.py                       → SAR oluşturma scripti
sar2_augmented.py             → Augmentation eklenmiş SAR scripti
rl_dataset_sar_raw.csv        → Normalize edilmemiş SAR (14.390 satır)
rl_dataset_sar_normalized.csv → Normalize + augmented SAR (28.780 satır)
scaler.pkl                    → MinMaxScaler (eğitim verisiyle fit edildi)
reward_config.json            → p50 eşikleri ve servis ağırlıkları
```

---

## Adım 1 — SAR Formatı Oluşturma (`sar2.py`)

`rl_dataset_clean.csv` dosyasından `(state, action, reward, next_state)` tuplası üretildi.

### State
6 servis × 5 metrik = **30 boyut**

| Metrik | Açıklama |
|---|---|
| num_pods | Çalışan pod sayısı |
| cpu_usage | CPU kullanımı |
| mem_usage | Bellek kullanımı |
| request_rate | Saniyedeki istek sayısı |
| latency | Yanıt süresi (ms) |

### Action
Pod sayısının bir önceki adıma göre farkından türetildi:
```
diff > 0  →  +1  (scale up)
diff = 0  →   0  (hold)
diff < 0  →  -1  (scale down)
```
`groupby('user_load')` ile run sınırlarında yanlış geçiş oluşması engellendi.

### Reward Fonksiyonu
```
r = Σ tanh(1 - latency_i / p50_i) × weight_i  −  Σ(pod_sayısı) × 0.05
```

**Servis ağırlıkları:**

| Servis | Ağırlık | Reward'da? | Neden |
|---|---|---|---|
| shipping | 2.0 | Evet | Birincil bottleneck |
| cart | 1.5 | Evet | Direkt kullanıcı etkileşimi |
| catalogue | 1.5 | Evet | Direkt kullanıcı etkileşimi |
| user | 1.0 | Evet | Oturum yönetimi |
| ratings | 0.5 | Evet | Yardımcı servis |
| payment | 0.0 | **Hayır** | Option C — bkz. aşağıda |

**p50 eşikleri veri setinden otomatik hesaplandı** (sıfır olmayan latency değerleri üzerinden). `reward_config.json`'a kaydedildi.

#### Neden tanh kullandık?
- Aşırı latency değerlerinde gradient sinyalini korur
- Her servis katkısını `(-weight, +weight)` aralığına doğal olarak sıkıştırır
- Manuel clip'e gerek kalmaz

#### Payment neden reward dışında? (Option C)
Payment latency değerleri diğer servislerden çok farklı ölçekte (p50=2672ms vs cart p50=52ms). Reward fonksiyonuna dahil edilirse diğer servisleri baskılar ve öğrenmeyi bozar. Payment **state'e dahil** — Digital Twin servisler arası bağımlılıkları öğrenebilsin diye.

### Next State
Her run içinde bir sonraki adımın state'i shift(-1) ile alındı. Her run'ın son satırı düşürüldü (geçerli next_state yok). **14.400 → 14.390 satır.**

### Normalizasyon
- MinMaxScaler sadece **eğitim split'ine (%80) fit edildi** — data leakage yok
- Aralık dışı değerler [0,1]'e clip edildi
- Action (-1/0/+1) ve reward normalize edilmedi

---

## Adım 2 — Gaussian Augmentation (`sar2_augmented.py`)

### Neden Augmentation?
14.390 satır DT ve MARL eğitimi için yeterliydi ama daha fazla veri çeşitliliği için augmentation uygulandı.

### Yöntem
```
Ham veri → Gaussian gürültü ekle → Normalize et
```
Gürültü normalize edilmiş veriye değil, **ham veriye** eklendi. Bu doğru sıra — tersini yapmak scaler istatistiklerini bozar.

### Parametreler
| Parametre | Değer | Açıklama |
|---|---|---|
| NOISE_RATIO | 0.02 | Her kolonun std'sinin %2'si kadar gürültü |
| AUG_COPIES | 1 | 1 kopya → 2x veri |

### Neye Gürültü Eklendi, Neye Eklenmedi?

| Metrik | Gürültü | Neden |
|---|---|---|
| cpu_usage | ✅ Evet | Sürekli değişken |
| mem_usage | ✅ Evet | Sürekli değişken |
| request_rate | ✅ Evet | Sürekli değişken |
| latency | ✅ Evet | Sürekli değişken |
| **num_pods** | **❌ Hayır** | Ayrık tam sayı — 2.3 pod fiziksel olarak anlamsız |
| action | ❌ Hayır | Türetilmiş değer, değişmemeli |
| reward | ❌ Hayır | Aşağıda açıklandı |

### Bilinen Sınırlılıklar

**1. Reward yeniden hesaplanmadı**
Latency değişti ama reward kopyalandı. Teorik tutarsızlık var.
Pratikte etkisiz — DT reward sütununu kullanmıyor, MARL ise reward'ı SAR'dan değil DT simülasyonundan anlık hesaplıyor.

**2. next_state augmented edilmedi**
`(state_aug, action) → next_state_orig` geçişi fiziksel olarak tam tutarlı değil.
Pratikte etkisiz — DT MAE=0.032 ile gürültüyü absorbe etti.

**3. Payment latency=0 olan satırlar korundu**
Sıfır payment latency ölçüm eksikliği anlamına geliyor. Bu satırlarda sıfır değeri bozulmadan korundu. Orijinalde payment>0 iken augmented versiyonda sıfıra düşen ~140 satır temizlendi.

### Sonuç
```
Orijinal SAR : 14.390 satır
Augmented    : 14.390 satır (kopya)
Toplam       : 28.780 satır  (2x)
```

---

## Doğrulama Kontrolleri

`sar2.py` çalıştıktan sonra şunlar doğrulandı:

- ✅ Normalize aralık: [0.000000, 1.000000]
- ✅ Action değerleri: sadece {-1, 0, 1}
- ✅ Run sınırlarında ilk satır action = 0
- ✅ State tutarlılığı: `next_state[i] == state[i+1]` (tüm run'larda)

---

## Çıktı Dosyaları

| Dosya | Satır | Kullanım |
|---|---|---|
| `rl_dataset_sar_raw.csv` | 14.390 | Ham SAR, normalize edilmemiş |
| `rl_dataset_sar_normalized.csv` | 28.780 | DT ve MARL eğitimi için kullanılan asıl veri |
| `scaler.pkl` | — | Normalize/denormalize için kaydedildi |
| `reward_config.json` | — | p50 eşikleri, servis ağırlıkları |

---

## Sıradaki Adım

Digital Twin eğitimi:
```
dt2.py  →  digital_twin_best.pth  (MAE=0.032, R²=0.945)
```
