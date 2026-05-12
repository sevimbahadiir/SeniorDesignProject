# Trajectory Validation — Notlar ve Kararlar

## Klasördeki Dosyalar

```
test2.py                     → Trajectory validation scripti
trajectory_results.csv       → Adım bazında MAE, RMSE, R² sonuçları
trajectory_plot.png          → 4-panel görselleştirme
digital_twin_best.pth        → DT klasöründen kopyalandı
dt_scaler.pkl                → DT klasöründen kopyalandı
rl_dataset_sar_normalized.csv → SAR klasöründen kopyalandı
```

---

## Amaç

Digital Twin tek adım tahmininde iyi sonuç veriyor (MAE=0.032). Ama MARL eğitiminde DT **otoregresif** kullanılıyor — yani DT'nin çıktısı bir sonraki adımın girdisi oluyor. Hata birikip birikmediğini test etmek için trajectory validation yapıldı.

```
Gerçek veri → DT → tahmin₁ → DT → tahmin₂ → ... → tahmin₈
                   (gerçek veriye bakmadan)
```

---

## Yöntem

- 200 başlangıç noktası (validation set'ten rastgele seçildi)
- Her başlangıçtan 8 adımlık otoregresif rollout
- Her adımda: tahmin edilen state vs gerçek state karşılaştırıldı
- `RANDOM_SEED = 42` — reproducible ✅

---

## Sonuçlar

| Adım | MAE | RMSE | R² | Durum |
|---|---|---|---|---|
| 1 | 0.033 | 0.054 | 0.936 | Excellent |
| 2 | 0.045 | 0.071 | 0.889 | Good |
| 3 | 0.054 | 0.085 | 0.842 | Acceptable |
| 4 | 0.063 | 0.099 | 0.791 | Acceptable |
| 5 | 0.069 | 0.107 | 0.759 | Acceptable |
| **6** | **0.076** | **0.117** | **0.709** | **Acceptable ✅** |
| 7 | 0.079 | 0.122 | 0.683 | Sınırda |
| 8 | 0.082 | 0.127 | 0.650 | ❌ Eşik aşıldı |

**Eşik:** MAE < 0.08 → kabul edilebilir

---

## MAX_STEPS = 6 Kararı

Step 6 MAE = **0.076 < 0.08** → eşiğin altında ✅  
Step 8 MAE = **0.082 > 0.08** → eşik aşıldı ❌

**Karar:** MARL eğitiminde her episode maksimum **6 adım** kullanılacak.

Bu karar iki şeyi dengeler:
- Çok kısa (1-2 adım) → MARL yeterince öğrenemiyor
- Çok uzun (8+ adım) → DT hatası birikip öğrenmeyi bozuyor

---

## Servis Bazında Hata (Step 8)

Shipping servisi en yüksek drift'i gösterdi — volatil ve yüksek latency'li bir servis olduğu için beklenen. Diğer servisler daha stabil seyretti.

---

## Bilinen Sınırlılıklar

**1. Gerçek aksiyon kullanıldı**
Rollout sırasında `action` değeri gerçek veriden alındı (SAR dosyasından). Gerçekte MARL ajanı kendi aksiyonunu üretir — bu hafif bir iyimser tahmin anlamına gelir. Sonuçlar gerçek MARL kullanımından biraz daha iyi görünebilir.

**2. dt2.py'de seed yoktu**
DT eğitimi her çalıştırmada farklı başlangıç ağırlıklarıyla başlıyordu. Test bu DT üzerinde yapıldı. DT'nin farklı bir checkpoint'i olsaydı sonuçlar biraz değişebilirdi.

---

## Reproducibility

```python
RANDOM_SEED = 42   # test2.py'de mevcut ✅
```

Aynı DT checkpoint'i ve aynı seed ile tekrar çalıştırılırsa aynı sonuçlar üretilir.

---

## Tekrar Çalıştırma Gerekiyor Mu?

**Hayır.** Sonuçlar kaydedildi, MAX_STEPS=6 kararı alındı ve MARL eğitiminde kullanıldı. Tekrar çalıştırmak aynı sonucu verir (seed=42 mevcut).

---

## Sıradaki Adım

MARL eğitimi — MAX_STEPS=6 ile:
```
train_ippo.py    →  IPPO eğitimi
train_mappo.py   →  MAPPO eğitimi
train_dqn.py     →  DQN eğitimi
train_a2c.py     →  A2C eğitimi
```
