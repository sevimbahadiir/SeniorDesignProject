# Digital Twin Eğitimi — Notlar ve Kararlar

## Klasördeki Dosyalar

```
dt2.py                   → DT eğitim scripti
digital_twin_best.pth    → En iyi model ağırlıkları (bunu kullan)
digital_twin_final.pth   → Son epoch ağırlıkları
dt_scaler.pkl            → scaler.pkl'in kopyası (referans)
dt_training_log.csv      → Epoch bazında loss geçmişi
dt_training_plot.png     → Loss eğrisi + tahmin scatter grafiği
reward_config.json       → SAR adımından kopyalandı
scaler.pkl               → SAR adımından kopyalandı
rl_dataset_sar_normalized.csv → DT eğitim verisi (28.780 satır)
```

---

## Amaç

Digital Twin, gerçek Kubernetes cluster'a dokunmadan MARL ajanlarının eğitilebileceği bir simülasyon ortamı sağlar.

```
f(state, action) → next_state
```

36 boyutlu girdi (30 state + 6 action) → 30 boyutlu çıktı (next_state)

---

## Mimari

```
Input (36)
  → Linear(256) → BatchNorm → ReLU → Dropout(0.2)
  → Linear(256) → BatchNorm → ReLU → Dropout(0.2)
  → Linear(128) → BatchNorm → ReLU → Dropout(0.2)
  → Linear(30)  → Sigmoid
Output (30)
```

**Neden bu kararlar?**

| Karar | Neden |
|---|---|
| MLP (LSTM değil) | Her SAR satırı bağımsız (s,a,s') tuplası; temporal bağımlılık state'te zaten kodlanmış |
| Huber loss | Shipping latency aykırı değerlerine MSE'den daha dayanıklı |
| BatchNorm | Farklı ölçeklerdeki feature'lar (ms'lik latency vs 0-1 CPU) arasında eğitimi stabilize eder |
| Sigmoid çıktı | next_state [0,1] normalize aralığında — Sigmoid bunu garantiler |
| Dropout(0.2) | 14k satır veri için overfitting önlemi |
| Tek model | Tüm servisler tek modelde — cross-service bağımlılıklar (shipping bottleneck → payment) öğrenilir |

---

## Hiperparametreler

| Parametre | Değer |
|---|---|
| Hidden dims | [256, 256, 128] |
| Dropout | 0.2 |
| Batch size | 64 |
| Max epochs | 300 |
| Learning rate | 1e-3 (başlangıç) |
| LR scheduler | ReduceLROnPlateau (patience=15, factor=0.5) |
| Early stopping | patience=30 |
| Loss | Huber (δ=1.0) |
| Optimizer | Adam |

---

## Eğitim Sonuçları

| Metrik | Değer |
|---|---|
| **Best epoch** | **282** |
| **Best val loss (Huber)** | **0.001287** |
| **MAE** | **0.032** |
| **R²** | **0.945** |
| Total epochs çalıştı | 300 (early stop tetiklenmedi) |
| Final LR | 0.000125 (2 kez düşürüldü) |

**Yorumlama:**
- MAE = 0.032 → [0,1] normalize aralığında ortalama 0.032 hata — **iyi** (eşik < 0.05)
- R² = 0.945 → Varyansın %94.5'ini açıklıyor — **çok iyi**
- Val loss < Train loss → Overfitting yok, smooth convergence ✅

---

## Trajectory Validation Kararı

DT'nin otoregresif kullanımda (MARL sırasında) ne kadar güvenilir olduğunu test etmek için ayrı bir `test2.py` çalıştırıldı.

**Yöntem:** 200 başlangıç noktasından 8 adımlık rollout:
```
Gerçek veri → DT → tahmin₁ → DT → tahmin₂ → ... → tahmin₈
```

**Sonuçlar:**

| Adım | MAE | Durum |
|---|---|---|
| 1 | 0.033 | İyi ✓ |
| 2 | 0.045 | İyi ✓ |
| 3 | 0.054 | Kabul edilebilir |
| 4 | 0.063 | Kabul edilebilir |
| 5 | 0.069 | Kabul edilebilir |
| **6** | **0.076** | **Eşiğin altında ✓** |
| 7 | 0.079 | Sınırda |
| 8 | 0.082 | Eşik aşıldı |

**Karar: MAX_STEPS = 6**
Step 6 MAE = 0.076 < 0.08 eşiği → MARL ortamında her episode maksimum 6 adım kullanılacak.

Shipping servisi en yüksek drift gösterdi (MAE≈0.118) — volatil bir servis olduğu için beklenen.

---

## Bilinen Sınırlılıklar

**1. Augmented veri kullanıldı**
28.780 satırın yarısı Gaussian augmentation ile üretildi. DT augmented veriyi orijinalden ayırt edemiyor — bu hem avantaj (daha fazla çeşitlilik) hem dezavantaj (sentetik geçişler).

**2. Tek model tüm servisleri tahmin ediyor**
Cross-service bağımlılıkları öğrenmesi için avantajlı, ama hata bir serviste büyükse diğerlerini de etkiliyor. Shipping latency tahminleri diğerlerinden daha yüksek hata gösterdi.

**3. Gerçek cluster dinamiğinin tam kopyası değil**
DT bir yaklaşım. Özellikle ani trafik spikeleri ve pod başlatma gecikmelerini tam yansıtmıyor. MARL bu simülasyonda öğreniyor — gerçek cluster'da performans farklı olabilir.

---

## Kullanım

```python
# MARL ortamında DT yükleme
model = DigitalTwin(input_dim=36, output_dim=30, ...)
model.load_state_dict(torch.load('digital_twin_best.pth'))
model.eval()

# Bir adım simülasyonu
next_state = model(torch.tensor([state + action]))
```

**Her zaman `digital_twin_best.pth` kullan** — `digital_twin_final.pth` değil. Best checkpoint epoch 282'deki en düşük val loss'a sahip.

---

## Sıradaki Adım

MARL eğitimi:
```
train_ippo.py   →  MARL ortamı DT üzerinde çalışır (MAX_STEPS=6)
train_mappo.py
train_dqn.py
train_a2c.py
```
