# Veri Ön İşleme — Adım 3: digital_twin.py

## Genel Amaç

Digital Twin (DT), SAR verisinden sistem dinamiklerini öğrenen
bir sinir ağı modelidir.

```
f(state, action) → next_state
```

MARL ajanları bu modeli **simülasyon ortamı** olarak kullanır.
Gerçek Kubernetes cluster'ına dokunmadan binlerce deneme yapılabilir.

---

## Giriş / Çıkış

| | Boyut | İçerik |
|-|-------|--------|
| Girdi | 36 | 30 state + 6 action |
| Çıktı | 30 | next_state (6 servis × 5 metrik) |

---

## Tasarım Kararları

### 1. Mimari: MLP (Feedforward Network)

**Seçim:** MLP — LSTM veya Transformer değil.

**Gerekçe:**
Her SAR satırı zaten tam bir transition tupledir: (s, a, s').
Temporal bağımlılık veri yapısı içinde örtük olarak temsil edilmektedir.
LSTM için ardışık pencere gerekir; bu hem run sınırlarında sorun
çıkarır hem de implementasyonu karmaşıklaştırır.

**Rapor için:**
> *"Each SAR row is a self-contained (s, a, s') tuple; temporal
> dependencies are implicitly captured through the state
> representation. An MLP architecture was therefore chosen over
> recurrent alternatives."*

**Katman boyutları:**
```
36 → 256 → 256 → 128 → 30
```

Her gizli katman sonrası:
- **BatchNorm:** Heterojen özellik ölçeklerini stabilize eder
- **ReLU:** Non-linearity
- **Dropout(0.2):** Overfitting önlemi

Çıktı katmanı: **Sigmoid** (veri normalize edilmiş [0,1] olduğundan)

**Toplam parametre:** 113.310

---

### 2. Tek Model vs. Servis Bazlı Ayrı Modeller

**Seçim:** Tek model.

**Gerekçe:**
6 ayrı model servisler arası bağımlılığı öğrenemez. Projenin
en önemli bulgularından biri "shipping tıkandığında payment
etkileniyor" ilişkisidir. Bu bağımlılık tek modelde öğrenilir.

**Rapor için:**
> *"A single model architecture was chosen to capture cross-service
> dependencies, particularly the causal relationship between
> shipping latency and payment traffic availability."*

---

### 3. Loss Fonksiyonu: Huber Loss

**Seçim:** Huber Loss (delta=1.0) — MSE veya MAE değil.

| Loss | Davranış | Sorun |
|------|----------|-------|
| MSE | Büyük hatalara çok duyarlı | Shipping outlier'ları domine eder |
| MAE | Uniform gradient | Küçük hataları yeterince optimize etmez |
| **Huber** | Küçük hata → MSE, büyük hata → MAE | Her ikisinin avantajı |

```
L(y, ŷ) = 0.5(y-ŷ)²           if |y-ŷ| ≤ δ
         = δ|y-ŷ| - 0.5δ²     otherwise
```

**Rapor için:**
> *"Huber loss was employed to balance sensitivity to small
> prediction errors with robustness to outliers in the shipping
> latency distribution."*

---

### 4. Optimizer ve Learning Rate

| Parametre | Değer | Gerekçe |
|-----------|-------|---------|
| Optimizer | Adam | Adaptive learning rate, standart |
| Initial LR | 1e-3 | Standart başlangıç |
| LR decay | ReduceLROnPlateau | Val loss iyileşmezse LR'yi düşür |
| Patience | 15 epoch | Erken düşürme, ama sinirli değil |
| Factor | 0.5 | Her tetiklemede yarıya indir |

---

### 5. Early Stopping

Val loss 30 epoch boyunca iyileşmezse eğitim durur.

**Neden gerekli?**
14.390 satır orta ölçekli bir dataset. Çok fazla epoch
train loss'u minimize ederken val loss'u artırabilir (overfitting).
Early stopping en iyi val loss'u veren ağırlıkları kaydeder.

---

### 6. Batch Size: 64

| Batch Size | Sorun |
|-----------|-------|
| 32 | Gürültülü gradient, yavaş |
| **64** | Denge noktası ✓ |
| 128 | Generalizasyon zayıflar |

---

## Eğitim Sonuçları (50 Epoch Test)

| Metrik | Değer | Yorum |
|--------|-------|-------|
| Best Val Loss (Huber) | 0.002244 | Çok iyi |
| MAE | 0.042704 | Good (< 0.05 eşiği) |
| RMSE | 0.066986 | Kabul edilebilir |
| R² | 0.903931 | Varyansın %90'ı açıklanıyor |

**Not:** Bu sonuçlar sadece 50 epoch ile elde edilmiştir.
300 epoch ile MAE'nin 0.01-0.03 aralığına düşmesi beklenmektedir.

### Per-Service MAE (50 epoch)

| Servis | num_pods | cpu | mem | rps | latency |
|--------|----------|-----|-----|-----|---------|
| cart | 0.062 | 0.025 | 0.051 | 0.027 | 0.037 |
| catalogue | 0.066 | 0.022 | 0.047 | 0.019 | 0.039 |
| payment | 0.063 | 0.054 | 0.049 | 0.007 | 0.015 |
| shipping | 0.052 | 0.043 | 0.052 | 0.056 | 0.055 |
| ratings | 0.047 | 0.035 | 0.050 | 0.037 | 0.058 |
| user | 0.062 | 0.036 | 0.051 | 0.031 | 0.034 |

**Dikkat çeken:** Payment latency MAE = 0.015 — en düşük.
Bu paradoksal görünebilir ama mantıklı: payment çoğunlukla
0 (sıfır), model "0 tahmin et" diyerek düşük hata yapıyor.
Makalede bunu not etmek gerekir.

---

## Kaydedilen Dosyalar

| Dosya | Açıklama |
|-------|----------|
| digital_twin_best.pth | En iyi val loss'taki ağırlıklar (kullan) |
| digital_twin_final.pth | Son epoch ağırlıkları |
| dt_training_log.csv | Epoch bazlı loss geçmişi |
| dt_training_plot.png | Loss eğrisi + tahmin scatter |
| dt_scaler.pkl | Scaler referans kopyası |

---

## MARL İçin Kullanım

DT eğitildikten sonra MARL ajanları şu şekilde kullanır:

```python
# Mevcut state + alınan action
state_action = torch.tensor([...state..., ...action...])

# DT ile next_state tahmin et
with torch.no_grad():
    next_state = dt_model(state_action)

# next_state'den reward hesapla
reward = compute_reward(next_state)

# Ajan bu reward'dan öğrenir
agent.update(state, action, reward, next_state)
```

---

## Literatür Bağlantısı

- **MLP for system modeling:**
  Lazic et al. (2018) — *"Data center cooling using model-predictive
  control"* — sistem dinamiklerini öğrenmek için feedforward NN

- **Huber loss in RL:**
  Mnih et al. (2015) — *"Human-level control through deep
  reinforcement learning"* (DQN) — Huber loss'u RL'de ilk
  yaygınlaştıran çalışma

- **BatchNorm:**
  Ioffe & Szegedy (2015) — *"Batch Normalization: Accelerating
  Deep Network Training"*

- **Early stopping:**
  Prechelt (1998) — *"Early Stopping — But When?"*

---

## Sonraki Adım

`digital_twin_best.pth` → `marl_train.py`
