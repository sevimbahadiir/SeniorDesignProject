# Robot Shop RL — Kurulum ve Veri Toplama Rehberi

> **Proje:** Digital Twin for Microservice Architecture Performance and Dynamic Auto-Scaling Optimization using Reinforcement Learning  
> **Grup:** Eylem Kahveci, Sevim Bahadıroğlu, Şevval Yavaş  
> **Danışman:** Doç. Dr. Tuğba Dalyan  

---

## İçindekiler

1. [Gereksinimler](#1-gereksinimler)
2. [WSL ve Docker Ayarları](#2-wsl-ve-docker-ayarları)
3. [Kind Kubernetes Cluster Kurulumu](#3-kind-kubernetes-cluster-kurulumu)
4. [Robot Shop Deploy](#4-robot-shop-deploy)
5. [Istio Kurulumu](#5-istio-kurulumu)
6. [Prometheus ve Grafana Kurulumu](#6-prometheus-ve-grafana-kurulumu)
7. [Veri Toplama Pipeline'ı](#7-veri-toplama-pipelineı)
8. [Veri Toplama Koşulları](#8-veri-toplama-koşulları)
9. [Sorun Giderme](#9-sorun-giderme)

---

## 1. Gereksinimler

Bilgisayarınızda aşağıdakilerin kurulu olması gerekmektedir:

| Araç | Sürüm | İndirme |
|------|-------|---------|
| Docker Desktop | En son | https://www.docker.com/products/docker-desktop |
| kubectl | En son | https://kubernetes.io/docs/tasks/tools/ |
| Kind | En son | https://kind.sigs.k8s.io/docs/user/quick-start/ |
| Helm | En son | https://helm.sh/docs/intro/install/ |
| Python | 3.11+ | https://www.python.org/downloads/ |
| Locust | En son | `pip install locust` |

**Python kütüphaneleri:**
```bash
pip install requests pandas numpy locust
```

**Donanım gereksinimleri (minimum):**
- RAM: 16 GB
- CPU: 8 çekirdek
- Disk: 50 GB boş alan

---

## 2. WSL ve Docker Ayarları

Docker Desktop, WSL 2 backend ile çalışmaktadır. Performans için kaynak limitlerini artırmanız gerekmektedir.

### 2.1 WSL Konfigürasyonu

PowerShell'i açın ve aşağıdaki komutu çalıştırın:

```powershell
@"
[wsl2]
memory=12GB
processors=6
swap=4GB
"@ | Out-File -FilePath "C:\Users\<KULLANICI_ADI>\.wslconfig" -Encoding utf8
```

> ⚠️ `<KULLANICI_ADI>` kısmını kendi Windows kullanıcı adınızla değiştirin.

### 2.2 WSL'i Yeniden Başlatın

```powershell
wsl --shutdown
```

### 2.3 Docker Desktop'ı Yeniden Başlatın

Sistem tepsisinde Docker ikonuna sağ tık → **Restart**

### 2.4 Doğrulama

Docker Desktop açıldıktan sonra **Containers** sekmesinde CPU limiti **600%** (6 CPU) ve memory **11.41GB** görünmelidir.

---

## 3. Kind Kubernetes Cluster Kurulumu

### 3.1 Kind Cluster Oluştur

```powershell
kind create cluster --name robot-shop --config kind-config.yaml
```

`kind-config.yaml` dosyası:
```yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
  - role: worker
  - role: worker
```

### 3.2 Cluster Durumunu Kontrol Et

```powershell
kubectl get nodes
```

Beklenen çıktı:
```
NAME                       STATUS   ROLES           AGE   VERSION
robot-shop-control-plane   Ready    control-plane   ...   v1.27.3
robot-shop-worker          Ready    <none>          ...   v1.27.3
robot-shop-worker2         Ready    <none>          ...   v1.27.3
```

> ✅ Tüm node'lar **Ready** olmalıdır.

---

## 4. Robot Shop Deploy

### 4.1 Namespace Oluştur

```powershell
kubectl create namespace robot-shop
```

### 4.2 Helm ile Robot Shop Kur

```powershell
helm install robot-shop --namespace robot-shop .
```

> 📁 Robot Shop Helm chart'ı için: https://github.com/instana/robot-shop

### 4.3 Pod'ları Kontrol Et

```powershell
kubectl get pods -n robot-shop
```

Tüm pod'ların `2/2 Running` durumunda olması beklenir. Bu birkaç dakika sürebilir.

> ⚠️ `shipping` servisi MySQL'e bağlandığı için diğerlerinden daha yavaş başlar (~30 saniye). `1/2 Running` görünmesi normaldir, bekleyin.

### 4.4 Takılı Pod'u Sil (Gerekirse)

Bir pod uzun süre `1/2 Running` kalırsa:
```powershell
kubectl delete pod <POD_ADI> -n robot-shop
```

---

## 5. Istio Kurulumu

Istio, servisler arası trafik metriklerini toplamak için kullanılmaktadır.

### 5.1 Istio'yu İndir

```powershell
# Windows için: https://github.com/istio/istio/releases adresinden indirin
# istioctl.exe'yi PATH'e ekleyin
```

### 5.2 Istio'yu Kur

```powershell
istioctl install --set profile=demo -y
```

### 5.3 Robot Shop Namespace'ine Sidecar Injection Ekle

```powershell
kubectl label namespace robot-shop istio-injection=enabled
```

### 5.4 Istio Eklentilerini Kur (Prometheus, Grafana, Kiali)

```powershell
kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.21/samples/addons/prometheus.yaml
kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.21/samples/addons/grafana.yaml
kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.21/samples/addons/kiali.yaml
```

---

## 6. Prometheus ve Grafana Kurulumu

### 6.1 Port-Forward Otomasyonu

`baslat.ps1` dosyasını proje klasörüne kopyalayın ve çalıştırın:

```powershell
cd C:\Users\<KULLANICI_ADI>\robot-shop\robot-shop-rl
powershell -ExecutionPolicy Bypass -File baslat.ps1
```

Bu script şu port-forward'ları otomatik başlatır ve düşünce yeniden bağlar:

| Servis | URL |
|--------|-----|
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |
| Web (Robot Shop) | http://localhost:8080 |

### 6.2 Manuel Port-Forward (Alternatif)

```powershell
# Terminal 1
kubectl port-forward -n istio-system svc/prometheus-server 9090:80

# Terminal 2
kubectl port-forward -n istio-system svc/grafana 3000:80

# Terminal 3
kubectl port-forward -n robot-shop svc/web 8080:8080
```

### 6.3 Doğrulama

Tarayıcıda http://localhost:9090 adresini açın. Prometheus arayüzü görünmelidir.

---

## 7. Veri Toplama Pipeline'ı

Veri toplama 3 script'in eş zamanlı çalışmasıyla gerçekleşir:

```
robot-shop.py (Locust)     → Trafik üretir
random_scaler_v2.py        → Pod sayılarını değiştirir
veri_cek_v2.py             → Prometheus'tan metrikleri çeker → CSV
```

### 7.1 Başlamadan Önce

Pod'ları başlangıç değerlerine döndürün:

```powershell
kubectl scale deployment cart catalogue payment shipping user ratings dispatch --replicas=2 -n robot-shop
```

Pod'ların hazır olduğunu kontrol edin:

```powershell
kubectl get pods -n robot-shop
```

Tüm pod'lar `2/2 Running` olmalıdır.

### 7.2 Terminal 1 — Port-Forward

```powershell
cd C:\Users\<KULLANICI_ADI>\robot-shop\robot-shop-rl
powershell -ExecutionPolicy Bypass -File baslat.ps1
```

### 7.3 Terminal 2 — Locust (Trafik Üretici)

```powershell
cd C:\Users\<KULLANICI_ADI>\robot-shop\robot-shop-rl
locust -f robot-shop.py --host=http://localhost:8080 --headless -u <KULLANICI_SAYISI> -r <RAMP_UP>
```

### 7.4 Terminal 3 — Random Scaler

```powershell
cd C:\Users\<KULLANICI_ADI>\robot-shop\robot-shop-rl
python random_scaler_v2.py
```

### 7.5 Run Bitince — Veri Toplama

```powershell
cd C:\Users\<KULLANICI_ADI>\robot-shop\robot-shop-rl
python veri_cek_v2.py
```

### 7.6 veri_cek_v2.py Ayarları

Her run için şu satırları güncelleyin:

```python
WINDOW_MINUTES = 120        # Run süresiyle eşleşmeli
STEP = '30s'                # Değiştirmeyin
out = 'dataset_XXXX.csv'    # Her run için farklı isim
```

---

## 8. Veri Toplama Koşulları

Tüm run'lar `random_scaler_v2.py` ile yapılmıştır. Script her 45 saniyede bir pod sayılarını değiştirir.

**Script ayarları:**
```python
DURATION_MINUTES = 120   # Run 3-5 için, Run 1-2 için 60
INTERVAL_SECONDS = 45
MIN_PODS = 1
MAX_PODS = 5
# Shipping minimum 2 pod (MySQL timeout sorunu)
# Payment minimum 2 pod (checkout zinciri)
```

### Tamamlanan Run'lar

| Run | Kullanıcı | Ramp-Up | Süre | Locust Komutu | Çıktı Dosyası |
|-----|-----------|---------|------|---------------|---------------|
| Run 1 | 150 | `-r 10` | 60 dk | `locust -f robot-shop.py --host=http://localhost:8080 --headless -u 150 -r 10` | `istio_robot_shop_rl_dataset_v2.csv` |
| Run 2 | 80 | `-r 5` | 60 dk | `locust -f robot-shop.py --host=http://localhost:8080 --headless -u 80 -r 5` | `istio_robot_shop_rl_dataset_80user.csv` |
| Run 3 | 40 | `-r 3` | 120 dk | `locust -f robot-shop.py --host=http://localhost:8080 --headless -u 40 -r 3` | `istio_robot_shop_rl_dataset_40user.csv` |
| Run 4 | 120 | `-r 8` | 120 dk | `locust -f robot-shop.py --host=http://localhost:8080 --headless -u 120 -r 8` | `istio_robot_shop_rl_dataset_120user.csv` |
| Run 5 | 200 | `-r 10` | 120 dk | `locust -f robot-shop.py --host=http://localhost:8080 --headless -u 200 -r 10` | `istio_robot_shop_rl_dataset_200user.csv` |

### Veri Kalitesi Hedefleri

| Metrik | Hedef |
|--------|-------|
| Sıfır-latency oranı | < %30 |
| Toplam satır (tüm run'lar) | ~1080 satır |
| Pod aralığı (cart, catalogue, ratings, user) | 1-5 |
| Pod aralığı (shipping, payment) | 2-5 |

### CSV Sütun Yapısı

Her CSV dosyası şu sütunları içerir:

```
date
cartservice_num_pods, cartservice_cpu_usage, cartservice_mem_usage, cartservice_request_rate, cartservice_latency
catalogueservice_num_pods, catalogueservice_cpu_usage, ...
dispatchservice_num_pods, ...          ← request_rate ve latency her zaman 0 (RabbitMQ servisi)
paymentservice_num_pods, ...
ratingsservice_num_pods, ...
shippingservice_num_pods, ...
userservice_num_pods, ...
webservice_num_pods, ...               ← request_rate ve latency her zaman 0 (nginx proxy)
```

> ⚠️ **Önemli Not:** `num_pods` sütunları Istio sidecar container'larını da saydığı için gerçek değerin ~2-3 katıdır. Veri işleme aşamasında düzeltilecektir.

---

## 9. Sorun Giderme

### Problem: `TLS handshake timeout`

```
Unable to connect to the server: net/http: TLS handshake timeout
```

**Neden:** Kubernetes API server yanıt vermiyor, Docker aşırı yüklenmiş.

**Çözüm:**
1. Docker Desktop'ı yeniden başlatın
2. WSL'i yeniden başlatın: `wsl --shutdown`
3. `.wslconfig` dosyasında CPU ve memory limitlerini kontrol edin

---

### Problem: `EOF` hatası

```
Unable to connect to the server: EOF
```

**Neden:** Docker yeni başlatıldı, API server henüz hazır değil.

**Çözüm:** 1-2 dakika bekleyin, tekrar deneyin.

---

### Problem: Shipping pod'u `1/2 Running` kalıyor

**Neden:** Shipping servisi MySQL'e bağlanırken readiness probe timeout alıyor.

**Çözüm 1:** Bekleyin (~30 saniye), kendiliğinden `2/2` olur.

**Çözüm 2:** Pod'u silin, Kubernetes yenisini açar:
```powershell
kubectl delete pod <SHIPPING_POD_ADI> -n robot-shop
```

---

### Problem: Locust JSON hatası

```
requests.exceptions.JSONDecodeError: Expecting value: line 1 column 1
```

**Neden:** Random scaling sırasında servis geçici olarak erişilemez oldu.

**Çözüm:** Güncellenmiş `robot-shop.py` dosyasını kullanın (hata toleranslı versiyon).

---

### Problem: Yüksek CPU kullanımı (`%500+`)

**Neden:** Docker'a yeterli CPU ayrılmamış.

**Çözüm:** `.wslconfig` dosyasında `processors=6` olduğunu kontrol edin:
```powershell
Get-Content C:\Users\<KULLANICI_ADI>\.wslconfig
```

---

### Problem: Port-forward düşüyor

**Neden:** Kubernetes API server ile bağlantı kesildi.

**Çözüm:** `baslat.ps1` scripti otomatik yeniden bağlanır. Manuel olarak da yeniden başlatabilirsiniz.

---

### Problem: Prometheus'tan veri gelmiyor (0 nokta)

**Neden:** Port-forward çalışmıyor veya Locust trafik üretmiyor.

**Çözüm:**
```powershell
# Port-forward'ı kontrol et
curl http://localhost:9090 -UseBasicParsing

# Locust'un çalıştığını doğrula
curl http://localhost:8080/api/user/uniqueid -UseBasicParsing
```

---

## Notlar

- `dispatch` ve `web` servisleri HTTP Istio metriği üretmez — bu normaldir.
- `payment` servisi az trafik alır (checkout'un son adımı) — bu normaldir.
- Tüm run'lar bittikten sonra CSV'ler birleştirilip `num_pods` sütunları düzeltilecektir.
- Sonraki aşama: State/Action/Reward dönüşümü → Digital Twin eğitimi → MARL ajanları.
