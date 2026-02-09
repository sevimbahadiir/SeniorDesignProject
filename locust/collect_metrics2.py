import requests
import time
import csv
from datetime import datetime

PROMETHEUS_URL = "http://localhost:9090/api/v1/query"

# Makaledeki Tablo V'e göre kısıtladığımız servisler
SERVICES = ['web', 'cart', 'payment']

def get_metric(query):
    try:
        response = requests.get(PROMETHEUS_URL, params={'query': query}, timeout=5)
        results = response.json()['data']['result']
        if results:
            # Bazı sonuçlar liste döner, ilk değeri alıyoruz
            return float(results[0]['value'][1])
        return 0
    except Exception as e:
        # Hata durumunda 0 dön ama arka planda hata olduğunu bil
        return 0

print("--- Robot-shop Veri Toplama Başladı ---")
print("İpucu: Eğer Latency hala 0 ise Locust'ta trafik olduğundan emin ol.")

with open('robot_shop_dataset.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    headers = ['timestamp', 'latency']
    for s in SERVICES:
        headers.extend([f'{s}_pods', f'{s}_cpu', f'{s}_mem'])
    writer.writerow(headers)

    while True:
        # 1. Gecikme (Latency) - Milisaniye cinsinden
        # Robot-shop 'web' servisinin genel yanıt süresini hesaplamaya çalışır
        # Senin Prometheus ekranındaki tam metrik isimlerini kullanan garantici sorgu
        latency_query = 'sum(rate(http_request_duration_seconds_sum{namespace="robot-shop"}[2m])) / sum(rate(http_request_duration_seconds_count{namespace="robot-shop"}[2m])) * 1000'
        latency = get_metric(latency_query)
        
        # Eğer yukarıdaki spesifik sorgu 0 dönerse, daha genel bir metrik deneyelim
        if latency == 0:
            latency = get_metric('avg(http_request_duration_seconds{kubernetes_namespace="robot-shop"}) * 1000')

        row = [datetime.now().strftime("%H:%M:%S"), round(latency, 2)]
        
        for s in SERVICES:
            # 2. Pod Sayısı (numPods)
            pod_query = f'count(kube_pod_container_status_running{{namespace="robot-shop", container="{s}"}})'
            pods = get_metric(pod_query)
            
            # 3. CPU Kullanımı (cpuAggr) - Millicore (m) cinsinden
            # Kind/Docker ortamında en sağlıklı çalışan CPU sorgusu:
            cpu_query = f'sum(rate(container_cpu_usage_seconds_total{{namespace="robot-shop", pod=~"{s}-.*", container!=""}}[2m])) * 1000'
            cpu = get_metric(cpu_query)
            
            # 4. Bellek Kullanımı (memAggr) - MiB cinsinden
            mem_query = f'sum(container_memory_working_set_bytes{{namespace="robot-shop", pod=~"{s}-.*", container!=""}}) / 1024 / 1024'
            mem = get_metric(mem_query)
            
            row.extend([int(pods) if pods > 0 else 1, round(cpu, 2), round(mem, 2)])
        
        writer.writerow(row)
        f.flush()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Kaydedildi -> Latency: {row[1]}ms | Web CPU: {row[3]}m | Pods: {row[2]},{row[5]},{row[8]}")
        time.sleep(10)