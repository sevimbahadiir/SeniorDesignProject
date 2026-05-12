"""
marl_env.py
-----------
Adım 1: MARL Ortamı (Environment)

Digital Twin'i saran Gym-benzeri çok-ajanlı ortam.
Her çağrıda tüm agentlar eş zamanlı aksiyon verir,
DT bir sonraki global state'i tahmin eder,
her agent kendi partial observation'ını alır.

Servis bağımlılık grafiği (robot_shop.py çağrı zincirinden):
    web → catalogue → cart → payment
                  ↘ shipping
             user

Partial Observability:
    Her agent kendi 5 metriğini + komşu servislerinin metriklerini görür.
    Agent başına obs boyutu: 10-15 arası (komşu sayısına göre).

State boyutu  : 30  (6 servis × 5 metrik, normalize [0,1])
Action/agent  : {0: scale_down, 1: keep, 2: scale_up}  ← DQN/SB3 uyumlu
Global action : 6 boyutlu  (-1/0/+1 iç temsil)
Episode uzunluğu: max 10 adım (DT trajectory güvenilirlik sınırı)

Dosya bağımlılıkları:
    digital_twin_best.pth       — DT model ağırlıkları
    reward_config.json          — p50 threshold'lar ve servis ağırlıkları
    scaler.pkl                  — MinMaxScaler (normalize/denormalize)
"""

import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


# ─── Sabitler ────────────────────────────────────────────────────────────────

SERVICES = ['cart', 'catalogue', 'payment', 'shipping', 'ratings', 'user']
METRICS  = ['num_pods', 'cpu_usage', 'mem_usage', 'request_rate', 'latency']

# Her servisin state vektörü içindeki başlangıç indeksi
# Örn: cart → [0:5], catalogue → [5:10], ...
SERVICE_IDX = {svc: i * len(METRICS) for i, svc in enumerate(SERVICES)}

# Partial Observability: her agent hangi servisleri gözlemleyebilir?
# Kendi servisi + robot_shop.py gerçek çağrı akışındaki komşuları
#
# robot_shop.py load() akışı:
#   user/uniqueid → catalogue/products → ratings/rate (% 30)
#                → cart/add → cart/cart → shipping/confirm → payment/pay
#
# Her servis: upstream (kendini tetikleyeni) + downstream (tetiklediği) görür.
#
#   user      → downstream: cart          (session ID cart'a gidiyor)
#   catalogue → downstream: ratings, cart (ürünler ratings ve cart'a besleniyor)
#   ratings   → upstream:   catalogue     (catalogue tarafından tetikleniyor)
#   cart      → upstream:   catalogue     (catalogue'dan item alıyor)
#             → downstream: shipping      (cart içeriği shipping'e gidiyor)
#   shipping  → upstream:   cart          (cart içeriğini confirm ediyor)
#             → downstream: payment       (confirm sonucu payment'a gidiyor)
#   payment   → upstream:   shipping      (shipping confirm çıktısını alıyor)
AGENT_NEIGHBORS: Dict[str, List[str]] = {
    'cart':      ['catalogue', 'shipping'],  # catalogue'dan input, shipping'e output
    'catalogue': ['ratings',   'cart'],      # ratings ve cart'ı besliyor
    'payment':   ['shipping'],               # shipping confirm'den sonra tetikleniyor
    'shipping':  ['cart',      'payment'],   # cart'tan input, payment'a output
    'ratings':   ['catalogue'],              # yalnızca catalogue tarafından tetikleniyor
    'user':      ['cart'],                   # session → cart akışı
}

# Aksiyon uzayı (agent perspektifinden)
# DQN/PPO uyumluluğu için 0,1,2 → içeride -1,0,+1'e çevrilir
ACTION_MAP = {0: -1, 1: 0, 2: 1}   # discrete → internal
ACTION_MAP_INV = {-1: 0, 0: 1, 1: 2}

N_ACTIONS = 3        # {scale_down, keep, scale_up}
MAX_STEPS = 6        # DT trajectory güvenilirlik sınırı (Step 6 MAE<0.08)


# ─── Digital Twin Model Tanımı ───────────────────────────────────────────────
# dt2.py ile birebir aynı olmalı!

class DigitalTwin(nn.Module):
    """
    MLP tabanlı Digital Twin.
    Mimari: 36 → 256 → 256 → 128 → 30
    Her gizli katman: Linear → BatchNorm → ReLU → Dropout
    Çıktı: Sigmoid ([0,1] normalize edilmiş next_state)
    """
    def __init__(self, input_dim=36, output_dim=30,
                 hidden_dims=(256, 256, 128), dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers += [nn.Linear(prev, output_dim), nn.Sigmoid()]
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ─── MARL Environment ────────────────────────────────────────────────────────

class MarlEnvironment:
    """
    Digital Twin üzerine kurulu çok-ajanlı RL ortamı.

    Kullanım:
        env = MarlEnvironment()
        obs = env.reset(initial_state)   # Dict[str, np.ndarray]
        obs, rewards, done, info = env.step(actions)  # actions: Dict[str, int]

    Parametreler:
        dt_path         : Digital Twin model dosyası (.pth)
        reward_cfg_path : Reward konfigürasyon dosyası (.json)
        scaler_path     : MinMaxScaler dosyası (.pkl)
        seed            : Tekrarlanabilirlik için random seed
    """

    def __init__(
        self,
        dt_path: str = 'digital_twin_best.pth',
        reward_cfg_path: str = 'reward_config.json',
        scaler_path: str = 'scaler.pkl',
        seed: Optional[int] = 42,
    ):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.device = torch.device('cpu')  # inference için CPU yeterli

        # Digital Twin yükle
        self.dt = DigitalTwin().to(self.device)
        self.dt.load_state_dict(
            torch.load(dt_path, map_location=self.device, weights_only=False)
        )
        self.dt.eval()

        # Reward konfigürasyonu yükle
        with open(reward_cfg_path, 'r') as f:
            self.reward_cfg = json.load(f)

        # Scaler yükle (denormalize için ileride kullanılabilir)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        # Observation space boyutları (servis başına)
        # Kendi 5 metriği + her komşu için 5 metrik
        self.obs_dims: Dict[str, int] = {
            svc: (1 + len(AGENT_NEIGHBORS[svc])) * len(METRICS)
            for svc in SERVICES
        }

        # Episode state
        self.global_state: Optional[np.ndarray] = None  # shape: (30,)
        self.step_count: int = 0

        print("MarlEnvironment başlatıldı.")
        print(f"  DT         : {dt_path}")
        print(f"  Reward cfg : {reward_cfg_path}")
        print(f"  Max steps  : {MAX_STEPS}")
        print()
        self._print_obs_space()

    # ── Yardımcı: Observation Space Bilgisi ──────────────────────────────────

    def _print_obs_space(self) -> None:
        print("  Observation Space (partial observability):")
        print(f"  {'Agent':<12} {'Gözlemlediği Servisler':<35} {'Obs Boyutu':>10}")
        print(f"  {'-'*60}")
        for svc in SERVICES:
            neighbors = AGENT_NEIGHBORS[svc]
            obs_list  = [svc] + neighbors
            dim       = self.obs_dims[svc]
            print(f"  {svc:<12} {str(obs_list):<35} {dim:>10}")

    # ── Partial Observation Çıkartma ─────────────────────────────────────────

    def _get_obs(self, global_state: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Global state'ten (30,) her agent için partial observation çıkart.

        Her agent: kendi 5 metriği + komşularının 5'er metriği
        Örnek → cart: [cart_5] + [catalogue_5] + [payment_5] = 15 boyut

        Returns:
            Dict: {servis_adı: np.ndarray(obs_dim,)}
        """
        observations = {}
        for svc in SERVICES:
            # Kendi metriklerini al
            start = SERVICE_IDX[svc]
            own_metrics = global_state[start: start + len(METRICS)]

            # Komşu metriklerini al
            neighbor_metrics = []
            for neighbor in AGENT_NEIGHBORS[svc]:
                n_start = SERVICE_IDX[neighbor]
                neighbor_metrics.append(
                    global_state[n_start: n_start + len(METRICS)]
                )

            # Birleştir: [kendi | komşu1 | komşu2 | ...]
            if neighbor_metrics:
                observations[svc] = np.concatenate(
                    [own_metrics] + neighbor_metrics
                ).astype(np.float32)
            else:
                observations[svc] = own_metrics.astype(np.float32)

        return observations

    # ── Reward Hesabı ─────────────────────────────────────────────────────────

    def _compute_rewards(
        self,
        global_state: np.ndarray,
        actions_internal: Dict[str, int],
    ) -> Dict[str, float]:
        """
        Global state'e bakarak her agent için reward hesapla.

        Formül (sar2.py ile tutarlı):
            servis_katkı = tanh(1 - latency / p50) * weight
            pod_ceza     = -0.05 * toplam_pod

        Tüm agentlar aynı global reward'ı alır.
        (Gelecekte per-agent reward'a genişletilebilir.)

        NOT: payment reward'a dahil değil (Option C — sar2.py kararı).

        Returns:
            Dict: {servis_adı: float}  — tüm değerler eşit (global reward)
        """
        reward = 0.0
        total_pods = 0

        for i, svc in enumerate(SERVICES):
            cfg = self.reward_cfg[svc]
            lat_idx = SERVICE_IDX[svc] + 4  # latency, 5. metrik (indeks 4)

            latency = global_state[lat_idx]

            # Pod sayısını topla (ceza için)
            pod_idx = SERVICE_IDX[svc] + 0  # num_pods, 1. metrik (indeks 0)
            total_pods += global_state[pod_idx]

            # payment reward'a dahil değil
            if not cfg.get('in_reward', True):
                continue

            # Sıfır latency = ölçüm yok → sinyal yok
            if latency <= 0:
                continue

            # tanh-compressed reward (sar2.py formülü)
            p50    = cfg['p50']
            weight = cfg['weight']

            # DİKKAT: global_state normalize edilmiş [0,1]
            # p50 ise sar2.py'de ms cinsinden saklanıyor
            # → scaler ile denormalize etmek gerekiyor (aşağıda açıklama var)
            # Şimdilik normalize state üzerinden göreceli hesap yapıyoruz.
            # MARL eğitiminde reward_config'deki p50'ler normalize edilmiş
            # karşılıklarıyla değiştirilmeli — bunu _normalize_p50() ile halledelim.
            p50_norm = self._get_normalized_p50(svc)
            reward += np.tanh(1.0 - latency / p50_norm) * weight

        # Pod ceza: toplam pod × 0.05
        # (normalize pod değerleri üzerinden orantılı)
        pod_cost = self.reward_cfg.get('cart', {}).get('pod_cost', 0.05)
        reward -= total_pods * 0.05

        # Tüm agentlar aynı global reward'ı alır
        return {svc: round(float(reward), 4) for svc in SERVICES}

    def _get_normalized_p50(self, svc: str) -> float:
        """
        reward_config.json'daki p50 (ms) değerini normalize edilmiş
        [0,1] ölçeğine çevir.

        Scaler'ın feature sırasını kullanarak doğru sütunu buluruz.
        """
        # Scaler feature sırası: tüm state_cols
        state_cols = [f'{s}_{m}' for s in SERVICES for m in METRICS]
        lat_col    = f'{svc}_latency'

        if lat_col not in state_cols:
            return 1.0  # fallback

        col_idx = state_cols.index(lat_col)

        # MinMaxScaler: x_norm = (x - min) / (max - min)
        p50_raw   = self.reward_cfg[svc]['p50']
        data_min  = self.scaler.data_min_[col_idx]
        data_max  = self.scaler.data_max_[col_idx]

        if data_max == data_min:
            return 0.5  # edge case

        p50_norm = (p50_raw - data_min) / (data_max - data_min)
        return float(np.clip(p50_norm, 0.01, 1.0))  # 0'a bölmeyi önle

    # ── reset() ───────────────────────────────────────────────────────────────

    def reset(
        self, initial_state: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Episode'u sıfırla.

        Parametreler:
            initial_state : (30,) normalize global state.
                            None verilirse SAR dataset'ten rastgele örnek alınır.
                            (Dataset'i dışarıdan inject etmek için kullanılır.)

        Returns:
            observations : Dict[str, np.ndarray]  — her agent için partial obs
        """
        if initial_state is not None:
            assert initial_state.shape == (30,), \
                f"initial_state boyutu (30,) olmalı, aldı: {initial_state.shape}"
            self.global_state = initial_state.astype(np.float32).copy()
        else:
            # Placeholder: gerçek kullanımda SAR dataset'ten inject edilir
            # Tüm değerler 0.5 (normalize orta nokta)
            self.global_state = np.full(30, 0.5, dtype=np.float32)

        self.step_count = 0
        return self._get_obs(self.global_state)

    # ── step() ────────────────────────────────────────────────────────────────

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict]:
        """
        Tüm agentların aksiyonlarını al, DT ile sonraki state'i tahmin et.

        Parametreler:
            actions : Dict[str, int]
                      Her servis için aksiyon: {0: scale_down, 1: keep, 2: scale_up}

        Returns:
            observations : Dict[str, np.ndarray]  — yeni partial obs
            rewards      : Dict[str, float]        — her agent için reward
            done         : bool                    — episode bitti mi?
            info         : Dict                    — debug bilgisi
        """
        assert self.global_state is not None, "reset() çağrılmadı!"
        assert set(actions.keys()) == set(SERVICES), \
            f"Tüm servisler için aksiyon gerekli. Eksik: {set(SERVICES) - set(actions.keys())}"

        # 1. Aksiyon vektörünü oluştur: {0,1,2} → {-1,0,+1}
        action_vector = np.array(
            [ACTION_MAP[actions[svc]] for svc in SERVICES],
            dtype=np.float32
        )  # shape: (6,)

        # 2. DT'ye besle: [state(30) | action(6)] → next_state(30)
        state_action = np.concatenate([self.global_state, action_vector])
        with torch.no_grad():
            inp        = torch.tensor(state_action, dtype=torch.float32).unsqueeze(0)
            self.dt.eval()
            next_state = self.dt(inp).squeeze(0).numpy()  # (30,)

        # 3. Reward hesapla (next_state üzerinden)
        rewards = self._compute_rewards(next_state, actions)

        # 4. State'i güncelle
        self.global_state = next_state
        self.step_count  += 1

        # 5. Episode bitiş koşulu
        done = self.step_count >= MAX_STEPS

        # 6. Partial observations döndür
        observations = self._get_obs(next_state)

        # 7. Debug bilgisi
        info = {
            'step':          self.step_count,
            'global_state':  next_state.copy(),
            'action_vector': action_vector,           # internal: -1/0/+1 (DT'ye giden)
            'actions':       actions,                 # discrete:  0/1/2  (agent çıktısı)
            'global_reward': list(rewards.values())[0],
        }

        return observations, rewards, done, info

    # ── Yardımcı: Bilgi Yazdır ───────────────────────────────────────────────

    def obs_dim(self, svc: str) -> int:
        """Belirtilen servisin observation boyutunu döndür."""
        return self.obs_dims[svc]

    def action_dim(self) -> int:
        """Aksiyon uzayı boyutu (tüm agentlar için aynı)."""
        return N_ACTIONS

    def print_step_summary(self, step_info: Dict) -> None:
        """step() çıktısını okunabilir formatta yazdır (debug için)."""
        # action_vector internal (-1/0/+1) — okunabilir etiketlerle göster
        labels = {-1: 'down', 0: 'keep', 1: 'up'}
        action_display = {
            s: labels[int(a)]
            for s, a in zip(SERVICES, step_info['action_vector'])
        }
        print(f"\n  [Step {step_info['step']}]")
        print(f"  Actions  : {action_display}")
        print(f"  Reward   : {step_info['global_reward']:.4f}")

        gs = step_info['global_state']
        print(f"\n  {'Servis':<12} {'pods':>6} {'cpu':>8} {'mem':>8} {'rps':>8} {'lat':>8}")
        print(f"  {'-'*55}")
        for svc in SERVICES:
            start = SERVICE_IDX[svc]
            vals  = gs[start: start + len(METRICS)]
            print(f"  {svc:<12} " + " ".join(f"{v:>8.4f}" for v in vals))


# ─── Hızlı Doğrulama (standalone çalıştırma) ────────────────────────────────

if __name__ == '__main__':
    print("=" * 65)
    print("MarlEnvironment — Hızlı Doğrulama")
    print("=" * 65)

    # Ortamı başlat
    env = MarlEnvironment(
        dt_path='digital_twin_best.pth',
        reward_cfg_path='reward_config.json',
        scaler_path='scaler.pkl',
    )

    print("\n" + "=" * 65)
    print("reset() testi")
    print("=" * 65)

    # SAR dataset'ten gerçek bir başlangıç state'i kullan
    import pandas as pd
    df = pd.read_csv('rl_dataset_sar_normalized.csv')
    state_cols = [f'{s}_{m}' for s in SERVICES for m in METRICS]
    initial    = df.iloc[100][state_cols].values.astype(np.float32)

    obs = env.reset(initial_state=initial)

    print(f"\n  global_state boyutu : {env.global_state.shape}")
    for svc in SERVICES:
        print(f"  {svc:<12} obs boyutu: {obs[svc].shape}  "
              f"(beklenen: {env.obs_dim(svc)})")

    print("\n" + "=" * 65)
    print("step() testi — 3 adım, tüm agentlar 'keep' (aksiyon 1)")
    print("=" * 65)

    for i in range(3):
        actions = {svc: 1 for svc in SERVICES}  # hepsi keep
        obs, rewards, done, info = env.step(actions)
        env.print_step_summary(info)
        if done:
            print("  → Episode bitti!")
            break

    print("\n" + "=" * 65)
    print("step() testi — aksiyon çeşitliliği")
    print("=" * 65)

    obs = env.reset(initial_state=initial)
    test_actions = {
        'cart':      2,   # scale up
        'catalogue': 1,   # keep
        'payment':   1,   # keep
        'shipping':  0,   # scale down
        'ratings':   1,   # keep
        'user':      2,   # scale up
    }
    obs, rewards, done, info = env.step(test_actions)
    env.print_step_summary(info)

    print("\n" + "=" * 65)
    print("Tam bir Episode (10 adım)")
    print("=" * 65)

    import random
    obs      = env.reset(initial_state=initial)
    ep_reward = 0.0
    step_n    = 0

    while True:
        # Rastgele aksiyonlar (baseline testi)
        actions = {svc: random.choice([0, 1, 2]) for svc in SERVICES}
        obs, rewards, done, info = env.step(actions)
        ep_reward += info['global_reward']
        step_n    += 1
        if done:
            break

    print(f"\n  Episode tamamlandı.")
    print(f"  Toplam adım     : {step_n}")
    print(f"  Toplam reward   : {ep_reward:.4f}")
    print(f"  Ortalama reward : {ep_reward / step_n:.4f}")

    print("\n  ✓ MarlEnvironment doğrulama başarılı!")
    print("=" * 65)