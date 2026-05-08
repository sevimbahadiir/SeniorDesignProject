"""
dqn_agent.py
------------
Independent DQN (IDQN) Agent

Her mikroservis için bağımsız bir DQN agent.
IPPO ile aynı ortamda çalışır, evaluate.py'a kolayca entegre edilir.

IPPO'dan farkları:
    - Actor-Critic yerine tek Q-Network
    - On-policy yerine off-policy (Replay Buffer)
    - Episode sonu değil her adımda update
    - Stochastic policy yerine ε-greedy
    - GAE yerine Bellman hedefi: Q_target = r + γ·max Q(s')
    - Target network: training stabilitesi için periyodik kopyalama

Bileşenler:
    QNetwork      : obs → Q değerleri (her aksiyon için)
    ReplayBuffer  : (obs, action, reward, next_obs, done) geçişleri
    DQNAgent      : ε-greedy seçim, Bellman update, target sync
    build_dqn_pool: environment'a göre agent pool oluşturma

Hiperparametreler:
    HIDDEN         = [128, 64]
    LR             = 1e-4
    GAMMA          = 0.99
    BUFFER_SIZE    = 10_000   replay buffer kapasitesi
    BATCH_SIZE     = 64       her update'te kaç örnek
    TARGET_SYNC    = 50       kaç adımda bir target network güncellenir
    EPS_START      = 1.0      başlangıç epsilon (tam keşif)
    EPS_END        = 0.05     minimum epsilon
    EPS_DECAY      = 2000     kaç adımda EPS_END'e ulaşır
    WARMUP_STEPS   = 500      buffer dolmadan update yapma
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Dict, Tuple, Optional, List

# ─── Hiperparametreler ────────────────────────────────────────────────────────

HIDDEN       = [128, 64]
LR           = 1e-4
GAMMA        = 0.99
BUFFER_SIZE  = 10_000
BATCH_SIZE   = 64
TARGET_SYNC  = 50       # adım sayısı (episode değil)
EPS_START    = 1.0
EPS_END      = 0.05
EPS_DECAY    = 2000
WARMUP_STEPS = 500


# ─── Q-Network ───────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    Q değer ağı: obs → [Q(s,a0), Q(s,a1), Q(s,a2)]

    Mimari: obs_dim → 128 → 64 → n_actions
    Aktivasyon: ReLU (Q değerleri sınırsız olabilir, Tanh kullanmıyoruz)
    Çıktı: ham Q değerleri (softmax yok — argmax ile aksiyon seçilir)
    """
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        layers = []
        prev   = obs_dim
        for h in HIDDEN:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)

        # Orthogonal init — kararlı başlangıç
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs → Q değerleri  (shape: [batch, n_actions])"""
        return self.net(obs)


# ─── Replay Buffer ────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Experience Replay Buffer.

    DQN'in on-policy PPO'dan temel farkı:
    Geçmiş deneyimleri saklayıp rastgele örnekleyerek:
      1. Veri verimliliği artar (her transition birden fazla kullanılır)
      2. Ardışık geçişler arasındaki temporal korelasyon kırılır
      3. Daha kararlı gradient tahmini sağlanır

    Kapasite dolunca eski geçişlerin üzerine yazılır (deque).
    """
    def __init__(self, capacity: int = BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> None:
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int) -> Tuple:
        """Rastgele batch örnekle."""
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs,      dtype=np.float32),
            np.array(actions,  dtype=np.int64),
            np.array(rewards,  dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones,    dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self) -> bool:
        """Warmup tamamlandı mı?"""
        return len(self.buffer) >= WARMUP_STEPS


# ─── Epsilon Scheduler ────────────────────────────────────────────────────────

class EpsilonScheduler:
    """
    Lineer epsilon decay: EPS_START → EPS_END, EPS_DECAY adımda.

    Başta tam keşif (ε=1.0), zamanla sömürüye geçiş (ε=0.05).
    Neden lineer? Exponential decay de kullanılabilir ama
    lineer daha öngörülü — ne zaman exploiting'e geçeceğimizi biliyoruz.
    """
    def __init__(self):
        self.step_count = 0

    def value(self) -> float:
        progress = min(self.step_count / EPS_DECAY, 1.0)
        return EPS_START + progress * (EPS_END - EPS_START)

    def increment(self) -> None:
        self.step_count += 1


# ─── DQN Agent ───────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Independent DQN Agent — tek bir mikroservis için.

    MarlEnvironment ile kullanım:
        agent = DQNAgent(obs_dim=15, n_actions=3, service='cart')

        # Her step:
        action = agent.select_action(obs)
        agent.buffer.push(obs, action, reward, next_obs, done)
        loss   = agent.update()   # her adımda (warmup sonrası)

    Parametreler:
        obs_dim   : partial observation boyutu (10 veya 15)
        n_actions : aksiyon uzayı boyutu (3: down/keep/up)
        service   : servis adı (loglama için)
        device    : 'cpu' veya 'cuda'
    """

    def __init__(
        self,
        obs_dim:   int,
        n_actions: int = 3,
        service:   str = 'unknown',
        device:    str = 'cpu',
    ):
        self.service   = service
        self.obs_dim   = obs_dim
        self.n_actions = n_actions
        self.device    = torch.device(device)

        # Online network: her adımda güncellenir
        self.q_net     = QNetwork(obs_dim, n_actions).to(self.device)

        # Target network: her TARGET_SYNC adımda senkronize edilir
        # Neden? Online net sürekli değişirken stable bir hedef sağlar
        self.q_target  = QNetwork(obs_dim, n_actions).to(self.device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.q_target.eval()   # target network hiç train moduna geçmez

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.buffer    = ReplayBuffer(BUFFER_SIZE)
        self.eps_sched = EpsilonScheduler()

        self.total_steps   = 0
        self.total_updates = 0

    # ── Aksiyon Seçimi ────────────────────────────────────────────────────────

    def select_action(self, obs: np.ndarray, training: bool = True) -> int:
        """
        ε-greedy aksiyon seçimi.

        Eğitimde:
            ε olasılıkla rastgele (keşif)
            1-ε olasılıkla argmax Q(s,a) (sömürü)

        Değerlendirmede (training=False):
            Her zaman argmax Q(s,a) — deterministic

        Parametreler:
            obs      : (obs_dim,) partial observation
            training : True → ε-greedy, False → greedy

        Returns:
            action : int {0, 1, 2}
        """
        if training and random.random() < self.eps_sched.value():
            return random.randint(0, self.n_actions - 1)

        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.q_net(obs_t)
        return int(q_vals.argmax(dim=-1).item())

    # ── Bellman Update ────────────────────────────────────────────────────────

    def update(self) -> Optional[float]:
        """
        Replay buffer'dan batch örnekle ve Q-network'ü güncelle.

        Bellman hedefi:
            Q_target(s,a) = r + γ · max_a' Q_target(s', a')
                                         ^^^^^^^^^^^^^^^^
                                         target network kullanılır

        Loss: MSE(Q_online(s,a), Q_target(s,a))

        Warmup tamamlanmamışsa None döner (buffer yeterince dolmamış).

        Returns:
            loss : float veya None
        """
        if not self.buffer.is_ready():
            return None

        # Batch örnekle
        obs, actions, rewards, next_obs, dones = self.buffer.sample(BATCH_SIZE)

        obs_t      = torch.tensor(obs,     dtype=torch.float32).to(self.device)
        actions_t  = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards_t  = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_obs_t = torch.tensor(next_obs,dtype=torch.float32).to(self.device)
        dones_t    = torch.tensor(dones,   dtype=torch.float32).to(self.device)

        # Mevcut Q değerleri: Q_online(s, a) — sadece seçilen aksiyonun Q'su
        q_current = self.q_net(obs_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Hedef Q değerleri: r + γ · max Q_target(s', ·)
        with torch.no_grad():
            q_next   = self.q_target(next_obs_t).max(dim=1).values
            q_target = rewards_t + GAMMA * q_next * (1.0 - dones_t)

        # Huber loss (MSE'ye göre outlier'lara daha robust)
        loss = nn.functional.smooth_l1_loss(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping — DQN'de önemli, Q değerleri patlamasın
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.total_updates += 1

        # Target network senkronizasyonu
        if self.total_updates % TARGET_SYNC == 0:
            self.q_target.load_state_dict(self.q_net.state_dict())

        return float(loss.item())

    def step(self) -> None:
        """Epsilon scheduler'ı ilerlet (her adımda çağrılır)."""
        self.eps_sched.increment()
        self.total_steps += 1

    # ── Model Kaydet / Yükle ─────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save({
            'q_net':       self.q_net.state_dict(),
            'q_target':    self.q_target.state_dict(),
            'optimizer':   self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'obs_dim':     self.obs_dim,
            'n_actions':   self.n_actions,
            'service':     self.service,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(ckpt['q_net'])
        self.q_target.load_state_dict(ckpt['q_target'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.total_steps = ckpt.get('total_steps', 0)

    def __repr__(self) -> str:
        params = sum(p.numel() for p in self.q_net.parameters())
        return (f"DQNAgent(service={self.service}, "
                f"obs_dim={self.obs_dim}, params={params:,}, "
                f"eps={self.eps_sched.value():.3f})")


# ─── Agent Pool ──────────────────────────────────────────────────────────────

def build_dqn_pool(
    env,
    device: str = 'cpu',
) -> Dict[str, DQNAgent]:
    """
    MarlEnvironment'a göre her servis için bir DQNAgent oluştur.
    """
    from marl_env import SERVICES
    agents = {}
    for svc in SERVICES:
        agents[svc] = DQNAgent(
            obs_dim   = env.obs_dim(svc),
            n_actions = env.action_dim(),
            service   = svc,
            device    = device,
        )
    return agents


# ─── Hızlı Doğrulama ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    import pandas as pd
    from marl_env import MarlEnvironment, SERVICES

    print("=" * 65)
    print("DQNAgent — Hızlı Doğrulama")
    print("=" * 65)

    env    = MarlEnvironment()
    agents = build_dqn_pool(env)

    print("\nAgent Pool:")
    print(f"  {'Servis':<12} {'Obs Dim':>8} {'Q-Net Params':>14} {'Buffer':>8}")
    print(f"  {'-'*46}")
    for svc, agent in agents.items():
        p = sum(p.numel() for p in agent.q_net.parameters())
        print(f"  {svc:<12} {agent.obs_dim:>8} {p:>14,} {len(agent.buffer):>8}")

    print("\n" + "=" * 65)
    print("Epsilon decay testi")
    print("=" * 65)
    test_agent = agents['cart']
    for step in [0, 500, 1000, 2000, 3000]:
        test_agent.eps_sched.step_count = step
        print(f"  Adım {step:>5}: ε = {test_agent.eps_sched.value():.4f}")

    print("\n" + "=" * 65)
    print("select_action() + buffer + update() testi")
    print("=" * 65)

    df         = pd.read_csv('rl_dataset_sar_normalized.csv')
    state_cols = [f'{s}_{m}' for s in SERVICES
                  for m in ['num_pods','cpu_usage','mem_usage','request_rate','latency']]
    initial    = df.iloc[100][state_cols].values.astype(np.float32)

    # Warmup için buffer doldur
    print(f"\n  Warmup: {WARMUP_STEPS} geçiş buffer'a dolduruluyor...")
    obs = env.reset(initial_state=initial)
    warmup_steps = 0
    while warmup_steps < WARMUP_STEPS:
        actions = {svc: agents[svc].select_action(obs[svc]) for svc in SERVICES}
        next_obs, rewards, done, info = env.step(actions)
        for svc in SERVICES:
            agents[svc].buffer.push(
                obs[svc], actions[svc],
                rewards[svc], next_obs[svc], done
            )
            agents[svc].step()
        obs = next_obs if not done else env.reset(initial_state=initial)
        warmup_steps += 1

    print(f"  Buffer boyutu: {len(agents['cart'].buffer)}")
    print(f"  Epsilon şu an: {agents['cart'].eps_sched.value():.4f}")

    # Birkaç update
    print(f"\n  İlk 5 update sonuçları (cart agent):")
    for i in range(5):
        loss = agents['cart'].update()
        print(f"    Update {i+1}: loss = {loss:.6f}")

    # Target sync kontrolü
    print(f"\n  Target sync: her {TARGET_SYNC} update'te bir")
    print(f"  Şu anki update sayısı: {agents['cart'].total_updates}")

    # Save/Load
    agents['cart'].save('cart_dqn_test.pth')
    agents['cart'].load('cart_dqn_test.pth')
    print(f"\n  save/load ✓")

    print("\n  ✓ DQNAgent doğrulama başarılı!")
    print("=" * 65)
