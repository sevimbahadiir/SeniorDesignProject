"""
ippo_agent.py
-------------
Adım 2: Independent PPO (IPPO) Agent

Her mikroservis için bağımsız bir PPO agent.
"Independent" = her agent kendi obs'unu görür, kendi policy'sini günceller,
diğer agentların parametrelerine erişmez.

IPPO bileşenleri:
    Actor   : obs → aksiyon olasılıkları (π_θ)  — Categorical dağılım
    Critic  : obs → durum değeri (V_φ)           — skaler
    Buffer  : bir episode boyunca geçişleri depolar
    GAE     : Generalized Advantage Estimation   — avantaj hesabı
    Update  : PPO-Clip loss ile actor + critic güncelleme

Neden bu tasarım?
    - Actor ve Critic ayrı ağlar: gradyanlar birbirini bozmaz
    - GAE (λ): bias-variance trade-off — saf TD(0) değil, saf MC değil
    - PPO-Clip: trust region olmadan büyük policy değişimlerini önler
    - Entropy bonus: keşfi teşvik eder, erken yakınsama riski azalır

Hiperparametreler (makale değerleriyle uyumlu):
    HIDDEN      = [128, 64]   — obs boyutu küçük (10-15), derin ağa gerek yok
    LR_ACTOR    = 3e-4
    LR_CRITIC   = 1e-3        — critic daha hızlı öğrenir
    GAMMA       = 0.99        — uzun vadeli ödül (10-adım episode)
    GAE_LAMBDA  = 0.95        — standart PPO değeri
    CLIP_EPS    = 0.2         — PPO clip parametresi
    ENTROPY_C   = 0.01        — entropy bonus katsayısı
    VALUE_C     = 0.5         — critic loss katsayısı
    K_EPOCHS    = 4           — her update'te kaç epoch
    MAX_GRAD    = 0.5         — gradient clipping
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Tuple, Dict, Optional


# ─── Hiperparametreler ────────────────────────────────────────────────────────

HIDDEN     = [128, 64]
LR_ACTOR   = 1e-4
LR_CRITIC  = 5e-4
GAMMA      = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS   = 0.2
ENTROPY_C  = 0.02
VALUE_C    = 0.5
K_EPOCHS   = 8
MAX_GRAD   = 0.5


# ─── Ağ Mimarileri ───────────────────────────────────────────────────────────

class ActorNet(nn.Module):
    """
    Policy ağı: obs → aksiyon olasılıkları (logits).

    Çıktı: ham logit'ler (softmax Categorical dağılımda yapılır).
    Neden logit? Numerical stabilite — softmax + log yerine log_softmax.

    Mimari: obs_dim → 128 → 64 → n_actions
    """
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in HIDDEN:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)

        # Ağırlık başlatma: küçük son katman → başlangıçta uniform policy
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs → logits  (shape: [batch, n_actions])"""
        return self.net(obs)

    def get_dist(self, obs: torch.Tensor) -> Categorical:
        """obs → Categorical dağılım (aksiyon seçimi ve log_prob için)"""
        logits = self.forward(obs)
        return Categorical(logits=logits)


class CriticNet(nn.Module):
    """
    Değer ağı: obs → V(obs) (skaler).

    Actor ile aynı obs'u görür ama farklı parametreler kullanır.
    Neden ayrı? Critic daha hızlı öğrenebilir (lr_critic > lr_actor),
    actor gradyanları critic'i bozmaz.

    Mimari: obs_dim → 128 → 64 → 1
    """
    def __init__(self, obs_dim: int):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in HIDDEN:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

        # Ağırlık başlatma: değer tahminleri başlangıçta sıfıra yakın
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs → V(obs)  (shape: [batch, 1])"""
        return self.net(obs)


# ─── Rollout Buffer ──────────────────────────────────────────────────────────

class RolloutBuffer:
    """
    Bir episode boyunca geçişleri depolar.

    step() çağrıldıkça dolup, update() sonrası temizlenir.
    Her geçiş: (obs, action, log_prob, reward, done, value)

    NOT: normalize edilmiş reward saklanmaz — compute_gae() içinde yapılır.
    """
    def __init__(self):
        self.obs:      List[np.ndarray] = []
        self.actions:  List[int]        = []
        self.log_probs: List[float]     = []
        self.rewards:  List[float]      = []
        self.dones:    List[bool]       = []
        self.values:   List[float]      = []

    def push(
        self,
        obs:      np.ndarray,
        action:   int,
        log_prob: float,
        reward:   float,
        done:     bool,
        value:    float,
    ) -> None:
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self) -> None:
        self.__init__()

    def __len__(self) -> int:
        return len(self.rewards)


# ─── GAE Hesabı ──────────────────────────────────────────────────────────────

def compute_gae(
    rewards:    List[float],
    values:     List[float],
    dones:      List[bool],
    last_value: float = 0.0,
    gamma:      float = GAMMA,
    lam:        float = GAE_LAMBDA,
) -> Tuple[List[float], List[float]]:
    """
    Generalized Advantage Estimation (GAE-λ).

    Formül:
        δ_t     = r_t + γ·V(s_{t+1}) - V(s_t)       (TD hatası)
        A_t     = δ_t + γλ·δ_{t+1} + (γλ)²·δ_{t+2} + ...

    Neden GAE?
        λ=0 → saf TD(0): düşük varyans, yüksek bias
        λ=1 → saf MC  : yüksek varyans, düşük bias
        λ=0.95 → ikisi arası en iyi trade-off (standart PPO)

    Returns:
        advantages : GAE avantajları  (A_t)
        returns    : hedef değerler   (V_target = A_t + V(s_t))
    """
    T          = len(rewards)
    advantages = [0.0] * T
    returns    = [0.0] * T
    gae        = 0.0

    # Geriye doğru hesapla (t=T-1'den t=0'a)
    for t in reversed(range(T)):
        # Son adımda bir sonraki değer = last_value (bootstrap)
        next_val  = last_value if t == T - 1 else values[t + 1]
        next_done = 1.0 - float(dones[t])

        # TD hatası
        delta     = rewards[t] + gamma * next_val * next_done - values[t]

        # GAE birikimi
        gae       = delta + gamma * lam * next_done * gae

        advantages[t] = gae
        returns[t]    = gae + values[t]   # V_target

    return advantages, returns


# ─── IPPO Agent ──────────────────────────────────────────────────────────────

class IPPOAgent:
    """
    Independent PPO Agent — tek bir mikroservis için.

    MarlEnvironment ile kullanım:
        agent = IPPOAgent(obs_dim=15, n_actions=3, service='cart')

        # Her step:
        action, log_prob, value = agent.select_action(obs)
        agent.buffer.push(obs, action, log_prob, reward, done, value)

        # Episode sonu:
        metrics = agent.update()
        agent.buffer.clear()

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

        # Ağlar
        self.actor  = ActorNet(obs_dim, n_actions).to(self.device)
        self.critic = CriticNet(obs_dim).to(self.device)

        # Ayrı optimizer — farklı öğrenme hızı
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=LR_ACTOR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # İstatistikler (loglama için)
        self.total_updates = 0

    # ── Aksiyon Seçimi ────────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(
        self, obs: np.ndarray
    ) -> Tuple[int, float, float]:
        """
        Policy'den aksiyon örnekle.

        Parametreler:
            obs : (obs_dim,) normalize edilmiş partial observation

        Returns:
            action   : int    — seçilen aksiyon {0, 1, 2}
            log_prob : float  — seçilen aksiyonun log olasılığı
            value    : float  — critic'in V(obs) tahmini
        """
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Actor: aksiyon dağılımı
        dist     = self.actor.get_dist(obs_t)
        action_t = dist.sample()
        log_prob = dist.log_prob(action_t)

        # Critic: değer tahmini
        value = self.critic(obs_t).squeeze()

        return (
            int(action_t.item()),
            float(log_prob.item()),
            float(value.item()),
        )

    @torch.no_grad()
    def get_value(self, obs: np.ndarray) -> float:
        """Son obs'un değerini döndür (bootstrap için)."""
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        return float(self.critic(obs_t).squeeze().item())

    # ── Policy Güncelleme ─────────────────────────────────────────────────────

    def update(self, last_obs: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Buffer'daki geçişlerle actor ve critic'i güncelle.

        PPO-Clip hedef fonksiyonu:
            L_CLIP = E[min(r_t·A_t,  clip(r_t, 1-ε, 1+ε)·A_t)]
            r_t    = π_θ(a|s) / π_θ_old(a|s)   (olasılık oranı)

            L_VALUE = (V(s) - V_target)²
            L_ENT   = H[π(·|s)]                  (entropi bonusu)

            L_total = -L_CLIP + c_v·L_VALUE - c_e·L_ENT

        K_EPOCHS kez aynı veriyle güncelle (PPO'nun temel avantajı).

        Returns:
            metrics : {'actor_loss', 'critic_loss', 'entropy', 'approx_kl'}
        """
        if len(self.buffer) == 0:
            return {}

        # Bootstrap: son obs'un değeri (episode bitmemişse)
        last_value = 0.0
        if last_obs is not None:
            last_value = self.get_value(last_obs)

        # GAE ve returns hesapla
        advantages, returns = compute_gae(
            self.buffer.rewards,
            self.buffer.values,
            self.buffer.dones,
            last_value=last_value,
        )

        # Tensor'lara çevir
        obs_t      = torch.tensor(np.array(self.buffer.obs),
                                  dtype=torch.float32).to(self.device)
        actions_t  = torch.tensor(self.buffer.actions,
                                  dtype=torch.long).to(self.device)
        old_lp_t   = torch.tensor(self.buffer.log_probs,
                                  dtype=torch.float32).to(self.device)
        adv_t      = torch.tensor(advantages,
                                  dtype=torch.float32).to(self.device)
        returns_t  = torch.tensor(returns,
                                  dtype=torch.float32).to(self.device)

        # Avantaj normalizasyonu (training stabilitesi)
        # Sadece batch boyutu > 1 ise normalize et
        if adv_t.shape[0] > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # K_EPOCHS kez güncelle
        metrics_acc = {
            'actor_loss':  0.0,
            'critic_loss': 0.0,
            'entropy':     0.0,
            'approx_kl':   0.0,
        }

        for epoch in range(K_EPOCHS):

            # ── Actor güncelle ──
            dist     = self.actor.get_dist(obs_t)
            new_lp   = dist.log_prob(actions_t)
            entropy  = dist.entropy().mean()

            # Olasılık oranı: π_new / π_old (log space'de fark)
            ratio    = torch.exp(new_lp - old_lp_t)

            # PPO-Clip
            surr1    = ratio * adv_t
            surr2    = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_t
            actor_loss = -torch.min(surr1, surr2).mean() - ENTROPY_C * entropy

            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD)
            self.actor_opt.step()

            # ── Critic güncelle ──
            values      = self.critic(obs_t).squeeze()
            critic_loss = VALUE_C * nn.functional.mse_loss(values, returns_t)

            self.critic_opt.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD)
            self.critic_opt.step()

            # Metrik biriktir
            with torch.no_grad():
                approx_kl = ((old_lp_t - new_lp) ** 2).mean() * 0.5

            metrics_acc['actor_loss']  += actor_loss.item()
            metrics_acc['critic_loss'] += critic_loss.item()
            metrics_acc['entropy']     += entropy.item()
            metrics_acc['approx_kl']   += approx_kl.item()

        # Epoch ortalaması
        for k in metrics_acc:
            metrics_acc[k] /= K_EPOCHS

        self.total_updates += 1
        return metrics_acc

    # ── Model Kaydet / Yükle ─────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Agent ağırlıklarını kaydet."""
        torch.save({
            'actor':       self.actor.state_dict(),
            'critic':      self.critic.state_dict(),
            'actor_opt':   self.actor_opt.state_dict(),
            'critic_opt':  self.critic_opt.state_dict(),
            'obs_dim':     self.obs_dim,
            'n_actions':   self.n_actions,
            'service':     self.service,
            'total_updates': self.total_updates,
        }, path)

    def load(self, path: str) -> None:
        """Kaydedilmiş ağırlıkları yükle."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.actor_opt.load_state_dict(ckpt['actor_opt'])
        self.critic_opt.load_state_dict(ckpt['critic_opt'])
        self.total_updates = ckpt.get('total_updates', 0)

    # ── Bilgi Yazdır ─────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        actor_p  = sum(p.numel() for p in self.actor.parameters())
        critic_p = sum(p.numel() for p in self.critic.parameters())
        return (
            f"IPPOAgent(service={self.service}, "
            f"obs_dim={self.obs_dim}, "
            f"actor_params={actor_p:,}, "
            f"critic_params={critic_p:,})"
        )


# ─── Agent Pool ──────────────────────────────────────────────────────────────

def build_agent_pool(
    env,
    device: str = 'cpu',
) -> Dict[str, IPPOAgent]:
    """
    MarlEnvironment'a göre her servis için bir IPPOAgent oluştur.

    Returns:
        Dict[service_name → IPPOAgent]
    """
    from marl_env import SERVICES

    agents = {}
    for svc in SERVICES:
        obs_dim = env.obs_dim(svc)
        agents[svc] = IPPOAgent(
            obs_dim=obs_dim,
            n_actions=env.action_dim(),
            service=svc,
            device=device,
        )
    return agents


# ─── Hızlı Doğrulama ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    import pandas as pd
    from marl_env import MarlEnvironment, SERVICES

    print("=" * 65)
    print("IPPOAgent — Hızlı Doğrulama")
    print("=" * 65)

    # ── 1. Ortam ve Agent Pool ──
    env    = MarlEnvironment()
    agents = build_agent_pool(env)

    print("\nAgent Pool:")
    print(f"  {'Servis':<12} {'Obs Dim':>8} {'Actor Params':>14} {'Critic Params':>14}")
    print(f"  {'-'*52}")
    for svc, agent in agents.items():
        ap = sum(p.numel() for p in agent.actor.parameters())
        cp = sum(p.numel() for p in agent.critic.parameters())
        print(f"  {svc:<12} {agent.obs_dim:>8} {ap:>14,} {cp:>14,}")

    # ── 2. select_action testi ──
    print("\n" + "=" * 65)
    print("select_action() testi")
    print("=" * 65)

    df         = pd.read_csv('rl_dataset_sar_normalized.csv')
    state_cols = [f'{s}_{m}' for s in SERVICES
                  for m in ['num_pods','cpu_usage','mem_usage','request_rate','latency']]
    initial    = df.iloc[100][state_cols].values.astype('float32')

    obs = env.reset(initial_state=initial)

    print(f"\n  {'Servis':<12} {'Aksiyon':>8} {'LogProb':>10} {'Value':>10}")
    print(f"  {'-'*45}")
    for svc in SERVICES:
        action, log_prob, value = agents[svc].select_action(obs[svc])
        label = {0:'down', 1:'keep', 2:'up'}[action]
        print(f"  {svc:<12} {label:>8} {log_prob:>10.4f} {value:>10.4f}")

    # ── 3. Tam Episode + Update ──
    print("\n" + "=" * 65)
    print("Tam Episode → Buffer → update() testi")
    print("=" * 65)

    obs           = env.reset(initial_state=initial)
    episode_reward = 0.0
    last_obs      = obs.copy()

    while True:
        actions = {}
        for svc in SERVICES:
            action, log_prob, value = agents[svc].select_action(obs[svc])
            actions[svc] = action
            agents[svc].buffer.push(
                obs=obs[svc],
                action=action,
                log_prob=log_prob,
                reward=0.0,   # reward adımdan sonra doldurulur
                done=False,
                value=value,
            )

        obs, rewards, done, info = env.step(actions)
        episode_reward += info['global_reward']

        # Reward'ı buffer'ın son kaydına yaz
        for svc in SERVICES:
            agents[svc].buffer.rewards[-1] = rewards[svc]
            agents[svc].buffer.dones[-1]   = done

        last_obs = obs.copy()
        if done:
            break

    print(f"\n  Episode reward : {episode_reward:.4f}")
    print(f"  Buffer boyutu  : {len(agents['cart'].buffer)} adım")

    # ── 4. Update ──
    print("\n  update() sonuçları:")
    print(f"  {'Servis':<12} {'ActorLoss':>11} {'CriticLoss':>12} "
          f"{'Entropy':>9} {'ApproxKL':>10}")
    print(f"  {'-'*58}")

    for svc in SERVICES:
        metrics = agents[svc].update(last_obs=last_obs[svc])
        agents[svc].buffer.clear()
        print(f"  {svc:<12} {metrics['actor_loss']:>11.4f} "
              f"{metrics['critic_loss']:>12.4f} "
              f"{metrics['entropy']:>9.4f} "
              f"{metrics['approx_kl']:>10.6f}")

    # ── 5. Save / Load testi ──
    print("\n" + "=" * 65)
    print("save() / load() testi")
    print("=" * 65)

    agents['cart'].save('cart_agent_test.pth')
    agents['cart'].load('cart_agent_test.pth')
    print("  cart agent kaydedildi ve yüklendi ✓")

    print("\n  ✓ IPPOAgent doğrulama başarılı!")
    print("=" * 65)
