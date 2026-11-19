"""RSSM world model and CEM planner, matching the official PlaNet networks.

Most of the runtime is Python dispatch in two tight loops, not GPU arithmetic on
the small dense layers: 50 RSSM steps per update and 30,000 RSSM steps per
episode of planning. So those loops are written to be TorchScript-compatible,
avoid torch.distributions objects, and batch work outside the loop wherever
possible (the reward head runs once over the whole rollout).
"""
import warnings
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import Config

# torch.compile lacks reliable Windows CUDA support, and uv.lock pins this torch version.
warnings.filterwarnings("ignore", message=".*torch.jit.script.*", category=FutureWarning)


def preprocess(frames: Tensor, bit_depth: int) -> Tensor:
    """uint8 to float in [-0.5, 0.5), reduced to bit_depth bits, with dequantization noise."""
    bins = 2**bit_depth
    x = torch.floor(frames.float() / 2 ** (8 - bit_depth)) / bins
    return x + torch.rand_like(x) / bins - 0.5


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU(inplace=True),
            nn.Flatten(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, feature_size: int, embed_size: int):
        super().__init__()
        self.embed_size = embed_size
        self.fc = nn.Linear(feature_size, embed_size)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(embed_size, 128, 5, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 5, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 6, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 6, stride=2),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.net(self.fc(features).view(-1, self.embed_size, 1, 1))


class RSSM(nn.Module):
    def __init__(self, cfg: Config, action_dim: int):
        super().__init__()
        self.state_size = cfg.state_size
        self.belief_size = cfg.belief_size
        self.min_std = cfg.min_std
        self.pre_gru = nn.Linear(cfg.state_size + action_dim, cfg.hidden_size)
        self.gru = nn.GRUCell(cfg.hidden_size, cfg.belief_size)
        self.prior_hidden = nn.Linear(cfg.belief_size, cfg.hidden_size)
        self.prior_mean = nn.Linear(cfg.hidden_size, cfg.state_size)
        self.prior_std = nn.Linear(cfg.hidden_size, cfg.state_size)
        self.post_hidden = nn.Linear(cfg.belief_size + cfg.embed_size, cfg.hidden_size)
        self.post_mean = nn.Linear(cfg.hidden_size, cfg.state_size)
        self.post_std = nn.Linear(cfg.hidden_size, cfg.state_size)

    def advance(self, state: Tensor, action: Tensor, belief: Tensor) -> Tensor:
        return self.gru(F.relu(self.pre_gru(torch.cat([state, action], -1))), belief)

    def prior(self, belief: Tensor) -> Tuple[Tensor, Tensor]:
        hidden = F.relu(self.prior_hidden(belief))
        return self.prior_mean(hidden), F.softplus(self.prior_std(hidden)) + self.min_std

    def posterior(self, belief: Tensor, embed: Tensor) -> Tuple[Tensor, Tensor]:
        hidden = F.relu(self.post_hidden(torch.cat([belief, embed], -1)))
        return self.post_mean(hidden), F.softplus(self.post_std(hidden)) + self.min_std

    @torch.jit.export
    def observe(self, embed: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Closed-loop filtering over (batch, time). Returns tensors stacked on dim 1."""
        batch = embed.size(0)
        state = embed.new_zeros([batch, self.state_size])
        belief = embed.new_zeros([batch, self.belief_size])
        states, beliefs, prior_means, prior_stds, post_means, post_stds = [], [], [], [], [], []
        for t in range(embed.size(1)):
            belief = self.advance(state, actions[:, t], belief)
            prior_mean, prior_std = self.prior(belief)
            post_mean, post_std = self.posterior(belief, embed[:, t])
            state = post_mean + post_std * torch.randn_like(post_std)
            states.append(state)
            beliefs.append(belief)
            prior_means.append(prior_mean)
            prior_stds.append(prior_std)
            post_means.append(post_mean)
            post_stds.append(post_std)
        return (
            torch.stack(states, 1), torch.stack(beliefs, 1),
            torch.stack(prior_means, 1), torch.stack(prior_stds, 1),
            torch.stack(post_means, 1), torch.stack(post_stds, 1),
        )

    @torch.jit.export
    def filter_step(self, state: Tensor, action: Tensor, belief: Tensor, embed: Tensor) -> Tuple[Tensor, Tensor]:
        """One posterior update while acting in the environment."""
        belief = self.advance(state, action, belief)
        mean, std = self.posterior(belief, embed)
        return mean + std * torch.randn_like(std), belief

    @torch.jit.export
    def imagine(self, state: Tensor, belief: Tensor, actions: Tensor) -> Tensor:
        """Open-loop prior rollout. actions is (batch, horizon, dim); returns features."""
        features = []
        for t in range(actions.size(1)):
            belief = self.advance(state, actions[:, t], belief)
            mean, std = self.prior(belief)
            state = mean + std * torch.randn_like(std)
            features.append(torch.cat([state, belief], -1))
        return torch.stack(features, 1)


class Planner(nn.Module):
    """Cross-entropy method over action sequences (paper Algorithm 2)."""

    def __init__(self, cfg: Config, rssm: nn.Module, reward: nn.Module, action_dim: int):
        super().__init__()
        self.rssm = rssm
        self.reward = reward
        self.horizon = cfg.horizon
        self.iterations = cfg.opt_iters
        self.candidates = cfg.candidates
        self.top_k = cfg.top_k
        self.action_dim = action_dim

    def forward(self, state: Tensor, belief: Tensor) -> Tensor:
        """Takes one (1, size) state and belief; returns the first planned action."""
        state = state.expand([self.candidates, -1])
        belief = belief.expand([self.candidates, -1])
        mean = state.new_zeros([self.horizon, self.action_dim])
        std = state.new_ones([self.horizon, self.action_dim])
        for _ in range(self.iterations):
            noise = torch.randn([self.candidates, self.horizon, self.action_dim], device=state.device)
            actions = (mean + std * noise).clamp(-1.0, 1.0)
            features = self.rssm.imagine(state, belief, actions)
            returns = self.reward(features).squeeze(-1).sum(1)
            _, top = returns.topk(self.top_k, sorted=False)
            best = actions[top]
            mean = best.mean(0)
            std = (best.var(0, unbiased=False) + 1e-6).sqrt()
        return mean[0]


class WorldModel(nn.Module):
    def __init__(self, cfg: Config, action_dim: int, device: torch.device):
        super().__init__()
        feature_size = cfg.state_size + cfg.belief_size
        self.encoder = Encoder()
        self.decoder = Decoder(feature_size, cfg.embed_size)
        layers = []
        size = feature_size
        for _ in range(cfg.reward_layers):
            layers += [nn.Linear(size, cfg.reward_units), nn.ReLU(inplace=True)]
            size = cfg.reward_units
        self.reward = nn.Sequential(*layers, nn.Linear(size, 1))
        self.rssm = RSSM(cfg, action_dim)
        self.apply(_init_weights)  # TensorFlow's Glorot uniform and zero bias defaults.
        self.to(device)  # Before scripting, so the compiled modules start on the right device.
        self.planner = Planner(cfg, self.rssm, self.reward, action_dim)
        if cfg.jit:
            self.planner = torch.jit.script(self.planner)
            self.rssm = self.planner.rssm  # Shares parameters with the scripted planner.
            self.reward = self.planner.reward


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


def gaussian_kl(q_mean: Tensor, q_std: Tensor, p_mean: Tensor, p_std: Tensor) -> Tensor:
    """KL(q || p) of diagonal Gaussians, summed over the last dimension."""
    variance_ratio = (q_std / p_std) ** 2
    mean_term = ((q_mean - p_mean) / p_std) ** 2
    return 0.5 * (variance_ratio + mean_term - 1.0 - variance_ratio.log()).sum(-1)
