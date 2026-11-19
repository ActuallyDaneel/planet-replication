"""Model update and environment interaction, shared by training and the benchmark."""
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch

from .config import Config
from .env import Env
from .models import WorldModel, gaussian_kl, preprocess


def configure_torch(cfg: Config) -> torch.device:
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # Input shapes never change.
        torch.set_num_threads(1)  # Stops idle CPU math threads competing with MuJoCo.
    return device


@dataclass
class Episode:
    frames: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray

    @property
    def score(self) -> float:
        return float(self.rewards.sum())


class Agent:
    def __init__(self, cfg: Config, action_dim: int, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.action_dim = action_dim
        self.model = WorldModel(cfg, action_dim, device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.lr, eps=cfg.adam_eps,
            fused=device.type == "cuda",  # Noticeably faster than the Python Adam loop.
        )
        self.params = list(self.model.parameters())

    def _cnn_autocast(self):
        """bfloat16 needs no loss scaling; the RSSM and all losses stay in float32."""
        if self.cfg.amp:
            return torch.autocast(self.device.type, dtype=torch.bfloat16)
        return nullcontext()

    def update(self, frames: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        model = self.model
        batch, length = frames.shape[:2]
        images = preprocess(frames.flatten(0, 1), cfg.bit_depth)

        with self._cnn_autocast():
            embed = model.encoder(images)
        embed = embed.float().view(batch, length, -1)

        states, beliefs, prior_mean, prior_std, post_mean, post_std = model.rssm.observe(embed, actions)
        features = torch.cat([states, beliefs], -1)

        with self._cnn_autocast():
            reconstruction = model.decoder(features.flatten(0, 1))
        # Unit-variance Gaussians. Constant terms are dropped as they have no gradient.
        image_loss = 0.5 * (reconstruction.float() - images).square().sum((1, 2, 3)).mean()
        reward_loss = 0.5 * (model.reward(features).squeeze(-1) - rewards).square().mean()
        kl = gaussian_kl(post_mean, post_std, prior_mean, prior_std)
        kl_loss = (kl - cfg.free_nats).clamp(min=0.0).mean()
        loss = image_loss + cfg.reward_scale * reward_loss + kl_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, cfg.grad_clip)
        self.optimizer.step()
        # Tensors, not floats: reading a value would force a GPU sync on every update.
        return dict(
            image_loss=image_loss.detach(), reward_loss=reward_loss.detach(),
            kl=kl.detach().mean(), grad_norm=grad_norm.detach(),
        )

    @torch.inference_mode()
    def run_episode(self, env: Env, noise: float) -> Episode:
        """Plans at every step. noise is the exploration stddev, 0 for evaluation."""
        cfg = self.cfg
        model = self.model
        steps = env.max_steps
        frames = np.zeros((steps + 1, 3, 64, 64), np.uint8)
        actions = np.zeros((steps + 1, self.action_dim), np.float32)
        rewards = np.zeros(steps + 1, np.float32)

        frames[0] = env.reset()
        state = torch.zeros(1, cfg.state_size, device=self.device)
        belief = torch.zeros(1, cfg.belief_size, device=self.device)
        action = torch.zeros(1, self.action_dim, device=self.device)
        for t in range(steps):
            image = torch.from_numpy(frames[t]).to(self.device)[None]
            with self._cnn_autocast():
                embed = model.encoder(preprocess(image, cfg.bit_depth))
            state, belief = model.rssm.filter_step(state, action, belief, embed.float())
            action = model.planner(state, belief)
            if noise > 0:
                action = (action + noise * torch.randn_like(action)).clamp(-1.0, 1.0)
            action = action[None]
            frames[t + 1], rewards[t + 1] = env.step(action[0].cpu().numpy())
            actions[t + 1] = action[0].cpu().numpy()
        return Episode(frames, actions, rewards)


def random_episode(env: Env) -> Episode:
    steps = env.max_steps
    frames = np.zeros((steps + 1, 3, 64, 64), np.uint8)
    actions = np.zeros((steps + 1, env.action_dim), np.float32)
    rewards = np.zeros(steps + 1, np.float32)
    frames[0] = env.reset()
    for t in range(steps):
        actions[t + 1] = env.sample_action()
        frames[t + 1], rewards[t + 1] = env.step(actions[t + 1])
    return Episode(frames, actions, rewards)
