"""CPU-only checks. Run with: uv run --with pytest pytest tests -q"""
import numpy as np
import pytest
import torch
from torch import distributions as D

from planet.agent import Agent, random_episode
from planet.config import Config, smoke_config
from planet.env import Env
from planet.models import Decoder, Encoder, Planner, WorldModel, gaussian_kl, preprocess
from planet.replay import Replay

CPU = torch.device("cpu")


def small(**overrides) -> Config:
    cfg = smoke_config(Config(device="cpu", replay_device="cpu"))
    return Config(**{**cfg.__dict__, **overrides})


def test_cnn_shapes_match_official_networks():
    assert Encoder()(torch.zeros(2, 3, 64, 64)).shape == (2, 1024)
    assert Decoder(230, 1024)(torch.zeros(2, 230)).shape == (2, 3, 64, 64)


def test_preprocess_range_and_quantization():
    frames = torch.arange(256, dtype=torch.uint8).repeat(10, 1)
    x = preprocess(frames, 5)
    assert x.min() >= -0.5 and x.max() < 0.5
    # Bins are 1/32 wide, and pixel value 255 lands in the top bin.
    assert ((x + 0.5) * 32).floor().max() == 31
    assert ((x + 0.5) * 32).floor()[0, 8] == 1


def test_kl_matches_torch_distributions():
    q_mean, p_mean = torch.randn(5, 30), torch.randn(5, 30)
    q_std, p_std = torch.rand(5, 30) + 0.1, torch.rand(5, 30) + 0.1
    expected = D.kl_divergence(D.Normal(q_mean, q_std), D.Normal(p_mean, p_std)).sum(-1)
    assert torch.allclose(gaussian_kl(q_mean, q_std, p_mean, p_std), expected, atol=1e-5)


def test_replay_chunks_stay_aligned_and_in_bounds():
    replay = Replay(3, 10, 2, CPU, "cpu")
    for episode in range(2):
        marker = np.arange(11, dtype=np.float32) + 100 * episode
        frames = np.broadcast_to((marker % 256).astype(np.uint8)[:, None, None, None], (11, 3, 64, 64)).copy()
        replay.add(frames, np.stack([marker, marker], 1), marker)
    frames, actions, rewards = replay.sample(64, 5)
    assert frames.shape == (64, 5, 3, 64, 64) and actions.shape == (64, 5, 2) and rewards.shape == (64, 5)
    assert torch.all(rewards[:, 1:] - rewards[:, :-1] == 1)  # Consecutive, within one episode.
    assert torch.all(actions[..., 0] == rewards)
    assert torch.all(frames[:, :, 0, 0, 0].float() == rewards % 256)


def test_scripted_modules_share_parameters_with_optimizer():
    agent = Agent(small(jit=True), 6, CPU)
    # Training has to change what the planner sees, or planning would use stale weights.
    ids = {id(p) for p in agent.params}
    assert all(id(p) in ids for p in agent.model.planner.parameters())
    assert len(ids) == len(list(WorldModel(small(jit=False), 6, CPU).parameters()))


@pytest.mark.parametrize("jit", [False, True])
def test_update_reduces_loss_on_a_fixed_batch(jit):
    torch.manual_seed(0)
    agent = Agent(small(jit=jit), 6, CPU)
    frames = torch.randint(0, 256, (4, 8, 3, 64, 64), dtype=torch.uint8)
    frames[:, :, :, 16:48, 16:48] = 200  # Structure to learn.
    actions, rewards = torch.rand(4, 8, 6) * 2 - 1, torch.rand(4, 8)
    first = agent.update(frames, actions, rewards)
    planner_before = next(iter(agent.model.planner.rssm.parameters())).clone()
    for _ in range(30):
        last = agent.update(frames, actions, rewards)
    assert last["image_loss"] < first["image_loss"]
    assert last["reward_loss"] < first["reward_loss"]
    assert not torch.equal(planner_before, next(iter(agent.model.planner.rssm.parameters())))


@pytest.mark.parametrize("jit", [False, True])
def test_cem_converges_on_a_known_reward(jit):
    """With a reward that prefers action 0.5 in the first dimension, CEM must find it."""
    cfg = small(jit=jit, candidates=300, top_k=30, opt_iters=8, horizon=3)
    action_dim = 2

    class IdentityRSSM(torch.nn.Module):
        @torch.jit.export
        def imagine(self, state: torch.Tensor, belief: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
            return actions

    class TargetReward(torch.nn.Module):
        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return -(features[..., :1] - 0.5) ** 2

    planner = Planner(cfg, IdentityRSSM(), TargetReward(), action_dim)
    if jit:
        planner = torch.jit.script(planner)
    action = planner(torch.zeros(1, cfg.state_size), torch.zeros(1, cfg.belief_size))
    assert action.shape == (action_dim,)
    assert abs(action[0].item() - 0.5) < 0.05


def test_default_config_matches_paper_appendix_a():
    cfg = Config()
    assert (cfg.batch_size, cfg.chunk_length, cfg.bit_depth) == (50, 50, 5)
    assert (cfg.horizon, cfg.opt_iters, cfg.candidates, cfg.top_k) == (12, 10, 1000, 100)
    assert (cfg.seed_episodes, cfg.collect_interval, cfg.exploration_noise) == (5, 100, 0.3)
    assert (cfg.lr, cfg.adam_eps, cfg.grad_clip, cfg.free_nats) == (1e-3, 1e-4, 1000.0, 3.0)
    assert (cfg.belief_size, cfg.state_size, cfg.action_repeat, cfg.total_episodes) == (200, 30, 4, 1000)
    assert cfg.agent_steps == 250


def test_environment_produces_pixels_and_rewards():
    env = Env("cheetah", "run", 4, 40, seed=0)
    assert env.action_dim == 6 and env.max_steps == 10
    episode = random_episode(env)
    assert episode.frames.shape == (11, 3, 64, 64) and episode.frames.dtype == np.uint8
    assert episode.frames.std() > 0  # A real image, not a blank buffer.
    assert episode.actions[0].sum() == 0 and episode.rewards[0] == 0
    assert (episode.frames[0] != episode.frames[-1]).any()
