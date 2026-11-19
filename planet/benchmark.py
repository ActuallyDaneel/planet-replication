"""Measures this machine's speed and projects total training time.

    uv run python -m planet.benchmark

Times real update and planning steps at full paper settings on synthetic data,
for about a minute. Nothing is written to disk. Compares the TorchScript and
bfloat16 options so you can choose the fastest combination.
"""
import dataclasses
import time

import numpy as np
import torch

from .agent import Agent, configure_torch
from .config import Config, parse_config
from .env import Env
from .models import preprocess
from .replay import Replay


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_env(env: Env, actions: int = 250) -> float:
    env.reset()
    action = np.zeros(env.action_dim, np.float32)
    start = time.perf_counter()
    for _ in range(actions):
        env.step(action)
    return (time.perf_counter() - start) / actions


def fill_replay(cfg: Config, replay: Replay, env: Env) -> None:
    """A few episodes of random pixels: sampling and CNN cost do not depend on content."""
    generator = np.random.default_rng(0)
    for _ in range(cfg.seed_episodes):
        replay.add(
            generator.integers(0, 256, (replay.length, 3, 64, 64), dtype=np.uint8),
            generator.uniform(-1, 1, (replay.length, env.action_dim)).astype(np.float32),
            generator.uniform(0, 1, replay.length).astype(np.float32),
        )


def time_variant(cfg: Config, env: Env, replay: Replay, device: torch.device, updates: int, plans: int) -> tuple[float, float]:
    agent = Agent(cfg, env.action_dim, device)
    batch = lambda: replay.sample(cfg.batch_size, cfg.chunk_length)

    # The first calls include TorchScript profiling and optimization; exclude them.
    for _ in range(8):
        agent.update(*batch())
    _sync(device)
    start = time.perf_counter()
    for _ in range(updates):
        agent.update(*batch())
    _sync(device)
    update_seconds = (time.perf_counter() - start) / updates

    state = torch.zeros(1, cfg.state_size, device=device)
    belief = torch.zeros(1, cfg.belief_size, device=device)
    action = torch.zeros(1, env.action_dim, device=device)
    frame = torch.zeros(1, 3, 64, 64, dtype=torch.uint8, device=device)

    def plan_step():
        with torch.inference_mode(), agent._cnn_autocast():
            embed = agent.model.encoder(preprocess(frame, cfg.bit_depth))
        with torch.inference_mode():
            s, b = agent.model.rssm.filter_step(state, action, belief, embed.float())
            result = agent.model.planner(s, b)
            result = result.cpu()
        return result

    for _ in range(8):
        plan_step()
    start = time.perf_counter()
    for _ in range(plans):
        plan_step()
    plan_seconds = (time.perf_counter() - start) / plans
    del agent
    return update_seconds, plan_seconds


def main() -> None:
    cfg, args = parse_config("Benchmark PlaNet and project training time.", no_variants="Measure only the given settings.")
    device = configure_torch(cfg)
    env = Env(cfg.domain, cfg.task, cfg.action_repeat, cfg.episode_steps, cfg.seed)

    env_seconds = time_env(env)
    print(f"Environment: {env_seconds * 1e3:.2f} ms per agent step (physics + render)")

    capacity = max(cfg.total_episodes, cfg.seed_episodes) if not args.smoke else cfg.seed_episodes
    replay = Replay(capacity, env.max_steps, env.action_dim, device, cfg.replay_device)
    fill_replay(cfg, replay, env)
    print(f"Replay ({capacity} episode capacity) in {replay.storage}")

    if args.no_variants:
        variants = [("selected", cfg)]
    else:
        variants = [
            ("plain python", dataclasses.replace(cfg, jit=False, amp=False)),
            ("jit (default)", dataclasses.replace(cfg, jit=True, amp=False)),
        ]
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            variants.append(("jit + amp", dataclasses.replace(cfg, jit=True, amp=True)))

    updates, plans = (5, 5) if args.smoke or device.type == "cpu" else (60, 40)
    planned_steps = (cfg.total_episodes - cfg.seed_episodes) * cfg.agent_steps
    eval_steps = (cfg.total_episodes // cfg.eval_every + cfg.final_eval_episodes) * cfg.agent_steps
    total_updates = (cfg.total_episodes - cfg.seed_episodes) * cfg.collect_interval

    print(f"\n{'variant':<16}{'update':>11}{'plan step':>12}{'per episode':>13}{'total':>10}")
    for name, variant in variants:
        update, plan = time_variant(variant, env, replay, device, updates, plans)
        step = plan + env_seconds
        total = total_updates * update + (planned_steps + eval_steps) * step
        episode = cfg.collect_interval * update + cfg.agent_steps * step
        print(f"{name:<16}{update * 1e3:>9.1f}ms{plan * 1e3:>10.1f}ms{episode:>12.1f}s{total / 3600:>9.1f}h")
    if device.type == "cuda":
        print(f"\nPeak VRAM reserved by training: {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB")
    print(f"Projection covers {total_updates:,} updates and {planned_steps + eval_steps:,} planned steps.")


if __name__ == "__main__":
    main()
