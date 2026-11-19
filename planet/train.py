"""PlaNet training loop. Run with: uv run python -m planet.train

Press Ctrl+C to stop: it finishes the current episode, saves, and exits. Rerunning with
the same --logdir resumes from there. A second Ctrl+C, or closing the window, stops at
once and resumes from the last periodic checkpoint, losing at most checkpoint_every episodes.
"""
import csv
import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .agent import Agent, Episode, configure_torch, random_episode
from .config import Config, parse_config
from .env import Env
from .interrupt import StopRequest
from .replay import Replay

FIELDS = [
    "episode", "env_steps", "updates", "train_return", "eval_return", "image_loss", "reward_loss",
    "kl", "grad_norm", "train_seconds", "collect_seconds", "elapsed_hours",
]


def save_episode(directory: Path, index: int, episode: Episode) -> None:
    np.savez(directory / f"{index:05d}.npz", frames=episode.frames, actions=episode.actions, rewards=episode.rewards)


def load_episodes(directory: Path, count: int, replay: Replay) -> None:
    for index in range(count):
        with np.load(directory / f"{index:05d}.npz") as data:
            replay.add(data["frames"], data["actions"], data["rewards"])


def save_checkpoint(path: Path, agent: Agent, episode: int, updates: int, elapsed: float) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(dict(
        model=agent.model.state_dict(), optimizer=agent.optimizer.state_dict(),
        episode=episode, updates=updates, elapsed=elapsed,
    ), temporary)
    os.replace(temporary, path)  # Atomic, so stopping mid-save cannot corrupt the checkpoint.


def trim_metrics(path: Path, episode: int) -> None:
    """Drops rows from after the checkpoint being resumed, as those episodes get redone."""
    if path.exists():
        with path.open(newline="") as file:
            rows = [row for row in csv.DictReader(file) if int(row["episode"]) <= episode]
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)


def append_metrics(path: Path, row: dict) -> None:
    is_new = not path.exists()
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def train(cfg: Config) -> None:
    stop = StopRequest("Stopping: finishing this episode, then saving.")
    device = configure_torch(cfg)
    logdir = Path(cfg.logdir)
    episode_dir = logdir / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)
    (logdir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
    checkpoint_path = logdir / "checkpoint.pt"

    env = Env(cfg.domain, cfg.task, cfg.action_repeat, cfg.episode_steps, cfg.seed)
    eval_env = Env(cfg.domain, cfg.task, cfg.action_repeat, cfg.episode_steps, cfg.seed + 1)
    agent = Agent(cfg, env.action_dim, device)
    replay = Replay(cfg.total_episodes, env.max_steps, env.action_dim, device, cfg.replay_device)
    print(f"Device {device} | replay in {replay.storage} | jit={cfg.jit} amp={cfg.amp}")

    episode = updates = 0
    previous_elapsed = 0.0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        agent.model.load_state_dict(checkpoint["model"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        episode, updates = checkpoint["episode"], checkpoint["updates"]
        previous_elapsed = checkpoint["elapsed"]
        load_episodes(episode_dir, episode, replay)
        print(f"Resumed at episode {episode}, update {updates}")
    else:
        for _ in range(cfg.seed_episodes):
            seed_episode = random_episode(env)
            save_episode(episode_dir, episode, seed_episode)
            replay.add(seed_episode.frames, seed_episode.actions, seed_episode.rewards)
            episode += 1
        print(f"Collected {episode} random seed episodes")

    trim_metrics(logdir / "metrics.csv", episode)
    start = time.perf_counter()
    start_episode = episode
    while episode < cfg.total_episodes:
        tick = time.perf_counter()
        metrics = {}
        for _ in range(cfg.collect_interval):
            metrics = agent.update(*replay.sample(cfg.batch_size, cfg.chunk_length))
        updates += cfg.collect_interval
        metrics = {key: value.item() for key, value in metrics.items()}
        if device.type == "cuda":
            torch.cuda.synchronize()
        tock = time.perf_counter()

        collected = agent.run_episode(env, cfg.exploration_noise)
        replay.add(collected.frames, collected.actions, collected.rewards)
        save_episode(episode_dir, episode, collected)
        episode += 1
        collect_seconds = time.perf_counter() - tock

        eval_return = ""
        if episode % cfg.eval_every == 0:
            eval_return = agent.run_episode(eval_env, 0.0).score

        now = time.perf_counter() - start
        elapsed = previous_elapsed + now
        append_metrics(logdir / "metrics.csv", dict(
            episode=episode, env_steps=episode * cfg.episode_steps, updates=updates,
            train_return=round(collected.score, 2),
            eval_return=round(eval_return, 2) if eval_return != "" else "",
            train_seconds=round(tock - tick, 3), collect_seconds=round(collect_seconds, 3),
            elapsed_hours=round(elapsed / 3600, 4),
            **{key: round(value, 4) for key, value in metrics.items()},
        ))
        remaining = (cfg.total_episodes - episode) * now / (episode - start_episode)
        print(
            f"ep {episode:4d}/{cfg.total_episodes} | return {collected.score:6.1f}"
            + (f" | eval {eval_return:6.1f}" if eval_return != "" else "")
            + f" | img {metrics['image_loss']:7.1f} rew {metrics['reward_loss']:.3f} kl {metrics['kl']:.2f}"
            + f" | train {tock - tick:5.1f}s collect {collect_seconds:5.1f}s"
            + f" | elapsed {elapsed / 3600:.2f}h, left {remaining / 3600:.2f}h",
            flush=True,
        )

        if episode % cfg.checkpoint_every == 0 or stop.requested:
            save_checkpoint(checkpoint_path, agent, episode, updates, elapsed)
        if stop.requested:
            print(f"Saved at episode {episode}. Run the same command to resume.", flush=True)
            return

    save_checkpoint(checkpoint_path, agent, episode, updates, previous_elapsed + time.perf_counter() - start)
    scores = []
    for _ in range(cfg.final_eval_episodes):
        if stop.requested:
            print("Training is saved. Run the same command to redo the final evaluation.", flush=True)
            return
        scores.append(agent.run_episode(eval_env, 0.0).score)
    summary = dict(mean=float(np.mean(scores)), std=float(np.std(scores)), scores=scores)
    (logdir / "final_eval.json").write_text(json.dumps(summary, indent=2))
    print(f"Final evaluation over {len(scores)} episodes: {summary['mean']:.1f} +/- {summary['std']:.1f}")


if __name__ == "__main__":
    config, _ = parse_config("Train PlaNet.")
    try:
        train(config)
    except KeyboardInterrupt:
        print("Stopped at once. Run the same command to resume from the last checkpoint.", flush=True)
