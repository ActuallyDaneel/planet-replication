"""Runs one stage's episodes: a grid on the simulator or a model, or a model screen.

    uv run python -m planet.study.evaluate --task cartpole_swingup --condition true --results out.jsonl
    uv run python -m planet.study.evaluate --task cheetah_run --condition learned --model seed0 --logdir runs/cheetah_run --results out.jsonl

Ctrl+C keeps finished episodes. Learned episodes finish first, as they take seconds. A
simulator episode can take minutes, so it stops at once and that episode is redone.
"""
import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from ..config import Config
from ..env import Env
from ..interrupt import StopRequest
from .results import GRID, SMOKE_GRID, Grid, append, episode_seed, remaining, top_k
from .sim_planner import SimPlanner, SimRollout
from .tasks import EPISODE_STEPS, TASKS, Task

import torch
from dm_control import suite

from ..agent import Agent, configure_torch

RESUME_MESSAGE = "Stopped. Finished episodes are saved. Run the same command to resume."


def record(path: Path, condition: str, model: str, job: tuple, score: float, tick: float, index: int, total: int) -> None:
    episode, horizon, candidates = job
    seconds = time.perf_counter() - tick
    append(path, dict(
        condition=condition, model=model, horizon=horizon, candidates=candidates, top_k=top_k(candidates),
        episode=episode, score=round(float(score), 2), seconds=round(seconds, 2),
    ))
    print(f"{index + 1}/{total} {condition} {model} H={horizon} J={candidates} ep={episode}: {score:.1f} ({seconds:.1f}s)", flush=True)


def run_true(task: Task, grid: Grid, threads: int, episode_steps: int, path: Path) -> None:
    stop = StopRequest("Stopping now: the episode in progress will be redone on resume.")
    env = suite.load(task.domain, task.task, task_kwargs={"random": 0})
    sim = SimRollout(env, task.reward, threads)
    todo = remaining(path, "true", "simulator", grid)
    print(f"True simulator, {task.name}: {len(todo)} episodes to run on {threads} threads", flush=True)
    for index, job in enumerate(todo):
        episode, horizon, candidates = job
        tick = time.perf_counter()
        env.task.random.seed(episode_seed(episode))
        env.reset()
        rng = np.random.default_rng([episode, horizon, candidates])
        planner = SimPlanner(sim, task.action_repeat, horizon, candidates, top_k(candidates), grid.iterations, rng)
        score = 0.0
        for _ in range(episode_steps // task.action_repeat):
            if stop.requested:
                print(RESUME_MESSAGE, flush=True)
                return
            action = planner.plan()
            for _ in range(task.action_repeat):
                score += env.step(action).reward or 0.0
        record(path, "true", "simulator", job, score, tick, index, len(todo))


def run_learned(condition: str, grid: Grid, logdir: Path, model: str, path: Path) -> None:
    stop = StopRequest("Stopping: finishing this episode first.")
    base = Config(**json.loads((logdir / "config.json").read_text()))
    device = configure_torch(base)
    weights = torch.load(logdir / "checkpoint.pt", map_location=device)
    print(f"Model {model}: checkpoint at episode {weights['episode']}", flush=True)
    env = Env(base.domain, base.task, base.action_repeat, base.episode_steps, 0)
    agents = {}
    todo = remaining(path, condition, model, grid)
    print(f"{condition}: {len(todo)} episodes to run", flush=True)
    for index, job in enumerate(todo):
        if stop.requested:
            print(RESUME_MESSAGE, flush=True)
            return
        episode, horizon, candidates = job
        tick = time.perf_counter()
        if (horizon, candidates) not in agents:
            cfg = dataclasses.replace(base, horizon=horizon, candidates=candidates, top_k=top_k(candidates), opt_iters=grid.iterations)
            agents[horizon, candidates] = Agent(cfg, env.action_dim, device)
            agents[horizon, candidates].model.load_state_dict(weights["model"])
        # Screens use their own seeds and starts, so they are fresh tests of the checkpoint.
        offset = 1_000 if condition == "screen" else 0
        torch.manual_seed((episode + offset) * 100_000 + horizon * 10_000 + candidates)
        env.seed(episode_seed(episode + offset))
        score = agents[horizon, candidates].run_episode(env, 0.0).score
        record(path, condition, model, job, score, tick, index, len(todo))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--condition", choices=["true", "learned", "screen"], required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--logdir", type=Path, help="Trained run, for learned and screen.")
    parser.add_argument("--model", help="The model's name in the results, such as seed0.")
    parser.add_argument("--threads", type=int, default=12, help="CPU threads for the true simulator.")
    parser.add_argument("--smoke", action="store_true", help="Tiny grid, to test the pipeline.")
    args = parser.parse_args()
    grid = SMOKE_GRID if args.smoke else GRID
    try:
        if args.condition == "true":
            run_true(TASKS[args.task], grid, args.threads, 120 if args.smoke else EPISODE_STEPS, args.results)
        else:
            run_learned(args.condition, grid, args.logdir, args.model, args.results)
    except KeyboardInterrupt:
        print(RESUME_MESSAGE, flush=True)


if __name__ == "__main__":
    main()
