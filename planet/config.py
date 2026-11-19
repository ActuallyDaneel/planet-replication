"""Hyperparameters. Defaults reproduce PlaNet on DM Control cheetah-run.

Values come from the paper's Appendix A (arXiv:1811.04551) and, where the paper
is silent or differs, from the official google-research/planet repository, since
that code produced the published results. See the README for the differences.
"""
import argparse
import sys
from dataclasses import dataclass, fields, replace


@dataclass
class Config:
    # Task
    domain: str = "cheetah"
    task: str = "run"
    action_repeat: int = 4
    episode_steps: int = 1000  # Environment steps, so 250 agent steps per episode.
    bit_depth: int = 5

    # Schedule: 5 random episodes, then 100 updates + 1 planned episode, repeated.
    seed_episodes: int = 5
    total_episodes: int = 1000
    collect_interval: int = 100
    batch_size: int = 50
    chunk_length: int = 50

    # World model
    belief_size: int = 200
    state_size: int = 30
    hidden_size: int = 200
    embed_size: int = 1024
    reward_units: int = 300  # Official code; the paper's text says 200.
    reward_layers: int = 3  # Official code; the paper's text says 2.
    min_std: float = 0.1

    # Loss and optimizer
    free_nats: float = 3.0
    reward_scale: float = 10.0  # Official code default; the paper does not state it.
    lr: float = 1e-3
    adam_eps: float = 1e-4
    grad_clip: float = 1000.0

    # CEM planner
    horizon: int = 12
    opt_iters: int = 10
    candidates: int = 1000
    top_k: int = 100
    exploration_noise: float = 0.3

    # Evaluation and persistence
    eval_every: int = 10  # One noiseless test episode per 10 train episodes (official).
    final_eval_episodes: int = 10  # The paper reports the mean of 10 trajectories.
    checkpoint_every: int = 10
    logdir: str = "runs/cheetah_run"
    seed: int = 0

    # Performance
    device: str = "cuda"
    replay_device: str = "auto"  # "auto" keeps frames in VRAM, falling back to RAM.
    jit: bool = True  # TorchScript the RSSM and planner loops.
    amp: bool = False  # bfloat16 for the CNNs only. Not in the paper, so opt-in.

    @property
    def agent_steps(self) -> int:
        return self.episode_steps // self.action_repeat


def smoke_config(cfg: Config) -> Config:
    """A tiny run that exercises the entire pipeline in seconds on a CPU."""
    return replace(
        cfg,
        episode_steps=120,
        seed_episodes=2,
        total_episodes=4,
        collect_interval=2,
        batch_size=4,
        chunk_length=8,
        belief_size=32,
        state_size=8,
        hidden_size=32,
        reward_units=32,
        reward_layers=2,
        horizon=4,
        opt_iters=2,
        candidates=32,
        top_k=4,
        eval_every=1,
        final_eval_episodes=1,
        checkpoint_every=1,
    )


def parse_config(description: str, **extra_flags: str) -> tuple[Config, argparse.Namespace]:
    """Expose every Config field as a CLI flag. extra_flags adds on/off flags."""
    # --smoke changes the defaults, so explicit flags still override it.
    defaults = smoke_config(Config()) if "--smoke" in sys.argv else Config()
    parser = argparse.ArgumentParser(description=description)
    for field in fields(Config):
        flag = "--" + field.name.replace("_", "-")
        default = getattr(defaults, field.name)
        if field.type is bool:
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        else:
            parser.add_argument(flag, type=field.type, default=default)
    parser.add_argument("--smoke", action="store_true", help="Tiny end-to-end test run.")
    for name, help_text in extra_flags.items():
        parser.add_argument("--" + name.replace("_", "-"), action="store_true", help=help_text)
    args = parser.parse_args()
    return Config(**{field.name: getattr(args, field.name) for field in fields(Config)}), args
