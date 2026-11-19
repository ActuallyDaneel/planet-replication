"""Each study task, with its reward computed in bulk from simulator rollouts.

The rewards use the same dm_control functions and values as the suite's tasks. Tests
check them against the real environments, because the planner relies on them matching.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
from dm_control.utils import rewards

EPISODE_STEPS = 1000  # Physics steps per episode, on both tasks.


def cheetah_run_reward(states: np.ndarray, sensors: np.ndarray, controls: np.ndarray) -> np.ndarray:
    speed = sensors[..., 0]  # torso_subtreelinvel, x component.
    return rewards.tolerance(speed, bounds=(10, float("inf")), margin=10, value_at_margin=0, sigmoid="linear")


def cartpole_swingup_reward(states: np.ndarray, sensors: np.ndarray, controls: np.ndarray) -> np.ndarray:
    # The full-physics state is [time, slider, hinge, slider velocity, hinge velocity].
    cart, hinge, hinge_velocity = states[..., 1], states[..., 2], states[..., 4]
    upright = (np.cos(hinge) + 1) / 2
    centered = (1 + rewards.tolerance(cart, margin=2)) / 2
    small_control = (4 + rewards.tolerance(controls[..., 0], margin=1, value_at_margin=0, sigmoid="quadratic")) / 5
    small_velocity = (1 + rewards.tolerance(hinge_velocity, margin=5)) / 2
    return upright * small_control * small_velocity * centered


@dataclass(frozen=True)
class Task:
    name: str
    domain: str
    task: str
    action_repeat: int
    reward: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    train_hours: float  # Estimated, for the time remaining shown by --list.


# The paper's action repeats (Appendix A). Cheetah comes first because only it has a
# published grid to compare with (Appendix J, Figure 12).
TASKS = {
    task.name: task
    for task in [
        Task("cheetah_run", "cheetah", "run", 4, cheetah_run_reward, train_hours=5.3),
        Task("cartpole_swingup", "cartpole", "swingup", 8, cartpole_swingup_reward, train_hours=4.0),
    ]
}
