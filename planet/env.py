"""DM Control suite task with action repeat and 64x64 pixel observations."""
import os

# GLFW is the only MuJoCo GL backend on Windows; it uses a hidden window.
if os.name == "nt":
    os.environ.setdefault("MUJOCO_GL", "glfw")

import absl.logging
import numpy as np
from dm_control import suite

absl.logging.set_verbosity(absl.logging.ERROR)  # Hides a harmless MJCF deprecation notice.

IMAGE_SIZE = 64


class Env:
    def __init__(self, domain: str, task: str, action_repeat: int, episode_steps: int, seed: int):
        self._env = suite.load(domain, task, task_kwargs={"random": seed})
        self._repeat = action_repeat
        self.max_steps = episode_steps // action_repeat
        self.action_dim = int(self._env.action_spec().shape[0])

    def seed(self, seed: int) -> None:
        """Sets what the next reset draws, so runs can share starting states."""
        self._env.task.random.seed(seed)

    def _render(self) -> np.ndarray:
        frame = self._env.physics.render(IMAGE_SIZE, IMAGE_SIZE, camera_id=0)
        return frame.transpose(2, 0, 1).copy()  # HWC to CHW, uint8.

    def reset(self) -> np.ndarray:
        self._env.reset()
        return self._render()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float]:
        """Applies the action for action_repeat steps, rendering only the last."""
        reward = 0.0
        for _ in range(self._repeat):
            reward += self._env.step(action).reward or 0.0
        return self._render(), reward

    def sample_action(self) -> np.ndarray:
        return np.random.uniform(-1.0, 1.0, self.action_dim).astype(np.float32)
