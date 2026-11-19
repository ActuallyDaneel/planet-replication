"""CEM imagining in the true simulator, matching models.Planner's search."""
import numpy as np

from ..env import Env  # noqa: F401  Sets MUJOCO_GL before MuJoCo loads.

import mujoco
from mujoco import rollout


class SimRollout:
    """Runs batches of action sequences from the environment's current state."""

    def __init__(self, env, reward, threads: int):
        self.reward = reward
        self.model = env.physics.model.ptr
        self.data = env.physics.data.ptr
        self.datas = [mujoco.MjData(self.model) for _ in range(threads)]
        self.runner = rollout.Rollout(nthread=threads)
        self.state = np.zeros(mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS))

    def rewards(self, actions: np.ndarray, repeat: int) -> np.ndarray:
        """actions is (batch, horizon, dim). Returns (batch, physics steps) rewards."""
        batch, horizon, dim = actions.shape
        mujoco.mj_getState(self.model, self.data, self.state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        # Sensors are read before each integration, so one extra step reports the last state.
        controls = np.zeros((batch, horizon * repeat + 1, dim))
        controls[:, :-1] = np.repeat(actions, repeat, axis=1)
        states, sensors = self.runner.rollout(
            self.model, self.datas, np.tile(self.state, (batch, 1)), controls,
            initial_warmstart=np.tile(self.data.qacc_warmstart, (batch, 1)),
        )
        return self.reward(states[:, :-1], sensors[:, 1:], controls[:, :-1])


class SimPlanner:
    def __init__(self, sim: SimRollout, repeat: int, horizon: int, candidates: int, top_k: int,
                 iterations: int, rng: np.random.Generator):
        self.sim, self.repeat, self.horizon, self.candidates = sim, repeat, horizon, candidates
        self.top_k, self.iterations, self.rng = top_k, iterations, rng
        self.action_dim = sim.model.nu

    def plan(self) -> np.ndarray:
        """Returns the first action of the planned sequence."""
        mean = np.zeros((self.horizon, self.action_dim))
        std = np.ones((self.horizon, self.action_dim))
        for _ in range(self.iterations):
            noise = self.rng.standard_normal((self.candidates, self.horizon, self.action_dim))
            actions = np.clip(mean + std * noise, -1.0, 1.0)
            returns = self.sim.rewards(actions, self.repeat).sum(1)
            best = actions[np.argpartition(returns, -self.top_k)[-self.top_k:]]
            mean = best.mean(0)
            std = np.sqrt(best.var(0) + 1e-6)
        return mean[0]
