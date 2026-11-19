"""The simulator planner must predict what the real environments do, on every task."""
import json

import numpy as np
import pytest
from dm_control import suite

from planet.study import results
from planet.study.pipeline import Stage, build_stages, grid_progress, train_progress
from planet.study.report import rank_correlation, write_report
from planet.study.results import GRID, Grid, append, load, remaining, top_k
from planet.study.sim_planner import SimPlanner, SimRollout
from planet.study.tasks import TASKS

TASK_NAMES = list(TASKS)


def running_env(task):
    env = suite.load(task.domain, task.task, task_kwargs={"random": 0})
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(100):  # Away from the reset state, with any contacts active.
        env.step(rng.uniform(-1, 1, env.action_spec().shape))
    return env, rng


@pytest.mark.parametrize("name", TASK_NAMES)
def test_rollout_rewards_match_the_real_environment(name):
    task = TASKS[name]
    env, rng = running_env(task)
    dim = env.action_spec().shape[0]
    actions = rng.uniform(-1, 1, (3, 12, dim))
    predicted = SimRollout(env, task.reward, threads=2).rewards(actions, task.action_repeat)
    assert predicted.shape == (3, 12 * task.action_repeat)
    saved = env.physics.get_state().copy()
    for sequence in range(3):
        env.physics.set_state(saved)
        env.physics.forward()
        actual = [env.step(a).reward for a in actions[sequence] for _ in range(task.action_repeat)]
        assert np.allclose(predicted[sequence], actual, atol=2e-3), np.abs(predicted[sequence] - actual).max()


@pytest.mark.parametrize("name", TASK_NAMES)
def test_rollout_leaves_the_environment_alone(name):
    task = TASKS[name]
    env, rng = running_env(task)
    before = env.physics.get_state().copy()
    dim = env.action_spec().shape[0]
    SimRollout(env, task.reward, threads=2).rewards(rng.uniform(-1, 1, (4, 6, dim)), task.action_repeat)
    assert np.array_equal(before, env.physics.get_state())


@pytest.mark.parametrize("name", TASK_NAMES)
def test_planner_returns_a_valid_action(name):
    task = TASKS[name]
    env, rng = running_env(task)
    action = SimPlanner(SimRollout(env, task.reward, threads=2), task.action_repeat, 8, 200, 20, 5, rng).plan()
    assert action.shape == env.action_spec().shape and np.all(np.abs(action) <= 1)


def test_starting_states_repeat_across_settings():
    task = TASKS["cartpole_swingup"]
    env = suite.load(task.domain, task.task, task_kwargs={"random": 0})
    states = []
    for _ in range(2):
        env.task.random.seed(results.episode_seed(3))
        env.reset()
        states.append(env.physics.get_state().copy())
        env.step(np.ones(1))
    assert np.array_equal(states[0], states[1])


def test_top_fraction_includes_the_papers_setting_and_always_selects():
    assert top_k(1000) == 100
    assert all(top_k(j) < j for j in GRID.candidates)


def row(**changes):
    return dict(dict(condition="true", model="simulator", horizon=6, candidates=100, top_k=10, episode=0, score=1.0, seconds=1.0), **changes)


def test_results_resume_after_a_cut_short_line(tmp_path):
    path = tmp_path / "results.jsonl"
    append(path, row())
    with path.open("a") as file:
        file.write('{"condition": "tr')  # A power cut mid-write.
    assert len(load(path)) == 1
    append(path, row(episode=1))
    assert [r["episode"] for r in load(path)] == [0, 1]
    assert len(remaining(path, "true", "simulator", GRID)) == GRID.episodes * 12 - 2


def test_conditions_and_models_do_not_count_as_each_other(tmp_path):
    path = tmp_path / "results.jsonl"
    append(path, row(condition="learned", model="seed0"))
    assert len(remaining(path, "learned", "seed0", GRID)) == GRID.episodes * 12 - 1
    assert len(remaining(path, "learned", "seed1", GRID)) == GRID.episodes * 12
    assert len(remaining(path, "true", "simulator", GRID)) == GRID.episodes * 12


def test_adding_a_setting_only_adds_the_new_work(tmp_path):
    path = tmp_path / "results.jsonl"
    small = Grid(horizons=(6,), candidates=(100,), episodes=1)
    append(path, row())
    assert remaining(path, "true", "simulator", small) == []
    assert remaining(path, "true", "simulator", Grid(horizons=(6, 8), candidates=(100,), episodes=1)) == [(0, 8, 100)]


def test_training_only_counts_as_done_after_its_final_evaluation(tmp_path):
    stage = Stage("t", [], train_progress(tmp_path, 3), 1.0)
    assert stage.progress() == (0, 3)
    (tmp_path / "metrics.csv").write_text("episode,x\n2,0\n3,0\n")
    assert stage.progress() == (2, 3) and not stage.done  # All episodes logged, but not evaluated.
    (tmp_path / "final_eval.json").write_text("{}")
    assert stage.done


def test_stages_cover_both_tasks_all_seeds_and_reuse_the_finished_run(tmp_path):
    from pathlib import Path
    stages = build_stages(Path("runs/study"), list(TASKS), [0, 1, 2], smoke=False)
    names = [stage.name for stage in stages]
    assert len(names) == len(set(names)) == 2 * (3 * 3 + 1)
    train = next(s for s in stages if s.name == "cheetah_run train seed0")
    assert "runs\\cheetah_run" in train.arguments or "runs/cheetah_run" in train.arguments
    for stage in stages:
        if "learned" in stage.name or "screen" in stage.name:
            assert stage.needs == [stage.name.replace("learned", "train").replace("screen", "train")]


def test_rank_correlation():
    assert rank_correlation(np.array([1.0, 2, 3]), np.array([10.0, 20, 30])) == pytest.approx(1)
    assert rank_correlation(np.array([1.0, 2, 3]), np.array([30.0, 20, 10])) == pytest.approx(-1)


def test_report_handles_partial_results(tmp_path):
    results_file = tmp_path / "cheetah_run" / "results.jsonl"
    append(results_file, row(score=500.0))
    append(results_file, row(condition="screen", model="seed0", horizon=12, candidates=1000, score=600.0))
    text = write_report(tmp_path, GRID)
    assert "cheetah_run" in text and "no data" in text and "(tmp_path)" not in text
    assert (tmp_path / "report.md").exists()
    json.dumps(text)
