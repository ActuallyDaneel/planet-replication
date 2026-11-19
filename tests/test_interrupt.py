"""Ctrl+C handling. Real console interrupts were also tested by hand; see the README."""
import csv
import signal

import pytest
import torch

from planet import train as train_module
from planet.config import Config, smoke_config
from planet.interrupt import StopRequest


@pytest.fixture
def restore_signals():
    saved = signal.getsignal(signal.SIGINT)
    yield
    signal.signal(signal.SIGINT, saved)


def test_first_press_requests_a_stop_and_second_stops_at_once(restore_signals):
    stop = StopRequest("Stopping.")
    assert not stop.requested
    signal.raise_signal(signal.SIGINT)
    assert stop.requested
    with pytest.raises(KeyboardInterrupt):
        signal.raise_signal(signal.SIGINT)


def test_training_saves_on_stop_and_resumes_at_the_next_episode(tmp_path, monkeypatch, restore_signals):
    cfg = smoke_config(Config(device="cpu", logdir=str(tmp_path), jit=False))
    cfg = type(cfg)(**{**cfg.__dict__, "checkpoint_every": 1000, "total_episodes": 5})

    class StopAtOnce:
        requested = True

        def __init__(self, _message):
            pass

    with monkeypatch.context() as patch:
        patch.setattr(train_module, "StopRequest", StopAtOnce)
        train_module.train(cfg)
    saved = torch.load(tmp_path / "checkpoint.pt", weights_only=True)
    # Saved after the first planned episode, although no periodic checkpoint was due.
    assert saved["episode"] == cfg.seed_episodes + 1
    assert not (tmp_path / "final_eval.json").exists()

    train_module.train(cfg)
    assert torch.load(tmp_path / "checkpoint.pt", weights_only=True)["episode"] == cfg.total_episodes
    assert (tmp_path / "final_eval.json").exists()
    with (tmp_path / "metrics.csv").open() as file:
        episodes = [int(row["episode"]) for row in csv.DictReader(file)]
    assert episodes == list(range(cfg.seed_episodes + 1, cfg.total_episodes + 1))
