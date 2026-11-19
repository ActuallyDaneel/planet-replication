"""The settings grid, and the append-only results that make every grid resumable.

A setting's result is one line per finished episode. Whatever is on disk defines what is
left to do, so a grid can stop at any time, and adding a setting only adds the new work.
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Grid:
    horizons: tuple[int, ...] = (6, 8, 12, 14)
    candidates: tuple[int, ...] = (100, 300, 1000)
    episodes: int = 10
    # Fresh episodes at the paper's setting, to catch a bad checkpoint before trusting a model.
    screen_episodes: int = 30
    iterations: int = 10
    paper_horizon: int = 12
    paper_candidates: int = 1000

    @property
    def settings(self) -> list[tuple[int, int]]:
        return [(horizon, candidates) for horizon in self.horizons for candidates in self.candidates]


GRID = Grid()
SMOKE_GRID = Grid(horizons=(2, 3), candidates=(10, 20), episodes=1, screen_episodes=1, iterations=2,
                  paper_horizon=3, paper_candidates=20)


def top_k(candidates: int) -> int:
    """The best 10%, as a fixed 100 would select nothing from 100 candidates.

    This is the fraction=0.1 row of the paper's Figure 12, and gives its J=1000, K=100.
    """
    return round(candidates * 0.1)


def episode_seed(episode: int) -> int:
    """Each episode index starts from the same state for every setting and condition."""
    return 50_000 + episode


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:  # A line cut short by a power loss gets redone.
            continue
    return rows


def append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Starts on a new line if the previous write was cut short.
    prefix = "\n" if path.exists() and path.stat().st_size and not path.read_bytes().endswith(b"\n") else ""
    with path.open("a") as file:
        file.write(prefix + json.dumps(row) + "\n")
        file.flush()
        os.fsync(file.fileno())


def planned(condition: str, grid: Grid) -> list[tuple[int, int, int]]:
    """(episode, horizon, candidates) in whole rounds, so stopping early stays balanced."""
    if condition == "screen":
        return [(episode, grid.paper_horizon, grid.paper_candidates) for episode in range(grid.screen_episodes)]
    return [(episode, horizon, candidates) for episode in range(grid.episodes) for horizon, candidates in grid.settings]


def remaining(path: Path, condition: str, model: str, grid: Grid) -> list[tuple[int, int, int]]:
    done = {
        (row["episode"], row["horizon"], row["candidates"])
        for row in load(path)
        if row["condition"] == condition and row["model"] == model
    }
    return [job for job in planned(condition, grid) if job not in done]
