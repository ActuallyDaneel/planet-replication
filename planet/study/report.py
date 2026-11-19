"""Answers the study's three questions from whatever results exist so far.

    uv run python -m planet.study.report

1. Do the same settings rank highest under both conditions?
2. Do the trends match, such as more candidates helping?
3. Where does the paper's setting rank under each?

Writes report.md under the study root. Safe to run at any time, including mid-study:
each section says how much of the grid it covers.
"""
import argparse
from pathlib import Path

import numpy as np

from .results import GRID, SMOKE_GRID, Grid, load
from .tasks import TASKS

BOOTSTRAPS = 2000


def ranks(values: np.ndarray) -> np.ndarray:
    """1 is the best. Each row of a 2D array is ranked separately."""
    return (-values).argsort(-1).argsort(-1).astype(float) + 1


def rank_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Correlation of ranks along the last axis, batched over any leading axes."""
    ra, rb = ranks(a), ranks(b)
    ra -= ra.mean(-1, keepdims=True)
    rb -= rb.mean(-1, keepdims=True)
    return (ra * rb).sum(-1) / np.sqrt((ra**2).sum(-1) * (rb**2).sum(-1))


def collect(rows: list[dict], grid: Grid) -> dict[str, np.ndarray]:
    """Per setting, the scores under the simulator and pooled over the learned models."""
    table = {}
    for condition in ("true", "learned"):
        table[condition] = {
            setting: np.array([
                r["score"] for r in rows if r["condition"] == condition and (r["horizon"], r["candidates"]) == setting
            ])
            for setting in grid.settings
        }
    return table


def agreement(table: dict, settings: list[tuple[int, int]], seed: int = 0) -> tuple[float, float, float]:
    """Rank correlation between conditions, with a 95% interval from resampling episodes."""
    rng = np.random.default_rng(seed)
    means = {c: np.array([table[c][s].mean() for s in settings]) for c in table}
    point = float(rank_correlation(means["true"], means["learned"]))
    resampled = {}
    for condition in table:
        draws = [
            rng.choice(table[condition][s], (BOOTSTRAPS, len(table[condition][s]))).mean(1) for s in settings
        ]
        resampled[condition] = np.stack(draws, 1)
    low, high = np.percentile(rank_correlation(resampled["true"], resampled["learned"]), [2.5, 97.5])
    return point, float(low), float(high)


def task_section(name: str, rows: list[dict], grid: Grid) -> list[str]:
    lines = [f"## {name}", ""]
    table = collect(rows, grid)
    models = sorted({r["model"] for r in rows if r["condition"] == "learned"})
    counts = {c: sum(len(v) for v in table[c].values()) for c in table}
    lines += [f"Episodes so far: simulator {counts['true']}, learned {counts['learned']} across models {models or 'none'}.", ""]

    lines += ["| H | J | Simulator mean ± SE | Rank | Learned mean ± SE | Rank |", "|---|---|---|---|---|---|"]
    rank_of = {}
    for condition in table:
        have = [s for s in grid.settings if len(table[condition][s])]
        order = ranks(np.array([table[condition][s].mean() for s in have])) if have else []
        rank_of[condition] = dict(zip(have, order))

    def cell(condition, setting):
        scores = table[condition][setting]
        if not len(scores):
            return "no data | -"
        return f"{scores.mean():.0f} ± {scores.std() / np.sqrt(len(scores)):.0f} | {rank_of[condition][setting]:.0f}"

    for setting in grid.settings:
        mark = " (paper)" if setting == (grid.paper_horizon, grid.paper_candidates) else ""
        lines.append(f"| {setting[0]} | {setting[1]}{mark} | {cell('true', setting)} | {cell('learned', setting)} |")
    lines.append("")

    both = [s for s in grid.settings if len(table["true"][s]) and len(table["learned"][s])]
    if len(both) >= 3:
        point, low, high = agreement(table, both)
        lines += [f"**Rank agreement over {len(both)} settings:** {point:.2f} (95% interval {low:.2f} to {high:.2f}, from episode variation).", ""]
    for condition, label in (("true", "Simulator"), ("learned", "Learned model")):
        for axis, values, index in (("horizon", grid.horizons, 0), ("candidates", grid.candidates, 1)):
            means = []
            for value in values:
                scores = [table[condition][s] for s in grid.settings if s[index] == value and len(table[condition][s])]
                if scores:
                    means.append(f"{value}: {np.concatenate(scores).mean():.0f}")
            if means:
                lines.append(f"- {label}, mean by {axis}: " + ", ".join(means))
    lines.append("")

    for model in sorted({r["model"] for r in rows if r["condition"] == "learned"}):
        scores = [r["score"] for r in rows if r["condition"] == "learned" and r["model"] == model]
        lines.append(f"- {model} grid mean: {np.mean(scores):.0f} over {len(scores)} episodes")
    lines.append("")

    for model in sorted({r["model"] for r in rows if r["condition"] == "screen"}):
        scores = np.array([r["score"] for r in rows if r["condition"] == "screen" and r["model"] == model])
        low = int((scores < 0.5 * np.median(scores)).sum())
        warning = f" **Warning: {low} scored under half the median, so this checkpoint may be a bad snapshot.**" if low else ""
        lines.append(f"- {model} screen at the paper's setting: {scores.mean():.0f} ± {scores.std():.0f} over {len(scores)} episodes.{warning}")
    lines.append("")
    return lines


def write_report(root: Path, grid: Grid) -> str:
    lines = ["# Planner study", ""]
    for name in TASKS:
        rows = load(root / name / "results.jsonl")
        if rows:
            lines += task_section(name, rows, grid)
    text = "\n".join(lines)
    root.mkdir(parents=True, exist_ok=True)
    (root / "report.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/study"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(write_report(args.root, SMOKE_GRID if args.smoke else GRID))


if __name__ == "__main__":
    main()
