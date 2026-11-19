"""The whole study as ordered stages. Stop and restart it as often as you like.

    uv run python -m planet.study                              Runs everything left to do
    uv run python -m planet.study --list                       Shows progress and time left
    uv run python -m planet.study --tasks cheetah_run --seeds 0 1
    uv run python -m planet.study --only "cheetah_run true"    Runs stages by name

Ctrl+C saves and stops. Rerunning continues from there, because each stage works out what
is left from its files on disk. A stage that fails by itself is retried, then the queue
moves on and skips only the stages that need it. While this runs, the PC is asked not to
idle-sleep; the request ends with the process and changes no settings.
"""
import argparse
import ctypes
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .results import GRID, SMOKE_GRID, Grid, planned, remaining
from .tasks import TASKS, Task

RETRIES = 3
RETRY_WAIT_SECONDS = 300
SIMULATOR_HOURS = 1.25  # Measured physics speed on 12 threads, plus planning overhead.
SEEDS = (0, 1, 2)
MIN_FREE_VRAM_MIB = 7000  # Training measured 6.9 GB on cartpole.

stop_requested = False


@dataclass
class Stage:
    name: str
    arguments: list[str]
    progress: Callable[[], tuple[int, int]]  # Units finished and units in total.
    hours: float  # Estimate for the whole stage.
    needs: list[str] = field(default_factory=list)
    uses_gpu: bool = True

    @property
    def done(self) -> bool:
        finished, total = self.progress()
        return finished >= total

    @property
    def hours_left(self) -> float:
        finished, total = self.progress()
        return self.hours * (total - finished) / total


def run_dir(root: Path, task: Task, seed: int) -> Path:
    """Reuses the finished reproduction run as cheetah's seed 0."""
    if task.name == "cheetah_run" and seed == 0 and root == Path("runs/study"):
        return Path("runs/cheetah_run")
    return root / task.name / f"seed{seed}"


def train_progress(logdir: Path, total: int) -> Callable[[], tuple[int, int]]:
    def progress() -> tuple[int, int]:
        if (logdir / "final_eval.json").exists():
            return total, total
        metrics = logdir / "metrics.csv"
        lines = metrics.read_text().splitlines() if metrics.exists() else []
        if len(lines) < 2:
            return 0, total
        # Only finished once the final evaluation is written.
        return min(int(lines[-1].split(",")[0]), total - 1), total
    return progress


def grid_progress(path: Path, condition: str, model: str, grid: Grid) -> Callable[[], tuple[int, int]]:
    total = len(planned(condition, grid))
    return lambda: (total - len(remaining(path, condition, model, grid)), total)


def build_stages(root: Path, tasks: list[str], seeds: list[int], smoke: bool) -> list[Stage]:
    """Quick results first: each task's first model and simulator grid, then more seeds."""
    grid = SMOKE_GRID if smoke else GRID
    flag = ["--smoke"] if smoke else []
    stages = []
    for name in tasks:
        task = TASKS[name]
        results = root / name / "results.jsonl"
        evaluate = ["-m", "planet.study.evaluate", "--task", name, "--results", str(results), *flag]
        for position, seed in enumerate(seeds):
            logdir = run_dir(root, task, seed)
            model = f"seed{seed}"
            train = f"{name} train {model}"
            # Learned episodes take time per planning step, and higher action repeats have fewer.
            speed = 4 / task.action_repeat
            stages.append(Stage(
                train,
                ["-m", "planet.train", "--domain", task.domain, "--task", task.task, "--action-repeat", str(task.action_repeat),
                 "--seed", str(seed), "--logdir", str(logdir), *flag],
                train_progress(logdir, 4 if smoke else 1000), task.train_hours,
            ))
            model_args = [*evaluate, "--logdir", str(logdir), "--model", model]
            stages.append(Stage(
                f"{name} screen {model}", [*model_args, "--condition", "screen"],
                grid_progress(results, "screen", model, grid), 0.05 * speed, [train],
            ))
            stages.append(Stage(
                f"{name} learned {model}", [*model_args, "--condition", "learned"],
                grid_progress(results, "learned", model, grid), 0.1 * speed, [train],
            ))
            if position == 0:
                stages.append(Stage(
                    f"{name} true", [*evaluate, "--condition", "true"],
                    grid_progress(results, "true", "simulator", grid), SIMULATOR_HOURS, uses_gpu=False,
                ))
    return stages


def request_stop(*_) -> None:
    # The console sends Ctrl+C to the running stage too, and the stage does the saving.
    # So this only records the request and keeps waiting for the stage to exit.
    global stop_requested
    stop_requested = True


class Log:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("a", encoding="utf-8")

    def write(self, text: str) -> None:
        print(text, end="", flush=True)
        self.file.write(text)
        self.file.flush()

    def event(self, message: str) -> None:
        self.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")


def run_child(arguments: list[str], log: Log) -> int:
    """Runs one stage, copying its output to the console and the log."""
    process = subprocess.Popen(
        [sys.executable, *arguments], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8",
        errors="replace", env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8"),
    )
    for line in process.stdout:
        log.write(line)
    return process.wait()


def warn_if_gpu_is_short(log: Log) -> None:
    """When VRAM overflows, Windows swaps it and training crawls without raising any error."""
    try:
        free = int(subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        ).stdout.split()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return
    if free < MIN_FREE_VRAM_MIB:
        log.event(f"WARNING only {free} MiB of VRAM is free and training can use about {MIN_FREE_VRAM_MIB}. "
                  "If another program is using the GPU, this stage may run far slower than estimated.")


def run_stage(stage: Stage, log: Log) -> bool:
    if stage.uses_gpu:
        warn_if_gpu_is_short(log)
    for attempt in range(1, RETRIES + 1):
        log.event(f"START {stage.name}, attempt {attempt}")
        code = run_child(stage.arguments, log)
        if stop_requested:
            return False
        if code == 0 and stage.done:
            log.event(f"DONE {stage.name}")
            return True
        log.event(f"FAILED {stage.name}, exit code {code}")
        if attempt < RETRIES:
            deadline = time.monotonic() + RETRY_WAIT_SECONDS
            while time.monotonic() < deadline and not stop_requested:
                time.sleep(1)
            if stop_requested:
                return False
    return False


def show_status(stages: list[Stage]) -> None:
    for stage in stages:
        finished, total = stage.progress()
        state = "done" if stage.done else (f"{finished}/{total}" if finished else "waiting")
        print(f"  {stage.name:42s} {state:>12s}  {stage.hours_left:5.1f} h left")
    print(f"  {'Total':42s} {'':>12s}  {sum(stage.hours_left for stage in stages):5.1f} h left")


def hold_single_instance(root: Path):
    """Stops a second copy starting by accident. The OS drops the lock if this process dies."""
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / ".lock").open("a+")
    if os.name == "nt":
        import msvcrt
        try:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            sys.exit(f"The study is already running on {root}. Stop that one first.")
    return lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--only", nargs="+", default=None, help="Run only these stages, by the names --list shows.")
    parser.add_argument("--list", action="store_true", help="Show progress and exit.")
    parser.add_argument("--root", type=Path, default=Path("runs/study"))
    parser.add_argument("--smoke", action="store_true", help="Tiny version of everything, to test the pipeline.")
    args = parser.parse_args()

    stages = build_stages(args.root, args.tasks, args.seeds, args.smoke)
    if args.only:
        unknown = set(args.only) - {stage.name for stage in stages}
        if unknown:
            parser.error(f"unknown stages: {sorted(unknown)}. See --list.")
    if args.list:
        show_status(stages)
        return

    lock = hold_single_instance(args.root)  # noqa: F841  Held until this process exits.
    log = Log(args.root / "study.log")
    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, request_stop)
    if os.name == "nt":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

    by_name = {stage.name: stage for stage in stages}
    failed = []
    for stage in stages:
        if stop_requested:
            break
        if args.only and stage.name not in args.only:
            continue
        if stage.done:
            log.event(f"SKIP {stage.name}, already finished")
            continue
        missing = [need for need in stage.needs if not by_name[need].done]
        if missing:
            log.event(f"SKIP {stage.name}, needs {missing}")
            failed.append(stage.name)
            continue
        if not run_stage(stage, log) and not stop_requested:
            failed.append(stage.name)

    report = ["-m", "planet.study.report", "--root", str(args.root), *(["--smoke"] if args.smoke else [])]
    subprocess.run([sys.executable, *report], stdout=subprocess.DEVNULL, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    if stop_requested:
        log.event(f"STOPPED by Ctrl+C. Progress is saved, and {args.root / 'report.md'} covers it so far. Run the same command to resume.")
    else:
        log.event(f"FINISHED, {'not finished: ' + ', '.join(failed) if failed else 'all stages succeeded'}. See {args.root / 'report.md'}.")
