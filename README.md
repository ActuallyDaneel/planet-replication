# PlaNet replication: cheetah-run

A PyTorch port of [PlaNet](https://arxiv.org/abs/1811.04551) (Hafner et al., 2019), optimized for a single
RTX 4080 Super on native Windows. The [official code](https://github.com/google-research/planet) uses
TensorFlow 1.x and CUDA 10, which cannot run on RTX 40-series cards, so this is a port and not a patched copy.

**Target:** the paper reports **662** on cheetah-run after 1,000 episodes (mean of 5 seeds, 10 trajectories each).
One seed can land well away from that number.

## Running it

Check the GPU path and measure this machine's speed (about a minute in total):

```bash
uv run python -m planet.train --smoke --logdir runs/smoke
```

```bash
uv run python -m planet.benchmark
```

Train:

```bash
uv run python -m planet.train
```

- **Stopping:** press Ctrl+C whenever you need the GPU back. It finishes the current episode (under 30 seconds), saves
  and exits, and the same command resumes from there. A second Ctrl+C or closing the window stops at once and resumes
  from the last periodic checkpoint, losing at most 10 episodes.
- **Output:** `runs/cheetah_run/` holds `metrics.csv`, `checkpoint.pt`, `episodes/` and, at the end, `final_eval.json`.
- **Sleep:** turn off Windows sleep for the duration, because it pauses the run.
- **Options:** every value in `planet/config.py` is a flag, such as `--seed 1 --logdir runs/seed1`.

## Trained models

Six trained models are attached to the `models` release, not committed: cheetah-run and cartpole-swingup, seeds 0-2,
each trained for 1,000 episodes. The zip is 119 MB. Download it and unzip at the project root:

```bash
gh release download models --repo ActuallyDaneel/planet-replication --pattern planet_models.zip
```

```bash
unzip planet_models.zip
```

This gives `models/<task>_seed<N>/`, each with `checkpoint.pt` (weights only, 21 MB, no optimizer state),
`config.json`, `metrics.csv` and `final_eval.json`. Score one at the paper's planner setting over 30 episodes:

```bash
uv run python -m planet.study.evaluate --task cheetah_run --condition screen --logdir models/cheetah_run_seed0 --model seed0 --results check.jsonl
```

Loading one in Python:

```python
import json, torch
from planet.agent import Agent, configure_torch
from planet.config import Config
from planet.env import Env

folder = "models/cheetah_run_seed0"
cfg = Config(**json.load(open(f"{folder}/config.json")))
device = configure_torch(cfg)
env = Env(cfg.domain, cfg.task, cfg.action_repeat, cfg.episode_steps, 0)
agent = Agent(cfg, env.action_dim, device)
agent.model.load_state_dict(torch.load(f"{folder}/checkpoint.pt", map_location=device)["model"])
print(agent.run_episode(env, 0.0).score)
```

These checkpoints cannot resume training, because the optimizer state is left out. To rerun the study's evaluations
on them, copy each `checkpoint.pt` into the matching run folder under `runs/`. Cheetah seed 0 goes in
`runs/cheetah_run`, and the others go in `runs/study/<task>/seed<N>`.

## Runtime estimate

**About 5 hours for 1,000 episodes, with the GPU otherwise idle.** The benchmark projects 4.7 hours; the rest is an
allowance for saving episodes and sustained load, which a one-minute benchmark does not capture. The actual run took
5.3 hours, with other programs sharing the CPU. The paper took 10 to 20 hours on a V100.

Measured with `planet.benchmark` on this RTX 4080 Super:

| Work | Count | Cost each | Total |
|---|---|---|---|
| Model updates | 99,500 | 104.6 ms | 2.9 h |
| Planning, train and eval | 276,250 | 21.8 ms | 1.7 h |
| Physics and rendering | 276,250 | 1.64 ms | 0.13 h |

| Variant | Update | Plan step | Projected total |
|---|---|---|---|
| Plain Python | 110.0 ms | 32.6 ms | 5.7 h |
| TorchScript (default) | 104.6 ms | 21.8 ms | 4.7 h |
| TorchScript + `--amp` | 89.5 ms | 21.8 ms | 4.3 h |

The same code on the CPU alone measured about 115 hours, so the GPU is required.

The benchmark reported 9.8 GiB of peak VRAM reserved, including the 3.1 GB replay. That figure spans all three
variants, so a single run may reserve less. Disk use is about 3 GB and RAM use is modest. If another program is using
the GPU, expect the run to slow down. If the replay does not fit in VRAM, it automatically moves to RAM.

## Optimizations

None of these change the algorithm's hyperparameters.

| Change | Why it helps |
|---|---|
| Replay stored in VRAM | Sampling no longer copies 30 MB from CPU to GPU on every update |
| TorchScript RSSM and CEM loops | Measured on GPU: planning 33% faster, updates 5% faster, about 1 hour saved in total |
| No `torch.distributions` objects | Closed-form KL and sampling avoid constructing objects in the hot loops |
| Reward head batched over the whole rollout | One call per CEM iteration instead of one per horizon step |
| Fused Adam, `cudnn.benchmark`, deferred metric reads | Faster optimizer step and no forced GPU sync on each update |
| One CPU math thread when using CUDA | Stops idle math threads competing with MuJoCo |

`--amp` runs the CNNs in bfloat16 and saves about 0.4 hours (8%). It is off by default because it is not in the paper
and its effect on results is untested, which is not worth risking for that saving.

## Choices where the sources differ

The official repository produced the published results, so it is the default where it disagrees with the paper's text.

| Setting | Paper text | Official code (used) | Flag |
|---|---|---|---|
| Reward head | 2 layers × 200 | 3 layers × 300 | `--reward-layers`, `--reward-units` |
| Reward loss scale | Not stated | 10 | `--reward-scale` |
| RSSM hidden layers | 2 × 200 | 1 × 200 | None |

Known remaining differences from the original:

- MuJoCo and `dm_control` are much newer, so task physics may differ slightly.
- Replay chunks are sampled uniformly, as in the paper's Algorithm 1. The official `recent_loader` is close to
  uniform at one new episode per collection, so this is unlikely to matter.
- PyTorch's `GRUCell` is initialized differently from TensorFlow's `GRUBlockCell`.

Latent overshooting is left out, as the paper's Appendix A says it was not necessary.

## Verification status

Tested on CPU: 11 unit tests (`uv run --with pytest python -m pytest tests -q`), an end-to-end run, and stop and resume.
Tested on GPU: a small end-to-end run, the full-settings benchmark, a full run to 1,000 episodes, and stop and resume.

**Seed 0 finished at 686.1 ± 18.5 over 10 episodes, against the paper's 662.** One seed cannot confirm the replication:
the paper's number averages 5 seeds, and this run's spread only reflects episodes within one seed.

Seed 0 against the paper. Episodes 100-750 use the Figure 4 median over 5 seeds, read by eye from the plot. Episode 1,000
uses the exact mean from Table 1.

| Episode | Paper | Mean over 10 episodes | Median |
|---|---|---|---|
| 100 | about 400 | 166 ± 73 | not computed |
| 250 | about 540 | 410 ± 180 | 521 |
| 500 | about 580 | 613.5 ± 8.6 | 614 |
| 750 | about 635 | 393.9 ± 218.3 | 546 |
| 1,000 | 662 | 686.1 ± 18.5 | 688 |

**The run lagged early, reached the paper's curve by episode 500 and finished above 662, but single snapshots of the
weights can score far below the trend, as episode 750 did.** At episode 250, 3 of 10 evaluation episodes scored under
150. At episode 500, all 10 scored 598-627. Training returns stayed flat until about episode 175, then rose.

Each checkpoint was re-tested with 30 fresh episodes, and with snapshots taken after further updates:

| Weights evaluated | Episodes | Scoring under 300 | Mean |
|---|---|---|---|
| Episode 750 checkpoint (final evaluation plus 30 fresh episodes) | 40 | 15 | 407 |
| 100-500 updates after episode 750, no new data, not saved | 50 | 0 | 646 |
| Periodic evaluations, episodes 701-750 (different weights each) | 5 | 0 | 642 |
| Episode 1,000 checkpoint (final evaluation plus 30 fresh episodes) | 40 | 0 | 688 |
| 100-500 updates after episode 1,000, no new data, not saved | 50 | 0 | 679 |
| Periodic evaluations, episodes 951-1,000 (different weights each) | 5 | 0 | 667 |

The episode 750 score comes from one bad set of weights and does not represent the run around it. In the one failed
episode viewed, the cheetah flipped onto its back within about 50 steps and stayed there, as seen in the training
episode viewed at episode 100. Why those weights do this is not established. The further updates were discarded, so the
checkpoint holds the episode 1,000 weights exactly. A score from one snapshot should not be read alone, and the
episode 500 result could not be re-tested this way.

Flipping is not fully gone. With exploration noise on, 8% of training episodes from 951 to 1,000 scored under 300, compared
with 22% from 451 to 500 and 2% from 801 to 850. An audit against the official code at episode 100 found no bug. The paper's
own result with the true simulator is 656, and newer MuJoCo physics may shift scores, so being above 662 should not be
read as beating the paper. Episodes 1-1,000 took 5.3 hours.

## Planner study (extension)

Tests whether planner settings rank the same when CEM imagines in the true simulator and in the learned model, on
cheetah-run and cartpole-swingup: horizon 6, 8, 12, 14 × candidates 100, 300, 1,000, with 10 CEM iterations and 10
episodes per setting, using 3 trained models per task. It refits to the best 10% instead of a fixed 100, because 100 of
100 candidates would select nothing.

### Results

All 20 stages finished on 9 November 2026, 34.6 hours after it started (22:10 on 7 November to 08:49 on 9 November), including two pauses with Ctrl+C. Full tables are in
`runs/study/report.md`.

**Settings tuned in the true simulator carried over to the learned model on cartpole, but not on cheetah. The
paper's setting, horizon 12 with 1,000 candidates, scored best with the learned model on both tasks.**

| | Cheetah | Cartpole |
|---|---|---|
| Best in the true simulator | H 8, J 1,000: 893 | H 14, J 1,000: 853 |
| That setting with the learned model | 545, ranked 8th of 12 | 808, ranked 2nd of 12 |
| Best with the learned model | H 12, J 1,000: 673 | H 12, J 1,000: 822 |
| Loss from using the simulator's pick | 19% | 2% |
| Rank agreement over 12 settings (95% interval) | 0.41 (0.35 to 0.58) | 0.94 (0.81 to 0.97) |
| Paper's setting: rank in simulator, rank with learned model | 4th, 1st | 2nd, 1st |

Mean score by horizon, averaged over candidate counts:

| Horizon | Cheetah, simulator | Cheetah, learned | Cartpole, simulator | Cartpole, learned |
|---|---|---|---|---|
| 6 | 605 | 350 | 404 | 359 |
| 8 | 864 | 544 | 486 | 376 |
| 12 | 750 | 636 | 840 | 795 |
| 14 | 690 | 600 | 844 | 799 |

- **Candidates:** more candidates helped in every task and condition.
- **Horizon on cheetah:** the best horizon moves from 8 in the simulator to 12 with the learned model. With the learned
  model, horizon 8 also varies widely between episodes, with standard errors of 22-28 against 3-5 at horizon 12.
  Flipping, as seen in training, would fit this, but replays were not checked.
- **The paper's figure:** Figure 12 puts the best horizon near 8, yet the paper uses 12. That figure matches the true
  simulator here, and the learned model is what favours 12.
- **Trained models:** each scored close to the paper at the paper's setting, over 30 episodes. Cheetah seeds 0-2
  scored 690, 663 and 671, against 662. Cartpole seeds 0-2 scored 826, 801 and 823, against 821. Cartpole training
  took about 4 hours per seed, as estimated.

Caveats:

- Only 3 models per task. The rank intervals reflect variation between episodes, not between models.
- Cheetah in the true simulator scores well above the paper, 773 at the paper's setting against 656, and the reason is
  unknown. Rankings are unaffected, because both conditions use the same task.
- Rewards in the true simulator were tested against the real environment, but only one simulator was used.

### Running it

Run everything left to do, which is about 26 hours from the start, mostly training:

```bash
uv run python -m planet.study
```

Show each stage's progress and the estimated time left:

```bash
uv run python -m planet.study --list
```

Run only part of it, by stage name or by task and seed:

```bash
uv run python -m planet.study --only "cheetah_run true" "cartpole_swingup true"
```

```bash
uv run python -m planet.study --tasks cheetah_run --seeds 0 1
```

Update the report from whatever has finished so far:

```bash
uv run python -m planet.study.report
```

**Stopping:** press Ctrl+C at any point, and run the same command to resume, as often as you like. Each stage works out
what is left from its files. Training saves after the current episode, and a grid keeps finished episodes and redoes the
one in progress, which can be up to about 2 minutes on the true simulator. A second copy refuses to start. A stage that
fails by itself is retried, then only the stages that need it are skipped.

**GPU:** close other GPU-heavy programs first. Training needs about 7 GB of VRAM. If VRAM overflows, Windows swaps it and
training crawls without any error, so the log warns when too little is free. The true-simulator stages use only the CPU.

**Output:** under `runs/study/`, each task has `results.jsonl` and its trained seeds, with `report.md` and `study.log` at
the top level. Cheetah's seed 0 reuses `runs/cheetah_run`, read-only.

**Status:** the full study ran to completion with all 20 stages succeeding. It was stopped with Ctrl+C twice, once
during full-size cheetah training, and resumed each time. Each simulator's rewards are tested against the real
environment.

In the true simulator at the paper's setting, cartpole scored 850 over 10 episodes, matching Table 1's 850, and cheetah
scored 773, against 656. That cheetah gap is unexplained, and newer MuJoCo physics is one possibility. If physics made
cheetah easier, it would also affect the 686 against 662 in the verification section.

## Layout

| File | Contents |
|---|---|
| `planet/config.py` | Hyperparameters and CLI flags |
| `planet/models.py` | Encoder, decoder, RSSM, CEM planner |
| `planet/agent.py` | Loss, update step, episode collection |
| `planet/replay.py` | Pre-allocated episode storage |
| `planet/env.py` | DM Control wrapper with pixels and action repeat |
| `planet/train.py` | Training loop, checkpoints, metrics |
| `planet/benchmark.py` | Speed measurement and runtime projection |
| `planet/interrupt.py` | Ctrl+C handling |
| `planet/study/pipeline.py` | The planner study's ordered, restartable stages |
| `planet/study/tasks.py` | Study tasks and their rewards from simulator rollouts |
| `planet/study/sim_planner.py` | CEM imagining in the true simulator |
| `planet/study/results.py` | Settings grid and append-only results |
| `planet/study/evaluate.py` | Runs one grid or model screen |
| `planet/study/report.py` | Rankings and comparisons from the results |
