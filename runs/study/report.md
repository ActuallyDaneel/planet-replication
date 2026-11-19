# Planner study

## cheetah_run

Episodes so far: simulator 120, learned 360 across models ['seed0', 'seed1', 'seed2'].

| H | J | Simulator mean ± SE | Rank | Learned mean ± SE | Rank |
|---|---|---|---|---|---|
| 6 | 100 | 597 ± 20 | 11 | 326 ± 29 | 12 |
| 6 | 300 | 654 ± 20 | 9 | 340 ± 28 | 11 |
| 6 | 1000 | 565 ± 27 | 12 | 383 ± 27 | 10 |
| 8 | 100 | 827 ± 16 | 3 | 514 ± 25 | 9 |
| 8 | 300 | 874 ± 11 | 2 | 573 ± 22 | 6 |
| 8 | 1000 | 893 ± 4 | 1 | 545 ± 28 | 8 |
| 12 | 100 | 713 ± 5 | 6 | 582 ± 4 | 5 |
| 12 | 300 | 763 ± 4 | 5 | 654 ± 4 | 2 |
| 12 | 1000 (paper) | 773 ± 5 | 4 | 673 ± 3 | 1 |
| 14 | 100 | 648 ± 7 | 10 | 554 ± 3 | 7 |
| 14 | 300 | 710 ± 4 | 8 | 607 ± 11 | 4 |
| 14 | 1000 | 712 ± 6 | 7 | 640 ± 5 | 3 |

**Rank agreement over 12 settings:** 0.41 (95% interval 0.35 to 0.58, from episode variation).

- Simulator, mean by horizon: 6: 605, 8: 864, 12: 750, 14: 690
- Simulator, mean by candidates: 100: 696, 300: 750, 1000: 736
- Learned model, mean by horizon: 6: 350, 8: 544, 12: 636, 14: 600
- Learned model, mean by candidates: 100: 494, 300: 543, 1000: 560

- seed0 grid mean: 557 over 120 episodes
- seed1 grid mean: 527 over 120 episodes
- seed2 grid mean: 514 over 120 episodes

- seed0 screen at the paper's setting: 690 ± 17 over 30 episodes.
- seed1 screen at the paper's setting: 663 ± 17 over 30 episodes.
- seed2 screen at the paper's setting: 671 ± 22 over 30 episodes.

## cartpole_swingup

Episodes so far: simulator 120, learned 360 across models ['seed0', 'seed1', 'seed2'].

| H | J | Simulator mean ± SE | Rank | Learned mean ± SE | Rank |
|---|---|---|---|---|---|
| 6 | 100 | 397 ± 8 | 12 | 370 ± 12 | 10 |
| 6 | 300 | 399 ± 4 | 11 | 354 ± 12 | 11 |
| 6 | 1000 | 414 ± 6 | 10 | 352 ± 13 | 12 |
| 8 | 100 | 486 ± 2 | 9 | 383 ± 6 | 7 |
| 8 | 300 | 486 ± 1 | 8 | 374 ± 6 | 8 |
| 8 | 1000 | 487 ± 1 | 7 | 371 ± 4 | 9 |
| 12 | 100 | 831 ± 2 | 6 | 765 ± 16 | 6 |
| 12 | 300 | 841 ± 6 | 4 | 797 ± 10 | 4 |
| 12 | 1000 (paper) | 850 ± 1 | 2 | 822 ± 6 | 1 |
| 14 | 100 | 834 ± 4 | 5 | 782 ± 7 | 5 |
| 14 | 300 | 845 ± 2 | 3 | 807 ± 5 | 3 |
| 14 | 1000 | 853 ± 1 | 1 | 808 ± 5 | 2 |

**Rank agreement over 12 settings:** 0.94 (95% interval 0.81 to 0.97, from episode variation).

- Simulator, mean by horizon: 6: 404, 8: 486, 12: 840, 14: 844
- Simulator, mean by candidates: 100: 637, 300: 643, 1000: 651
- Learned model, mean by horizon: 6: 359, 8: 376, 12: 795, 14: 799
- Learned model, mean by candidates: 100: 575, 300: 583, 1000: 588

- seed0 grid mean: 588 over 120 episodes
- seed1 grid mean: 580 over 120 episodes
- seed2 grid mean: 578 over 120 episodes

- seed0 screen at the paper's setting: 826 ± 61 over 30 episodes.
- seed1 screen at the paper's setting: 801 ± 36 over 30 episodes.
- seed2 screen at the paper's setting: 823 ± 46 over 30 episodes.
