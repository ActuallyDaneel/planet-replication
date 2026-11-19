"""Do planner settings rank the same with the true simulator and the learned model?

Run with: uv run python -m planet.study   (see pipeline.py for options)

    tasks.py        The tasks, and their rewards computed from simulator rollouts
    sim_planner.py  CEM imagining in the true simulator
    results.py      The settings grid and the append-only results files
    evaluate.py     Runs one grid for one model, or for the simulator
    report.py       Rankings and comparisons from the results
    pipeline.py     The ordered, restartable stages that run all of the above
"""
