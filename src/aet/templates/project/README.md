# aet project

This directory was scaffolded by `aet init`.

## Quickstart

```bash
# Initialise a run
aet init-run --suite default --seed 1

# Validate the run
aet validate <run_path>

# Compare results across seeds/methods
aet compare
```

## Layout

```
configs/        Project and budget configuration
runs/           One sub-directory per run (gitignored by default)
reports/        Comparison tables and metrics CSV
```

Edit `configs/project.yaml` to set your project name, suite, and tracking mode.
