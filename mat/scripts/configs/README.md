# YAML experiment configs

Use one YAML file per experiment family. Each case file is self-contained:
launcher settings, method settings, map settings, and sweep settings live in
the same file.

Run examples from the repository root:

```bash
uv run python mat/scripts/run_yaml.py mat/scripts/configs/smac/smac_single.yaml --dry-run
uv run python mat/scripts/run_yaml.py mat/scripts/configs/smac/partial_enemy_jitter.yaml --dry-run
uv run python mat/scripts/run_yaml.py mat/scripts/configs/smac/smac_distillation_partial_enemy_jitter.yaml --dry-run
```

Main student-teacher random/partial experiment:

```bash
uv run python mat/scripts/run_yaml.py mat/scripts/configs/smac/smac_distillation_partial_enemy_jitter.yaml
```

Submit expanded runs as separate Slurm array tasks:

```bash
CONFIG_PATH=mat/scripts/configs/smac/smac_distillation_partial_enemy_jitter.yaml \
mat/scripts/submit_yaml_array.sh
```

Slurm array settings can live in the YAML:

```yaml
resources:
  num_gpus: 3
  gpus_per_run: 1

slurm:
  script: mat/scripts/run_yaml_slurm.sh
```

`resources` applies to both local YAML execution and Slurm submission. With
`cuda_visible_devices: auto`, the launcher assigns at most one active run per
GPU from this pool. Slurm submission uses the same values to submit one expanded
run per array task with at most `num_gpus` concurrent tasks.

Available experiment families:

- `smac/*.yaml`: SMAC training, few-shot, PLD, CDBD, distillation, and
  partial-observation experiments.
- `vmas/*.yaml`: VMAS navigation, sampling, and discovery experiments.
- `marbler/*.yaml`: MARBLER and Robotarium experiments.
- `_ci_smoke/*.yaml`: short CI smoke-test manifests; these are not intended as
  research experiment templates.

`smac/experiments.yaml` is kept as the older single-registry style, but new
runs should prefer the split case files.

YAML keys under `args` are translated directly to CLI flags. For example:

```yaml
args:
  map_name: 3m
  use_eval: true
  train_maps: [3m, MMM, 3s5z]
```

becomes:

```bash
--map_name 3m --use_eval --train_maps 3m MMM 3s5z
```

Use `seeds` for seed loops and `matrix` for sweeps. String values can reference
generated arguments:

```yaml
experiment_name: "partial_sight{sight_range}_ejitter{enemy_position_jitter}"
```
