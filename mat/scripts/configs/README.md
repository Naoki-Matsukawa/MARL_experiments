# YAML experiment configs

The recommended entry point is the single experiment registry:

```bash
python run_yaml.py configs/experiments.yaml --list
python run_yaml.py configs/experiments.yaml --run smac_single --dry-run
python run_yaml.py configs/experiments.yaml --run smac_single
```

The file is organized as:

```yaml
base:       # defaults shared by most runs
presets:   # reusable blocks such as smac_single, smac_multi, mat, pld
runs:      # named executable experiments
```

Each run can combine presets:

```yaml
runs:
  - name: partial_obs
    use: [smac_single, mat, partial_obs, enemy_group_jitter]
    seeds: [1, 2, 3]
    matrix:
      sight_range: [9, 7, 5]
      enemy_position_jitter: [0.0, 1.5]
    args:
      map_name: 3s5z_vs_3s6z
      experiment_name: "partial_sight{sight_range}_ejitter{enemy_position_jitter}"
```

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

Use `seeds` for seed loops and `matrix` for simple sweeps. String values can
reference generated arguments:

```yaml
experiment_name: "partial_{map_name}_sight{sight_range}"
```

For SMAC, the registry uses headless-friendly visualization settings. The
current SMAC environment does not produce RGB frames from `render()`, so GIF
debug rendering should be implemented as a top-down renderer from raw unit
positions rather than as a StarCraft II screen capture.

Planned headless debug render settings live in YAML as normal args:

```yaml
use: [smac_single, mat, enemy_group_jitter, headless_debug_render]
args:
  save_debug_render: true
  debug_render_format: gif
  debug_render_show_sight: true
```

Enemy spawn randomization is configured in the same way:

```yaml
use: [enemy_group_jitter]
args:
  randomize_enemy_position: true
  enemy_position_jitter: 1.5
  enemy_position_jitter_mode: group
```

The older per-script YAML files are kept as smaller examples:

- `smac_single.yaml`
- `smac_multi.yaml`
- `smac_few_shot.yaml`
- `smac_pld.yaml`
- `smac_distillation.yaml`
