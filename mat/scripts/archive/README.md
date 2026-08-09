# Archived script files

This directory keeps old one-off shell launchers and root-level logs that are no
longer part of the current YAML-based experiment workflow.

- `legacy_shell/`: previous direct `train_*.sh`, `run_smac.sh`, and test/cleanup shell scripts.
- `logs/`: old root-level stdout/stderr/tmux logs.

Current launch entry points should stay directly under `mat/scripts/`, especially
`run_yaml.py`, `run_yaml_slurm.sh`, and `submit_yaml_array.sh`.
