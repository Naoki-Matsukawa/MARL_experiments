#!/usr/bin/env python
"""Run training commands from a YAML experiment file.

This is intentionally a thin launcher: it keeps the existing train/*.py entry
points unchanged and only translates YAML parameters into CLI arguments.
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


class SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def format_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format_map(SafeFormatDict(context))
    if isinstance(value, list):
        return [format_value(v, context) for v in value]
    if isinstance(value, dict):
        return {k: format_value(v, context) for k, v in value.items()}
    return value


def expand_matrix(matrix: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not matrix:
        return [{}]
    keys = list(matrix.keys())
    value_lists = []
    for key in keys:
        values = matrix[key]
        if not isinstance(values, list):
            raise ValueError(f"matrix.{key} must be a list")
        value_lists.append(values)
    return [dict(zip(keys, values)) for values in itertools.product(*value_lists)]


def get_base_config(config: dict[str, Any]) -> dict[str, Any]:
    base = config.get("base", {})
    defaults = config.get("defaults", {})
    if not isinstance(base, dict):
        raise ValueError("base must be a mapping")
    if not isinstance(defaults, dict):
        raise ValueError("defaults must be a mapping")
    return deep_merge(base, defaults)


def get_presets(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    presets = config.get("presets", {})
    if not isinstance(presets, dict):
        raise ValueError("presets must be a mapping")
    for name, preset in presets.items():
        if not isinstance(preset, dict):
            raise ValueError(f"presets.{name} must be a mapping")
    return presets


def apply_presets(
    run: dict[str, Any],
    base: dict[str, Any],
    presets: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    merged = deep_merge(base, {})
    preset_names = run.get("use", [])
    if isinstance(preset_names, str):
        preset_names = [preset_names]
    if not isinstance(preset_names, list):
        raise ValueError(f"run {run.get('name', '<unnamed>')} has non-list use")

    for preset_name in preset_names:
        if preset_name not in presets:
            raise ValueError(f"unknown preset: {preset_name}")
        merged = deep_merge(merged, presets[preset_name])

    run_without_use = {k: v for k, v in run.items() if k != "use"}
    return deep_merge(merged, run_without_use)


def iter_configured_runs(config: dict[str, Any], selected: set[str] | None) -> list[dict[str, Any]]:
    base = get_base_config(config)
    presets = get_presets(config)

    runs = config.get("runs")
    if runs is None:
        excluded = {"base", "defaults", "presets"}
        run = {k: v for k, v in config.items() if k not in excluded}
        runs = [run]
    if not isinstance(runs, list):
        raise ValueError("runs must be a list")

    configured_runs = []
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("each run must be a mapping")
        name = run.get("name")
        if selected is not None and name not in selected:
            continue
        configured_runs.append(apply_presets(run, base, presets))
    return configured_runs


def normalize_runs(config: dict[str, Any], selected: set[str] | None = None) -> list[dict[str, Any]]:
    configured_runs = iter_configured_runs(config, selected)
    expanded = []
    for run in configured_runs:
        seeds = run.pop("seeds", None)
        if seeds is None:
            seeds = [run.get("args", {}).get("seed")]
        if not isinstance(seeds, list):
            seeds = [seeds]

        matrix_items = expand_matrix(run.pop("matrix", None))
        for matrix_args in matrix_items:
            for seed in seeds:
                item = deep_merge(run, {})
                item_args = dict(item.get("args", {}))
                item_args.update(matrix_args)
                if seed is not None:
                    item_args["seed"] = seed
                item["args"] = item_args
                expanded.append(item)
    return expanded


def list_runs(config: dict[str, Any]) -> None:
    runs = config.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError("runs must be a list")
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("each run must be a mapping")
        name = run.get("name", "<unnamed>")
        preset_names = run.get("use", [])
        if isinstance(preset_names, str):
            preset_names = [preset_names]
        preset_text = ", ".join(preset_names) if preset_names else "-"
        print(f"{name}\tuse=[{preset_text}]")


def cli_args(args_dict: dict[str, Any]) -> list[str]:
    cli: list[str] = []
    for key, value in args_dict.items():
        if value is None:
            continue
        flag = "--" + key.replace("_", "-") if key.startswith("_") else "--" + key
        if isinstance(value, bool):
            if value:
                cli.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                continue
            cli.append(flag)
            cli.extend(str(v) for v in value)
            continue
        cli.extend([flag, str(value)])
    return cli


def resolve_cwd(value: str | None) -> Path:
    if value is None:
        return SCRIPT_DIR
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def auto_gpu() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,index", "--format=csv,nounits,noheader"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "0"

    best_index = "0"
    best_memory = -1
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            memory = int(parts[0])
        except ValueError:
            continue
        if memory > best_memory:
            best_memory = memory
            best_index = parts[1]
    return best_index


def build_command(run: dict[str, Any]) -> tuple[list[str], Path, dict[str, str]]:
    args_dict = run.get("args", {})
    if not isinstance(args_dict, dict):
        raise ValueError("args must be a mapping")

    context = dict(args_dict)
    name = run.get("name")
    if name is not None:
        context["name"] = name

    formatted_run = format_value(run, context)
    args_dict = formatted_run.get("args", {})
    python = formatted_run.get("python", "python")
    script = formatted_run.get("script")
    if script is None:
        raise ValueError("script is required")

    command = [python, script] + cli_args(args_dict)
    cwd = resolve_cwd(formatted_run.get("cwd"))

    env = os.environ.copy()
    env_updates = formatted_run.get("env", {})
    if env_updates is None:
        env_updates = {}
    if not isinstance(env_updates, dict):
        raise ValueError("env must be a mapping")
    env.update({str(k): str(v) for k, v in env_updates.items()})

    gpu = formatted_run.get("cuda_visible_devices")
    if gpu == "auto":
        gpu = auto_gpu()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    return command, cwd, env


def print_command(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    gpu = env.get("CUDA_VISIBLE_DEVICES")
    prefix = f"CUDA_VISIBLE_DEVICES={gpu} " if gpu is not None else ""
    print(f"(cd {cwd} && {prefix}{' '.join(command)})")


def run_commands(runs: list[dict[str, Any]], dry_run: bool) -> int:
    processes: list[subprocess.Popen[Any]] = []
    for run in runs:
        command, cwd, env = build_command(run)
        print_command(command, cwd, env)
        if dry_run:
            continue
        if run.get("parallel", True):
            processes.append(subprocess.Popen(command, cwd=str(cwd), env=env))
        else:
            completed = subprocess.run(command, cwd=str(cwd), env=env)
            if completed.returncode != 0:
                return completed.returncode

    exit_code = 0
    for process in processes:
        exit_code = max(exit_code, process.wait())
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run mat/scripts experiments from YAML")
    parser.add_argument("config", type=Path, help="Path to an experiment YAML file")
    parser.add_argument("--run", action="append", dest="run_names", help="Run only the named run. Can be repeated")
    parser.add_argument("--list", action="store_true", help="List runs and exit")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    if args.list:
        list_runs(config)
        return 0

    selected = set(args.run_names) if args.run_names else None
    runs = normalize_runs(config, selected)
    if selected and not runs:
        print(f"No runs matched: {', '.join(sorted(selected))}", file=sys.stderr)
        return 1
    return run_commands(runs, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
