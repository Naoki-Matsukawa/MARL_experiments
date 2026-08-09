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
import time
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


class SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def load_yaml(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    if seen is None:
        seen = set()
    if path in seen:
        raise ValueError(f"circular YAML extends detected at {path}")
    seen.add(path)

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")

    extends = data.get("extends")
    if extends is None:
        return data

    if isinstance(extends, (str, Path)):
        extends = [extends]
    if not isinstance(extends, list):
        raise ValueError("extends must be a path or a list of paths")

    merged: dict[str, Any] = {}
    for parent in extends:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        merged = deep_merge(merged, load_yaml(parent_path, seen))

    child = {k: v for k, v in data.items() if k != "extends"}
    return deep_merge(merged, child)


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
        excluded = {"base", "defaults", "presets", "extends"}
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
    gpus = detect_gpu_pool()
    return gpus[0] if gpus else "0"


def detect_gpu_pool(respect_visible_devices: bool = True) -> list[str]:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if respect_visible_devices and visible_devices and visible_devices not in {"NoDevFiles", "-1"}:
        devices = [device.strip() for device in visible_devices.split(",") if device.strip()]
        if devices:
            return devices

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,index", "--format=csv,nounits,noheader"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ["0"]

    gpus: list[tuple[int, str]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            memory = int(parts[0])
        except ValueError:
            continue
        gpus.append((memory, parts[1]))
    if not gpus:
        return ["0"]
    return [index for _, index in sorted(gpus, reverse=True)]


def get_resource_config(config: dict[str, Any]) -> dict[str, Any]:
    resources = config.get("resources", {})
    if resources is None:
        return {}
    if not isinstance(resources, dict):
        raise ValueError("resources must be a mapping")
    return resources


def configured_gpu_pool(config: dict[str, Any], max_needed: int | None = None) -> list[str]:
    resources = get_resource_config(config)
    gpu_ids = resources.get("gpu_ids")
    if gpu_ids is not None:
        if not isinstance(gpu_ids, list):
            raise ValueError("resources.gpu_ids must be a list")
        gpu_pool = [str(gpu_id) for gpu_id in gpu_ids]
        return gpu_pool[:max_needed] if max_needed is not None else gpu_pool

    num_gpus = resources.get("num_gpus")
    if num_gpus is None:
        pool = detect_gpu_pool()
        return pool[:max_needed] if max_needed is not None else pool
    try:
        num_gpus = int(num_gpus)
    except (TypeError, ValueError) as exc:
        raise ValueError("resources.num_gpus must be an integer") from exc
    if num_gpus <= 0:
        raise ValueError("resources.num_gpus must be positive")
    if max_needed is not None:
        num_gpus = min(num_gpus, max_needed)

    respect_visible_devices = bool(
        os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_ARRAY_JOB_ID")
        or resources.get("respect_cuda_visible_devices", False)
    )
    pool = detect_gpu_pool(respect_visible_devices=respect_visible_devices)
    if len(pool) < num_gpus:
        raise RuntimeError(
            f"resources.num_gpus={num_gpus} but only {len(pool)} GPU(s) are visible: {pool}. "
            "Use resources.gpu_ids to choose explicit GPUs, or run where enough GPUs are visible."
        )
    return pool[:num_gpus]


def configured_startup_stagger_seconds(config: dict[str, Any]) -> float:
    resources = get_resource_config(config)
    value = resources.get("startup_stagger_seconds", 0)
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("resources.startup_stagger_seconds must be a number") from exc
    if seconds < 0:
        raise ValueError("resources.startup_stagger_seconds must be non-negative")
    return seconds


def configured_require_cuda(config: dict[str, Any]) -> bool:
    resources = get_resource_config(config)
    return bool(resources.get("require_cuda", False))


def build_command(run: dict[str, Any], gpu_override: str | None = None) -> tuple[list[str], Path, dict[str, str]]:
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
    if gpu_override is not None:
        gpu = gpu_override
    elif gpu == "auto":
        gpu = auto_gpu()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    return command, cwd, env


def assert_cuda_available(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    python = command[0]
    probe = (
        "import sys, torch; "
        "print('torch', torch.__version__); "
        "print('torch cuda', torch.version.cuda); "
        "print('cuda available', torch.cuda.is_available()); "
        "sys.exit(0 if torch.cuda.is_available() else 1)"
    )
    result = subprocess.run(
        [python, "-c", probe],
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        raise RuntimeError(
            "CUDA is required for this run, but PyTorch cannot use CUDA "
            f"with CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', 'unset')}."
        )


def print_command(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    gpu = env.get("CUDA_VISIBLE_DEVICES")
    prefix = f"CUDA_VISIBLE_DEVICES={gpu} " if gpu is not None else ""
    print(f"(cd {cwd} && {prefix}{' '.join(command)})")


def uses_auto_gpu(run: dict[str, Any]) -> bool:
    return run.get("cuda_visible_devices") == "auto"


def run_commands(
    runs: list[dict[str, Any]],
    dry_run: bool,
    gpu_pool: list[str] | None = None,
    startup_stagger_seconds: float = 0,
    require_cuda: bool = False,
) -> int:
    auto_pool = gpu_pool if gpu_pool is not None else []
    if any(uses_auto_gpu(run) for run in runs) and not auto_pool:
        auto_pool = detect_gpu_pool()
    if any(uses_auto_gpu(run) for run in runs):
        print(f"Auto GPU pool: {auto_pool}")
    auto_idx = 0

    processes: list[subprocess.Popen[Any]] = []
    for index, run in enumerate(runs):
        assigned_gpu = None
        if uses_auto_gpu(run):
            assigned_gpu = auto_pool[auto_idx % len(auto_pool)] if auto_pool else "0"
            auto_idx += 1

        command, cwd, env = build_command(run, gpu_override=assigned_gpu)
        print_command(command, cwd, env)
        if dry_run:
            continue
        if require_cuda:
            assert_cuda_available(command, cwd, env)
        processes.append(subprocess.Popen(command, cwd=str(cwd), env=env))
        if startup_stagger_seconds > 0 and index < len(runs) - 1:
            print(f"Waiting {startup_stagger_seconds:g}s before launching next run...")
            time.sleep(startup_stagger_seconds)

    exit_code = 0
    for process in processes:
        exit_code = max(exit_code, process.wait())
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run mat/scripts experiments from YAML")
    parser.add_argument("config", type=Path, help="Path to an experiment YAML file")
    parser.add_argument("--run", action="append", dest="run_names", help="Run only the named run. Can be repeated")
    parser.add_argument("--index", type=int, help="Run only the zero-based expanded run index after filtering")
    parser.add_argument("--count", action="store_true", help="Print the number of expanded runs after filtering and exit")
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
    if args.count:
        print(len(runs))
        return 0
    if args.index is not None:
        if args.index < 0 or args.index >= len(runs):
            print(f"--index {args.index} is out of range for {len(runs)} expanded runs", file=sys.stderr)
            return 1
        runs = [runs[args.index]]
    auto_run_count = sum(1 for run in runs if uses_auto_gpu(run))
    max_needed = auto_run_count if auto_run_count > 0 else None
    return run_commands(
        runs,
        args.dry_run,
        gpu_pool=configured_gpu_pool(config, max_needed=max_needed),
        startup_stagger_seconds=configured_startup_stagger_seconds(config),
        require_cuda=configured_require_cuda(config),
    )


if __name__ == "__main__":
    sys.exit(main())
