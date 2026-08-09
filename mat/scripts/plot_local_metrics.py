#!/usr/bin/env python3
"""Plot local SMAC/W&B metrics without contacting wandb.ai."""

import argparse
import csv
import fnmatch
import json
import math
import os
import re
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt


UPDATE_RE = re.compile(
    r"Exp\s+(?P<exp>\S+)\s+updates\s+(?P<update>\d+)/(?P<total_updates>\d+)"
    r".*?total num timesteps\s+(?P<step>\d+)/(?P<total_steps>\d+),\s+FPS\s+(?P<fps>\d+)"
)
INCRE_RE = re.compile(r"incre win rate is (?P<value>[-+0-9.eE]+)")
EVAL_RE = re.compile(r"eval win rate is (?P<value>[-+0-9.eE]+)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract local run metrics and save CSV/PNG plots."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Run directories, wandb/run-* directories, or experiment directories.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("mat/scripts/results/StarCraft2/3s5z_vs_3s6z/mat"),
        help="Root used for auto-discovery when paths are omitted.",
    )
    parser.add_argument(
        "--pattern",
        default="distill_partial_sight*",
        help="Experiment directory glob used with --root.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=["*ejitter*"],
        help="Exclude experiment/run paths matching this glob. Can be repeated.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("mat/scripts/results/plots/latest_partial_sight"),
        help="Directory for CSV and PNG outputs.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=5,
        help="Moving-average window for plotted curves. CSV remains unsmoothed.",
    )
    return parser.parse_args()


def discover_runs(paths, root, pattern, exclude):
    candidates = paths or sorted(root.glob(pattern))
    runs = []
    for path in candidates:
        path = path.resolve()
        if not path.exists():
            continue
        if any(fnmatch.fnmatch(str(path), pat) or fnmatch.fnmatch(path.name, pat) for pat in exclude):
            continue
        if path.name.startswith("run-") and (path / "files").is_dir():
            runs.append(path)
            continue
        runs.extend(
            run
            for run in sorted(path.glob("wandb/run-*"))
            if not any(fnmatch.fnmatch(str(run), pat) or fnmatch.fnmatch(run.name, pat) for pat in exclude)
        )
        runs.extend(
            run
            for run in sorted(path.glob("run-*"))
            if not any(fnmatch.fnmatch(str(run), pat) or fnmatch.fnmatch(run.name, pat) for pat in exclude)
        )
    unique = []
    seen = set()
    for run in runs:
        if run in seen or not (run / "files").is_dir():
            continue
        seen.add(run)
        unique.append(run)
    return unique


def run_label(run_dir):
    # .../<experiment>/wandb/run-* is the common layout.
    if run_dir.parent.name == "wandb":
        return run_dir.parent.parent.name
    return run_dir.name


def parse_output_log(run_dir):
    output_log = run_dir / "files" / "output.log"
    rows = []
    current = None
    if not output_log.exists():
        return rows

    for line in output_log.read_text(errors="replace").splitlines():
        update = UPDATE_RE.search(line)
        if update:
            current = {
                "experiment": update.group("exp"),
                "update": int(update.group("update")),
                "total_updates": int(update.group("total_updates")),
                "step": int(update.group("step")),
                "total_steps": int(update.group("total_steps")),
                "fps": int(update.group("fps")),
            }
            rows.append(current)
            continue
        if current is None:
            continue
        incre = INCRE_RE.search(line)
        if incre:
            current["incre_win_rate"] = float(incre.group("value").rstrip("."))
            continue
        eval_rate = EVAL_RE.search(line)
        if eval_rate:
            current["eval_win_rate"] = float(eval_rate.group("value").rstrip("."))
            continue
    return rows


def parse_wandb_history(run_dir):
    try:
        from wandb.proto import wandb_internal_pb2
        from wandb.sdk.internal import datastore
    except Exception as exc:
        return [], f"wandb import failed: {exc}"

    wandb_files = sorted(run_dir.glob("run-*.wandb"))
    if not wandb_files:
        return [], "no run-*.wandb file"

    ds = datastore.DataStore()
    try:
        ds.open_for_scan(str(wandb_files[-1]))
    except Exception as exc:
        return [], f"open failed: {exc}"

    rows = []
    parse_errors = 0
    scan_error = None
    while True:
        try:
            data = ds.scan_data()
        except Exception as exc:
            scan_error = f"scan stopped: {exc}"
            break
        if data is None:
            break
        proto = wandb_internal_pb2.Record()
        try:
            proto.ParseFromString(data)
        except Exception:
            parse_errors += 1
            continue
        if proto.WhichOneof("record_type") != "history":
            continue
        row = {}
        for item in proto.history.item:
            try:
                row[item.key] = json.loads(item.value_json)
            except Exception:
                row[item.key] = item.value_json
        rows.append(row)

    warning = None
    if parse_errors or scan_error:
        warning = f"parse_errors={parse_errors}"
        if scan_error:
            warning += f", {scan_error}"
    return rows, warning


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def numeric_pairs(rows, x_key, y_key):
    pairs = []
    for idx, row in enumerate(rows):
        if y_key not in row:
            continue
        x = row.get(x_key, row.get("_step", idx))
        y = row[y_key]
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            if math.isfinite(float(x)) and math.isfinite(float(y)):
                pairs.append((float(x), float(y)))
    return pairs


def smooth_pairs(pairs, window):
    if window <= 1 or len(pairs) < window:
        return pairs
    smoothed = []
    values = []
    for x, y in pairs:
        values.append(y)
        if len(values) > window:
            values.pop(0)
        smoothed.append((x, sum(values) / len(values)))
    return smoothed


def plot_metric(out_path, series, title, x_label="step", y_label=None, smooth=1):
    plt.figure(figsize=(10, 5.5))
    plotted = False
    for label, rows, x_key, y_key in series:
        pairs = smooth_pairs(numeric_pairs(rows, x_key, y_key), smooth)
        if not pairs:
            continue
        xs, ys = zip(*pairs)
        plt.plot(xs, ys, label=label)
        plotted = True
    if not plotted:
        plt.close()
        return False
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label or title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()
    return True


def main():
    args = parse_args()
    runs = discover_runs(args.paths, args.root, args.pattern, args.exclude)
    if not runs:
        raise SystemExit("No run directories found.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_by_run = {}
    wandb_by_run = {}
    summary = []

    for run in runs:
        label_base = run_label(run)
        label = label_base
        suffix = 2
        while label in output_by_run:
            label = f"{label_base}_{run.name.removeprefix('run-')}"
            if label in output_by_run:
                label = f"{label_base}_{suffix}"
                suffix += 1
        output_rows = parse_output_log(run)
        wandb_rows, warning = parse_wandb_history(run)
        output_by_run[label] = output_rows
        wandb_by_run[label] = wandb_rows
        write_csv(args.out_dir / f"{label}.output_log.csv", output_rows)
        write_csv(args.out_dir / f"{label}.wandb_history.csv", wandb_rows)
        summary.append(
            {
                "label": label,
                "run_dir": str(run),
                "output_rows": len(output_rows),
                "output_last_step": output_rows[-1]["step"] if output_rows else None,
                "wandb_history_rows": len(wandb_rows),
                "wandb_last_step": wandb_rows[-1].get("_step") if wandb_rows else None,
                "wandb_warning": warning,
            }
        )

    write_csv(args.out_dir / "summary.csv", summary)

    plot_metric(
        args.out_dir / "output_incre_win_rate.png",
        [(label, rows, "step", "incre_win_rate") for label, rows in output_by_run.items()],
        "Incremental Win Rate",
        y_label="win rate",
        smooth=args.smooth,
    )
    plot_metric(
        args.out_dir / "output_eval_win_rate.png",
        [(label, rows, "step", "eval_win_rate") for label, rows in output_by_run.items()],
        "Eval Win Rate",
        y_label="win rate",
        smooth=max(1, args.smooth // 2),
    )
    plot_metric(
        args.out_dir / "output_fps.png",
        [(label, rows, "step", "fps") for label, rows in output_by_run.items()],
        "FPS",
        y_label="fps",
        smooth=args.smooth,
    )

    wandb_metrics = [
        "average_step_rewards",
        "eval_average_episode_rewards",
        "student_eval_average_episode_rewards",
        "value_loss",
        "student_kl_loss",
        "student_rl_loss",
        "student_value_loss",
        "dead_ratio",
        "dist_entropy",
        "student_rl_coef",
    ]
    for metric in wandb_metrics:
        plot_metric(
            args.out_dir / f"wandb_{metric}.png",
            [(label, rows, "_step", metric) for label, rows in wandb_by_run.items()],
            metric,
            y_label=metric,
            smooth=args.smooth,
        )

    print(f"Wrote plots and CSV files to {args.out_dir}")
    for item in summary:
        print(
            f"{item['label']}: output rows={item['output_rows']} "
            f"last_step={item['output_last_step']} | wandb rows={item['wandb_history_rows']} "
            f"last_step={item['wandb_last_step']}"
        )
        if item["wandb_warning"]:
            print(f"  wandb warning: {item['wandb_warning']}")


if __name__ == "__main__":
    main()
