#!/usr/bin/env python
import os
import socket
import sys
from pathlib import Path

import numpy as np
import setproctitle
import torch
import wandb

sys.path.append("../../")
from mat.config import get_config
from mat.envs.env_wrappers import DummyVecEnv, SubprocVecEnv
from mat.envs.robotarium import RobotariumEnv
from mat.runner.shared.mpe_runner import MPERunner as Runner


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = RobotariumEnv(all_args)
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = RobotariumEnv(all_args)
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str, default="navigation")
    parser.add_argument("--num_agents", type=int, default=4)
    parser.add_argument("--robotarium_neighbor_count", type=int, default=3)
    parser.add_argument("--robotarium_sensing_radius", type=float, default=0.75)
    parser.add_argument("--robotarium_goal_radius", type=float, default=0.08)
    parser.add_argument("--robotarium_collision_penalty", type=float, default=1.0)
    parser.add_argument("--robotarium_progress_reward_scale", type=float, default=10.0)
    parser.add_argument("--robotarium_step_penalty", type=float, default=0.01)
    parser.add_argument("--robotarium_goal_reward", type=float, default=1.0)
    parser.add_argument("--robotarium_use_distance_sensors", action="store_true", default=False)
    parser.add_argument("--robotarium_show_figure", action="store_true", default=False)
    parser.add_argument("--robotarium_sim_in_real_time", action="store_true", default=False)
    return parser.parse_known_args(args)[0]


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        raise RuntimeError("CUDA is required. Check CUDA_VISIBLE_DEVICES and the PyTorch CUDA build.")

    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / all_args.env_name
        / all_args.scenario_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    if all_args.use_wandb:
        run = wandb.init(
            config=all_args,
            project=all_args.env_name,
            entity=all_args.user_name,
            notes=socket.gethostname(),
            name=f"{all_args.algorithm_name}_{all_args.experiment_name}_seed{all_args.seed}",
            group=all_args.scenario_name,
            dir=str(run_dir),
            job_type="training",
            reinit=True,
        )
    else:
        existing = [
            int(folder.name.split("run")[1])
            for folder in run_dir.iterdir()
            if folder.name.startswith("run") and folder.name.split("run")[1].isdigit()
        ]
        run_dir = run_dir / f"run{max(existing, default=0) + 1}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run = None

    setproctitle.setproctitle(
        f"{all_args.algorithm_name}-{all_args.env_name}-{all_args.experiment_name}@{all_args.user_name}"
    )

    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": all_args.num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    runner = Runner(config)
    runner.run()

    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
