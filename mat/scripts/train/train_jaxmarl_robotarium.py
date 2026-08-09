#!/usr/bin/env python
import os
import socket
import sys
from pathlib import Path

import numpy as np
import setproctitle
import torch
import wandb
import jax

sys.path.append("../../")
from mat.config import get_config
from mat.envs.env_wrappers import DummyVecEnv
from mat.envs.jaxmarl_robotarium import JaxMARLRobotariumDiscoveryEnv
from mat.runner.shared.jaxmarl_robotarium_runner import JaxMARLRobotariumRunner as Runner


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = JaxMARLRobotariumDiscoveryEnv(all_args)
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    # JAX starts background threads and is not fork-safe. Keep JaxMARL envs
    # in-process; using SubprocVecEnv/fork can corrupt CUDA context creation.
    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = JaxMARLRobotariumDiscoveryEnv(all_args)
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str, default="discovery")
    parser.add_argument("--num_agents", type=int, default=4)
    parser.add_argument("--jaxmarl_num_landmarks", type=int, default=6)
    parser.add_argument("--jaxmarl_num_sensing", type=int, default=2)
    parser.add_argument("--jaxmarl_num_tagging", type=int, default=2)
    parser.add_argument("--jaxmarl_sensing_radius", type=float, default=0.45)
    parser.add_argument("--jaxmarl_tagging_radius", type=float, default=0.25)
    parser.add_argument("--jaxmarl_heterogeneity_obs_type", type=str, default="none")
    parser.add_argument("--jaxmarl_update_frequency", type=int, default=30)
    parser.add_argument("--jaxmarl_step_dist", type=float, default=0.2)
    parser.add_argument("--jaxmarl_sense_shaping", type=float, default=1.0)
    parser.add_argument("--jaxmarl_tag_shaping", type=float, default=5.0)
    parser.add_argument("--jaxmarl_violation_shaping", type=float, default=0.0)
    parser.add_argument("--jaxmarl_time_shaping", type=float, default=-0.05)
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
    jax_backend = jax.default_backend()
    if jax_backend not in {"gpu", "cuda"}:
        raise RuntimeError(
            f"JAX CUDA backend is required, but JAX default backend is {jax_backend!r}. "
            "Install CUDA-enabled jaxlib/jax-cuda plugin and check CUDA_VISIBLE_DEVICES."
        )

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
