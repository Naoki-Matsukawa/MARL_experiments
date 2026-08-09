import os
import sys
import tempfile
from pathlib import Path

import gym
import numpy as np
import yaml
from gym import spaces


REPO_ROOT = Path(__file__).resolve().parents[3]
MARBLER_ROOT = REPO_ROOT / "3rdparty" / "MARBLER"
ROBOTARIUM_ROOT = REPO_ROOT / "3rdparty" / "robotarium_python_simulator_marbler"

for path in (ROBOTARIUM_ROOT, MARBLER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from robotarium_gym.wrapper import Wrapper
except ImportError as exc:
    raise ImportError(
        "MARBLER is missing. Expected GT-STAR-Lab/MARBLER at "
        f"{MARBLER_ROOT} and Robotarium simulator commit 6bb184e at {ROBOTARIUM_ROOT}."
    ) from exc


class MARBLEREnv(gym.Env):
    """MAT-compatible wrapper around official non-JAX MARBLER Gym scenarios."""

    def __init__(self, all_args):
        self.scenario_name = str(getattr(all_args, "scenario_name", "Simple"))
        self.episode_length = int(getattr(all_args, "episode_length", 100))
        self._seed = int(getattr(all_args, "seed", 1))

        self.show_figure_frequency = int(getattr(all_args, "marbler_show_figure_frequency", -1))
        self.save_gif = bool(getattr(all_args, "marbler_save_gif", False))
        self.enable_logging = bool(getattr(all_args, "marbler_enable_logging", False))
        self.real_time = bool(getattr(all_args, "marbler_real_time", False))
        self.robotarium = bool(getattr(all_args, "marbler_robotarium", False))
        self.k_neighbors = getattr(all_args, "marbler_k_neighbors", None)
        self.map_left = getattr(all_args, "marbler_map_left", None)
        self.map_right = getattr(all_args, "marbler_map_right", None)
        self.map_up = getattr(all_args, "marbler_map_up", None)
        self.map_down = getattr(all_args, "marbler_map_down", None)
        self.start_dist = getattr(all_args, "marbler_start_dist", None)
        self.robot_init_right_thresh = getattr(all_args, "marbler_robot_init_right_thresh", None)
        self.prey_init_left_thresh = getattr(all_args, "marbler_prey_init_left_thresh", None)

        self._tmp_config_path = None
        self._env = self._make_env()
        self.num_agents = int(self._env.n_agents)
        self.n_agents = self.num_agents
        self.n = self.num_agents

        sample_obs = np.asarray(self._env.reset(), dtype=np.float32)
        if sample_obs.ndim != 2:
            raise ValueError(f"MARBLER observation must be 2-D, got shape {sample_obs.shape}")
        self.obs_dim = int(sample_obs.shape[1])
        self.share_obs_dim = self.obs_dim * self.num_agents

        action_spaces = list(getattr(self._env.action_space, "spaces", []))
        if len(action_spaces) != self.num_agents:
            raise ValueError("MARBLER action space must be a Tuple with one Discrete space per agent")
        self.action_space = action_spaces
        self.observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
            for _ in range(self.num_agents)
        ]
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(self.share_obs_dim,), dtype=np.float32)
            for _ in range(self.num_agents)
        ]

    def seed(self, seed=None):
        if seed is not None:
            self._seed = int(seed)
        np.random.seed(self._seed)

    def reset(self):
        return np.asarray(self._env.reset(), dtype=np.float32)

    def step(self, actions):
        action_indices = self._decode_actions(actions)
        obs, rewards, dones, info = self._env.step(action_indices)

        obs = np.asarray(obs, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(self.num_agents, 1)
        dones = np.asarray(dones, dtype=bool).reshape(self.num_agents)
        infos = self._format_infos(info, rewards)
        return obs, rewards, dones, infos

    def render(self, mode="human"):
        if hasattr(self._env.env, "render"):
            return self._env.env.render(mode=mode)
        return None

    def close(self):
        if hasattr(self._env.env, "close"):
            self._env.env.close()
        if self._tmp_config_path and os.path.exists(self._tmp_config_path):
            os.unlink(self._tmp_config_path)
            self._tmp_config_path = None

    def _make_env(self):
        config_path = self._write_config()
        return Wrapper(self.scenario_name, config_path)

    def _write_config(self):
        source = MARBLER_ROOT / "robotarium_gym" / "scenarios" / self.scenario_name / "config.yaml"
        if not source.exists():
            raise FileNotFoundError(f"Unknown MARBLER scenario or missing config: {source}")
        with source.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        config["seed"] = self._seed
        config["max_episode_steps"] = self.episode_length
        config["show_figure_frequency"] = self.show_figure_frequency
        config["save_gif"] = self.save_gif
        config["enable_logging"] = self.enable_logging
        config["real_time"] = self.real_time
        config["robotarium"] = self.robotarium

        # Fix slot count to num_robots-1 for consistent obs_dim, then apply k masking.
        num_robots = config.get("predator", 2) + config.get("capture", 2)
        config["num_neighbors"] = num_robots - 1
        if self.k_neighbors is not None:
            config["k_neighbors"] = int(self.k_neighbors)

        # Map boundaries, spawn distances, and spawn zones.
        for attr, key in [
            ("map_left", "LEFT"), ("map_right", "RIGHT"),
            ("map_up", "UP"), ("map_down", "DOWN"),
            ("start_dist", "start_dist"),
            ("robot_init_right_thresh", "ROBOT_INIT_RIGHT_THRESH"),
            ("prey_init_left_thresh", "PREY_INIT_LEFT_THRESH"),
        ]:
            val = getattr(self, attr)
            if val is not None:
                config[key] = float(val)

        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix="marbler_", delete=False)
        with tmp:
            yaml.safe_dump(config, tmp)
        self._tmp_config_path = tmp.name
        return self._tmp_config_path

    def _decode_actions(self, actions):
        actions = np.asarray(actions)
        if actions.ndim == 2:
            return np.argmax(actions, axis=1).astype(np.int64).tolist()
        return actions.reshape(self.num_agents).astype(np.int64).tolist()

    def _format_infos(self, info, rewards):
        if isinstance(info, dict):
            infos = []
            for agent_id in range(self.num_agents):
                agent_info = {"individual_reward": float(rewards[agent_id, 0])}
                for key, value in info.items():
                    if isinstance(value, (list, tuple, np.ndarray)) and len(value) == self.num_agents:
                        agent_info[key] = value[agent_id]
                    else:
                        agent_info[key] = value
                infos.append(agent_info)
            return infos
        return [{"individual_reward": float(rewards[i, 0])} for i in range(self.num_agents)]
