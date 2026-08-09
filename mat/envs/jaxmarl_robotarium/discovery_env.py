import importlib.util
import sys
import types
from pathlib import Path

import gym
import numpy as np
from gym import spaces


REPO_ROOT = Path(__file__).resolve().parents[3]
JAXMARL_ROOT = REPO_ROOT / "3rdparty" / "JaxMARL-Robotarium"
MARBLER_ROOT = JAXMARL_ROOT / "jaxmarl" / "environments" / "marbler"
RPS_JAX_ROOT = MARBLER_ROOT / "robotarium_python_simulator"


def _load_module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_discovery_class():
    if not RPS_JAX_ROOT.exists():
        raise ImportError(
            "JaxMARL-Robotarium rps_jax submodule is missing. Run "
            "`git submodule update --init --recursive` inside 3rdparty/JaxMARL-Robotarium."
        )

    if str(RPS_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(RPS_JAX_ROOT))

    # Avoid importing jaxmarl.__init__, which pulls unrelated environments such as Brax.
    sys.modules.setdefault("jaxmarl", types.ModuleType("jaxmarl"))
    sys.modules.setdefault("jaxmarl.environments", types.ModuleType("jaxmarl.environments"))
    sys.modules.setdefault("jaxmarl.environments.marbler", types.ModuleType("jaxmarl.environments.marbler"))
    sys.modules.setdefault("jaxmarl.environments.marbler.scenarios", types.ModuleType("jaxmarl.environments.marbler.scenarios"))

    _load_module("jaxmarl.environments.spaces", JAXMARL_ROOT / "jaxmarl" / "environments" / "spaces.py")
    _load_module("jaxmarl.environments.marbler.constants", MARBLER_ROOT / "constants.py")
    _load_module("jaxmarl.environments.marbler.robotarium_visualizer", MARBLER_ROOT / "robotarium_visualizer.py")
    _load_module("jaxmarl.environments.marbler.robotarium_env", MARBLER_ROOT / "robotarium_env.py")
    discovery_module = _load_module(
        "jaxmarl.environments.marbler.scenarios.discovery",
        MARBLER_ROOT / "scenarios" / "discovery.py",
    )
    return discovery_module.Discovery


class JaxMARLRobotariumDiscoveryEnv(gym.Env):
    """MAT wrapper for GT-STAR-Lab/JaxMARL-Robotarium Discovery.

    This class does not reimplement the task logic. It calls the original
    JaxMARL-Robotarium `Discovery` environment and converts its dict/JAX API
    into the array API expected by this MAT codebase.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, all_args):
        import jax
        import jax.numpy as jnp

        self.jax = jax
        self.jnp = jnp
        self.num_agents = int(getattr(all_args, "num_agents", 4))
        self.n = self.num_agents
        self.max_steps = int(getattr(all_args, "episode_length", 80))
        self.num_landmarks = int(getattr(all_args, "jaxmarl_num_landmarks", 6))
        self.num_sensing = int(getattr(all_args, "jaxmarl_num_sensing", 2))
        self.num_tagging = int(getattr(all_args, "jaxmarl_num_tagging", 2))
        if self.num_sensing + self.num_tagging != self.num_agents:
            raise ValueError("jaxmarl_num_sensing + jaxmarl_num_tagging must equal num_agents.")

        sensing_radius = float(getattr(all_args, "jaxmarl_sensing_radius", 0.45))
        tagging_radius = float(getattr(all_args, "jaxmarl_tagging_radius", 0.25))
        het_values = [[sensing_radius, 0.0] for _ in range(self.num_sensing)]
        het_values += [[0.0, tagging_radius] for _ in range(self.num_tagging)]
        obs_type = getattr(all_args, "jaxmarl_heterogeneity_obs_type", None)
        if obs_type == "none":
            obs_type = None

        Discovery = _load_discovery_class()
        self.env = Discovery(
            num_agents=self.num_agents,
            max_steps=self.max_steps,
            num_landmarks=self.num_landmarks,
            num_sensing=self.num_sensing,
            num_tagging=self.num_tagging,
            action_type="Discrete",
            update_frequency=int(getattr(all_args, "jaxmarl_update_frequency", 30)),
            step_dist=float(getattr(all_args, "jaxmarl_step_dist", 0.2)),
            sense_shaping=float(getattr(all_args, "jaxmarl_sense_shaping", 1.0)),
            tag_shaping=float(getattr(all_args, "jaxmarl_tag_shaping", 5.0)),
            violation_shaping=float(getattr(all_args, "jaxmarl_violation_shaping", 0.0)),
            time_shaping=float(getattr(all_args, "jaxmarl_time_shaping", -0.05)),
            heterogeneity={
                "type": "capability_set",
                "obs_type": obs_type,
                "values": het_values,
                "sample": False,
            },
            controller={
                "controller": "clf_uni_position",
                "barrier_fn": "robust_barriers",
            },
            robotarium={
                "number_of_robots": self.num_agents,
                "show_figure": False,
                "sim_in_real_time": False,
            },
        )

        self.agents = list(self.env.agents)
        self.obs_dim = int(self.env.obs_dim)
        self.share_obs_dim = self.obs_dim * self.num_agents
        self.action_dim = int(self.env.action_dim)
        self.observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
            for _ in range(self.num_agents)
        ]
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(self.share_obs_dim,), dtype=np.float32)
            for _ in range(self.num_agents)
        ]
        self.action_space = [spaces.Discrete(self.action_dim) for _ in range(self.num_agents)]
        self._seed = 0
        self._key = None
        self._state = None

    def seed(self, seed=None):
        self._seed = 0 if seed is None else int(seed)
        self._key = self.jax.random.PRNGKey(self._seed)

    def reset(self):
        if self._key is None:
            self.seed(self._seed)
        self._key, reset_key = self.jax.random.split(self._key)
        obs, self._state = self.env.reset(reset_key)
        return self._obs_dict_to_array(obs)

    def step(self, actions):
        if self._state is None:
            self.reset()
        self._key, step_key = self.jax.random.split(self._key)
        action_dict = self._actions_to_dict(actions)
        obs, self._state, rewards, dones, infos = self.env.step_env(step_key, self._state, action_dict)
        reward_array = self._reward_dict_to_array(rewards)
        return (
            self._obs_dict_to_array(obs),
            reward_array,
            self._done_dict_to_array(dones),
            self._info_dict_to_list(infos, reward_array),
        )

    def render(self, mode="rgb_array"):
        frame = np.full((360, 576, 3), 255, dtype=np.uint8)
        if mode == "rgb_array":
            return frame
        if mode == "human":
            return None
        raise NotImplementedError(mode)

    def close(self):
        self._state = None

    def _actions_to_dict(self, actions):
        actions = np.asarray(actions)
        if actions.ndim == 2:
            action_indices = np.argmax(actions, axis=1)
        else:
            action_indices = actions.reshape(self.num_agents)
        return {
            agent: self.jnp.asarray(int(action_indices[i]), dtype=self.jnp.int32)
            for i, agent in enumerate(self.agents)
        }

    def _obs_dict_to_array(self, obs):
        return np.stack([np.asarray(obs[agent], dtype=np.float32) for agent in self.agents], axis=0)

    def _reward_dict_to_array(self, rewards):
        return np.asarray([[float(np.asarray(rewards[agent]))] for agent in self.agents], dtype=np.float32)

    def _done_dict_to_array(self, dones):
        return np.asarray([bool(np.asarray(dones[agent])) for agent in self.agents], dtype=bool)

    def _info_dict_to_list(self, infos, rewards):
        out = []
        for i, agent in enumerate(self.agents):
            info = {}
            for key, value in infos.items():
                arr = np.asarray(value)
                if arr.shape == ():
                    item = arr.item()
                else:
                    item = arr[i].item()
                info[key] = item
            info["individual_reward"] = float(rewards[i, 0])
            out.append(info)
        return out
