import numpy as np
import torch
from gym import spaces


def _patch_gym_box():
    """Patch gym 0.12's Box to accept array low/high with explicit shape.

    gym 0.12 raises AssertionError when shape is given alongside array bounds.
    VMAS 1.x passes both, so we silently drop shape when the arrays already
    have the right shape.
    """
    import gym.spaces.box as box_mod

    if getattr(box_mod.Box, "_vmas_patched", False):
        return

    _orig_init = box_mod.Box.__init__

    def _patched_init(self, low, high, shape=None, dtype=np.float32):
        if shape is not None and not np.isscalar(low):
            low = np.asarray(low, dtype=dtype).reshape(shape)
            high = np.asarray(high, dtype=dtype).reshape(shape)
            shape = None  # use array-path in original init
        _orig_init(self, low, high, shape=shape, dtype=dtype)

    box_mod.Box.__init__ = _patched_init
    box_mod.Box._vmas_patched = True
    import gym.spaces
    gym.spaces.Box = box_mod.Box


_patch_gym_box()


class VMASEnv:
    """MAT-compatible wrapper around a single VMAS environment instance.

    VMAS is natively vectorized; we use num_envs=1 here so SubprocVecEnv can
    spawn independent workers the same way as other environments.

    Observations from all agents are concatenated to form share_obs in the
    runner (use_centralized_V=True path in MPERunner).
    """

    def __init__(self, all_args):
        try:
            import vmas
        except ImportError as exc:
            raise ImportError(
                "vmas is not installed. Run: pip install vmas"
            ) from exc

        self.scenario_name = str(getattr(all_args, "scenario_name", "navigation"))
        self.episode_length = int(getattr(all_args, "episode_length", 100))
        self.continuous_actions = bool(getattr(all_args, "vmas_continuous_actions", True))
        self._device = str(getattr(all_args, "vmas_device", "cpu"))

        # Collect scenario-specific kwargs prefixed with "vmas_scenario_"
        scenario_kwargs = {}
        for k, v in vars(all_args).items():
            if k.startswith("vmas_scenario_") and v is not None:
                scenario_kwargs[k[len("vmas_scenario_"):]] = v

        n_agents_arg = int(getattr(all_args, "vmas_n_agents", 4))
        scenario_kwargs.setdefault("n_agents", n_agents_arg)

        self._env = vmas.make_env(
            scenario=self.scenario_name,
            num_envs=1,
            device=self._device,
            continuous_actions=self.continuous_actions,
            max_steps=self.episode_length,
            seed=int(getattr(all_args, "seed", 1)),
            **scenario_kwargs,
        )

        self.num_agents = len(self._env.agents)
        self.n_agents = self.num_agents

        # Probe spaces via a reset
        obs_list = self._reset_raw()
        self.obs_dim = int(obs_list[0].shape[-1])
        self.share_obs_dim = self.obs_dim * self.num_agents

        self.observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
            for _ in range(self.num_agents)
        ]
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(self.share_obs_dim,), dtype=np.float32)
            for _ in range(self.num_agents)
        ]

        # env.action_space is a Tuple of per-agent spaces
        per_agent_space = self._env.action_space.spaces[0]
        self.action_space = [per_agent_space for _ in range(self.num_agents)]

    # ------------------------------------------------------------------
    def seed(self, seed=None):
        pass  # VMAS seed is set at construction; re-seeding is a no-op here

    def reset(self):
        obs_list = self._reset_raw()
        return self._stack_obs(obs_list)

    def step(self, actions):
        """
        actions: (n_agents, act_dim) for continuous, (n_agents,) for discrete.
        Returns: obs, rewards, dones, infos  — shapes (n_agents, *).
        """
        action_tensors = self._make_action_tensors(actions)
        result = self._env.step(action_tensors)

        obs_list, reward_list, done_raw, info = result[0], result[1], result[2], result[3]

        obs = self._stack_obs(obs_list)

        # reward_list: list of n_agents tensors [1] or [1,1]
        rewards = np.array(
            [r.detach().cpu().numpy().reshape(-1)[0] for r in reward_list],
            dtype=np.float32,
        ).reshape(self.num_agents, 1)

        # done_raw: single Tensor [num_envs] (env-level done, broadcast to all agents)
        if isinstance(done_raw, (list, tuple)):
            done_val = bool(done_raw[0].detach().cpu().numpy().reshape(-1)[0])
        else:
            done_val = bool(done_raw.detach().cpu().numpy().reshape(-1)[0])
        dones = np.array([done_val] * self.num_agents)

        infos = [{} for _ in range(self.num_agents)]
        return obs, rewards, dones, infos

    def render(self, mode="rgb_array"):
        try:
            return self._env.render(mode=mode, visualize_when_rgb=True)
        except Exception:
            return None

    def close(self):
        pass

    # ------------------------------------------------------------------
    def _reset_raw(self):
        result = self._env.reset()
        return result[0] if isinstance(result, tuple) else result

    def _stack_obs(self, obs_list):
        return np.stack(
            [o.detach().cpu().numpy().reshape(-1) for o in obs_list], axis=0
        ).astype(np.float32)  # (n_agents, obs_dim)

    def _make_action_tensors(self, actions):
        tensors = []
        for i in range(self.num_agents):
            if self.continuous_actions:
                a = np.asarray(actions[i], dtype=np.float32).reshape(1, -1)
                tensors.append(torch.FloatTensor(a).to(self._device))
            else:
                a = int(actions[i]) if np.ndim(actions[i]) == 0 else int(np.argmax(actions[i]))
                tensors.append(torch.LongTensor([[a]]).to(self._device))
        return tensors
