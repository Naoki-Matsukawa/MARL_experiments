import sys
from pathlib import Path

import gym
import numpy as np
from gym import spaces


REPO_ROOT = Path(__file__).resolve().parents[3]
ROBOTARIUM_ROOT = REPO_ROOT / "3rdparty" / "robotarium_python_simulator"
if str(ROBOTARIUM_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOTARIUM_ROOT))

try:
    from rps.robotarium import Robotarium
    from rps.robotarium_abc import ARobotarium
except ImportError as exc:
    raise ImportError(
        "Robotarium official simulator is missing. Expected it at "
        f"{ROBOTARIUM_ROOT}. Clone https://github.com/robotarium/robotarium_python_simulator "
        "and install cvxopt in the active Python environment."
    ) from exc


class RobotariumEnv(gym.Env):
    """MAT-compatible wrapper around the official Robotarium simulator.

    The simulator dynamics and robot constants come from `rps.robotarium.Robotarium`.
    This wrapper only defines an RL navigation task: observations, discrete actions,
    rewards, and episode termination.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, all_args):
        self.num_agents = int(getattr(all_args, "num_agents", 4))
        self.n = self.num_agents
        self.episode_length = int(getattr(all_args, "episode_length", 200))
        self.neighbor_count = int(getattr(all_args, "robotarium_neighbor_count", 3))
        self.sensing_radius = float(getattr(all_args, "robotarium_sensing_radius", 0.75))
        self.goal_radius = float(getattr(all_args, "robotarium_goal_radius", 0.08))
        self.collision_penalty = float(getattr(all_args, "robotarium_collision_penalty", 1.0))
        self.progress_reward_scale = float(getattr(all_args, "robotarium_progress_reward_scale", 10.0))
        self.step_penalty = float(getattr(all_args, "robotarium_step_penalty", 0.01))
        self.goal_reward = float(getattr(all_args, "robotarium_goal_reward", 1.0))
        self.use_distance_sensors = bool(getattr(all_args, "robotarium_use_distance_sensors", False))
        self.show_figure = bool(getattr(all_args, "robotarium_show_figure", False))
        self.sim_in_real_time = bool(getattr(all_args, "robotarium_sim_in_real_time", False))

        self.boundaries = ARobotarium.BOUNDARIES.astype(np.float32)
        self.robot_diameter = float(ARobotarium.ROBOT_DIAMETER)
        self.collision_diameter = float(ARobotarium.COLLISION_DIAMETER)
        self.max_linear_velocity = float(ARobotarium.MAX_LINEAR_VELOCITY)
        self.max_angular_velocity = float(ARobotarium.MAX_ANGULAR_VELOCITY)
        self._ids = np.arange(self.num_agents)

        # own pose(4) + goal rel(2) + nearest neighbors(rel xy + visible flag) + agent id.
        self.obs_dim = 6 + self.neighbor_count * 3 + self.num_agents
        self.share_obs_dim = self.obs_dim * self.num_agents
        self.action_dim = 7

        self.observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
            for _ in range(self.num_agents)
        ]
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(self.share_obs_dim,), dtype=np.float32)
            for _ in range(self.num_agents)
        ]
        self.action_space = [spaces.Discrete(self.action_dim) for _ in range(self.num_agents)]

        self._rng = np.random.default_rng()
        self._robotarium = None
        self._poses = None
        self._goals = None
        self._previous_distances = None
        self._step_count = 0

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)
        np.random.seed(seed)

    def reset(self):
        self._step_count = 0
        initial_conditions = self._sample_collision_free_poses(self.num_agents)
        self._goals = self._sample_goals(initial_conditions)
        self._robotarium = Robotarium(
            number_of_robots=self.num_agents,
            show_figure=self.show_figure,
            initial_conditions=initial_conditions,
            use_distance_sensors=self.use_distance_sensors,
            sim_in_real_time=self.sim_in_real_time,
            skip_initialization=True,
        )
        self._poses = self._robotarium.get_poses()
        self._previous_distances = self._goal_distances(self._poses)
        return self._observations()

    def step(self, actions):
        if self._robotarium is None:
            self.reset()

        action_indices = self._decode_actions(actions)
        velocities = self._actions_to_velocities(action_indices)
        previous_distances = self._previous_distances.copy()

        self._robotarium.set_velocities(self._ids, velocities)
        self._robotarium.step()
        self._poses = self._robotarium.get_poses()
        self._step_count += 1

        distances = self._goal_distances(self._poses)
        self._previous_distances = distances
        reached = distances <= self.goal_radius
        collision = self._collision_flags(self._poses)
        out_of_bounds = self._out_of_bounds_flags(self._poses)
        done = bool(np.all(reached) or self._step_count >= self.episode_length)

        progress = previous_distances - distances
        rewards = (
            self.progress_reward_scale * progress
            - self.step_penalty
            + self.goal_reward * reached.astype(np.float32)
            - self.collision_penalty * collision.astype(np.float32)
            - self.collision_penalty * out_of_bounds.astype(np.float32)
        ).astype(np.float32)

        infos = [
            {
                "individual_reward": float(rewards[i]),
                "distance_to_goal": float(distances[i]),
                "reached_goal": bool(reached[i]),
                "collision": bool(collision[i]),
                "out_of_bounds": bool(out_of_bounds[i]),
            }
            for i in range(self.num_agents)
        ]

        return (
            self._observations(),
            rewards.reshape(self.num_agents, 1),
            np.full(self.num_agents, done, dtype=bool),
            infos,
        )

    def render(self, mode="rgb_array"):
        frame = self._render_frame()
        if mode == "rgb_array":
            return frame
        if mode == "human":
            return None
        raise NotImplementedError(mode)

    def close(self):
        self._robotarium = None

    def _decode_actions(self, actions):
        actions = np.asarray(actions)
        if actions.ndim == 2:
            return np.argmax(actions, axis=1).astype(np.int64)
        return actions.reshape(self.num_agents).astype(np.int64)

    def _actions_to_velocities(self, action_indices):
        v = 0.75 * self.max_linear_velocity
        w = 0.75 * self.max_angular_velocity
        table = np.array(
            [
                [0.0, 0.0],
                [v, 0.0],
                [0.0, w],
                [0.0, -w],
                [v, w],
                [v, -w],
                [-0.5 * v, 0.0],
            ],
            dtype=np.float32,
        )
        return table[np.clip(action_indices, 0, self.action_dim - 1)].T

    def _observations(self):
        poses = self._poses
        goals = self._goals
        obs = np.zeros((self.num_agents, self.obs_dim), dtype=np.float32)
        x_min, x_max, y_min, y_max = self.boundaries
        x_scale = max(x_max - x_min, 1e-6)
        y_scale = max(y_max - y_min, 1e-6)

        for i in range(self.num_agents):
            theta = poses[2, i]
            goal_rel = goals[:, i] - poses[:2, i]
            pieces = [
                np.array(
                    [
                        2.0 * (poses[0, i] - x_min) / x_scale - 1.0,
                        2.0 * (poses[1, i] - y_min) / y_scale - 1.0,
                        np.cos(theta),
                        np.sin(theta),
                        goal_rel[0] / x_scale,
                        goal_rel[1] / y_scale,
                    ],
                    dtype=np.float32,
                )
            ]

            neighbor_features = []
            rels = poses[:2, :].T - poses[:2, i]
            distances = np.linalg.norm(rels, axis=1)
            order = [j for j in np.argsort(distances) if j != i]
            for j in order[: self.neighbor_count]:
                visible = distances[j] <= self.sensing_radius
                if visible:
                    neighbor_features.extend([rels[j, 0] / x_scale, rels[j, 1] / y_scale, 1.0])
                else:
                    neighbor_features.extend([0.0, 0.0, 0.0])
            while len(neighbor_features) < self.neighbor_count * 3:
                neighbor_features.extend([0.0, 0.0, 0.0])
            pieces.append(np.array(neighbor_features, dtype=np.float32))

            agent_id = np.zeros(self.num_agents, dtype=np.float32)
            agent_id[i] = 1.0
            pieces.append(agent_id)
            obs[i] = np.concatenate(pieces)
        return obs

    def _sample_collision_free_poses(self, count):
        poses = np.zeros((3, count), dtype=np.float32)
        x_min, x_max, y_min, y_max = self.boundaries
        margin = self.robot_diameter
        min_spacing = max(0.25, 1.8 * self.collision_diameter)
        placed = []
        for i in range(count):
            for _ in range(1000):
                xy = np.array(
                    [
                        self._rng.uniform(x_min + margin, x_max - margin),
                        self._rng.uniform(y_min + margin, y_max - margin),
                    ],
                    dtype=np.float32,
                )
                if all(np.linalg.norm(xy - prev) >= min_spacing for prev in placed):
                    placed.append(xy)
                    poses[:2, i] = xy
                    poses[2, i] = self._rng.uniform(-np.pi, np.pi)
                    break
            else:
                raise RuntimeError("Could not sample collision-free Robotarium initial poses.")
        return poses

    def _sample_goals(self, initial_conditions):
        poses = self._sample_collision_free_poses(self.num_agents)
        goals = poses[:2, :].copy()
        # Prefer goals on the opposite half of the arena to create non-trivial navigation.
        x_mid = 0.5 * (self.boundaries[0] + self.boundaries[1])
        for i in range(self.num_agents):
            for _ in range(100):
                candidate = self._sample_collision_free_poses(1)[:2, 0]
                if (initial_conditions[0, i] < x_mid and candidate[0] > x_mid) or (
                    initial_conditions[0, i] >= x_mid and candidate[0] < x_mid
                ):
                    goals[:, i] = candidate
                    break
        return goals.astype(np.float32)

    def _goal_distances(self, poses):
        return np.linalg.norm((self._goals - poses[:2, :]).T, axis=1).astype(np.float32)

    def _collision_flags(self, poses):
        flags = np.zeros(self.num_agents, dtype=bool)
        for i in range(self.num_agents - 1):
            for j in range(i + 1, self.num_agents):
                if np.linalg.norm(poses[:2, i] - poses[:2, j]) <= self.collision_diameter:
                    flags[i] = True
                    flags[j] = True
        return flags

    def _out_of_bounds_flags(self, poses):
        x_min, x_max, y_min, y_max = self.boundaries
        return (
            (poses[0, :] < x_min)
            | (poses[0, :] > x_max)
            | (poses[1, :] < y_min)
            | (poses[1, :] > y_max)
        )

    def _render_frame(self):
        height, width = 360, 576
        frame = np.full((height, width, 3), 255, dtype=np.uint8)
        if self._poses is None or self._goals is None:
            return frame

        x_min, x_max, y_min, y_max = self.boundaries

        def to_px(xy):
            x = int((xy[0] - x_min) / (x_max - x_min) * (width - 1))
            y = int((y_max - xy[1]) / (y_max - y_min) * (height - 1))
            return np.array([np.clip(x, 0, width - 1), np.clip(y, 0, height - 1)])

        colors = np.array(
            [
                [31, 119, 180],
                [255, 127, 14],
                [44, 160, 44],
                [214, 39, 40],
                [148, 103, 189],
                [140, 86, 75],
                [227, 119, 194],
                [127, 127, 127],
            ],
            dtype=np.uint8,
        )
        for i in range(self.num_agents):
            color = colors[i % len(colors)]
            goal = to_px(self._goals[:, i])
            robot = to_px(self._poses[:2, i])
            self._draw_disc(frame, goal, 5, np.array([40, 40, 40], dtype=np.uint8))
            self._draw_disc(frame, robot, 7, color)
        return frame

    @staticmethod
    def _draw_disc(frame, center, radius, color):
        h, w = frame.shape[:2]
        y0, y1 = max(0, center[1] - radius), min(h, center[1] + radius + 1)
        x0, x1 = max(0, center[0] - radius), min(w, center[0] + radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius ** 2
        frame[y0:y1, x0:x1][mask] = color
