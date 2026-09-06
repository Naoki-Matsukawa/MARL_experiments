import copy
import time
from pathlib import Path

import imageio
import numpy as np
import torch

from mat.runner.shared.jaxmarl_robotarium_runner import JaxMARLRobotariumRunner, _t2n


class VMASRunner(JaxMARLRobotariumRunner):
    """Runner for VMAS environments.

    Overrides collect() for continuous Box action spaces and replaces
    MARBLER-specific preflight/GIF methods with VMAS-compatible versions.
    """

    # ------------------------------------------------------------------
    # collect: support Box (continuous) in addition to Discrete
    # ------------------------------------------------------------------

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic = (
            self.trainer.policy.get_actions(
                np.concatenate(self.buffer.share_obs[step]),
                np.concatenate(self.buffer.obs[step]),
                np.concatenate(self.buffer.rnn_states[step]),
                np.concatenate(self.buffer.rnn_states_critic[step]),
                np.concatenate(self.buffer.masks[step]),
            )
        )

        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        action_space = self.envs.action_space[0]
        cls = action_space.__class__.__name__

        if cls == "Discrete":
            actions_env = np.squeeze(np.eye(action_space.n)[actions], 2)
        elif cls == "MultiDiscrete":
            pieces = []
            for i in range(action_space.shape):
                pieces.append(np.eye(action_space.high[i] + 1)[actions[:, :, i]])
            actions_env = np.concatenate(pieces, axis=2)
        elif cls == "Box":
            actions_env = np.clip(actions, action_space.low, action_space.high)
        else:
            raise NotImplementedError(f"Unsupported action space: {cls}")

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    # ------------------------------------------------------------------
    # eval: support Box (continuous) action spaces
    # ------------------------------------------------------------------

    @torch.no_grad()
    def eval(self, total_num_steps):
        import random
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        if self.use_centralized_V:
            eval_share_obs = eval_obs.reshape(self.n_eval_rollout_threads, -1)
            eval_share_obs = np.expand_dims(eval_share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            eval_share_obs = eval_obs

        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32
        )
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        previous_eval_obs = eval_obs.copy()

        action_space = self.eval_envs.action_space[0]

        for _ in range(self.episode_length):
            self.trainer.prep_rollout()
            if self.algorithm_name == "r_mappo":
                eval_action, eval_rnn_states = self.trainer.policy.act(
                    np.concatenate(eval_obs),
                    np.concatenate(eval_rnn_states),
                    np.concatenate(eval_masks),
                    deterministic=True,
                )
            else:
                eval_action, eval_rnn_states = self.trainer.policy.act(
                    np.concatenate(eval_share_obs),
                    np.concatenate(eval_obs),
                    np.concatenate(eval_rnn_states),
                    np.concatenate(eval_masks),
                    deterministic=True,
                )

            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            cls = action_space.__class__.__name__
            if cls == "Discrete":
                eval_actions_env = np.squeeze(np.eye(action_space.n)[eval_actions], 2)
            elif cls == "MultiDiscrete":
                pieces = []
                for i in range(action_space.shape):
                    pieces.append(np.eye(action_space.high[i] + 1)[eval_actions[:, :, i]])
                eval_actions_env = np.concatenate(pieces, axis=2)
            elif cls == "Box":
                eval_actions_env = np.clip(eval_actions, action_space.low, action_space.high)
            else:
                raise NotImplementedError(f"Unsupported action space: {cls}")

            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            for i in range(len(eval_obs)):
                if random.random() < self.eval_noise_rate:
                    eval_obs[i] = previous_eval_obs[i]
            previous_eval_obs = eval_obs.copy()

            if self.use_centralized_V:
                eval_share_obs = eval_obs.reshape(self.n_eval_rollout_threads, -1)
                eval_share_obs = np.expand_dims(eval_share_obs, 1).repeat(self.num_agents, axis=1)
            else:
                eval_share_obs = eval_obs

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32
            )
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos["eval_average_episode_rewards"] = np.sum(eval_episode_rewards, axis=0)
        avg = np.mean(eval_env_infos["eval_average_episode_rewards"])
        print(f"eval average episode rewards of agent: {avg}")
        self.log_env(eval_env_infos, total_num_steps)

    # ------------------------------------------------------------------
    # preflight: VMAS-compatible random-action smoke test
    # ------------------------------------------------------------------

    def _preflight_check(self):
        print("[preflight] Checking pipelines before training...")
        action_space = self.envs.action_space[0]

        def _random_actions(n_threads, n_agents):
            if action_space.__class__.__name__ == "Discrete":
                n = action_space.n
                return np.eye(n)[np.random.randint(n, size=(n_threads, n_agents))]
            else:
                return np.stack(
                    [[action_space.sample() for _ in range(n_agents)] for _ in range(n_threads)]
                )

        self.envs.reset()
        for _ in range(2):
            dummy = _random_actions(self.n_rollout_threads, self.num_agents)
            _, _, _, infos = self.envs.step(dummy)
            env_infos = self._collect_env_infos(infos)
            for key, values in env_infos.items():
                try:
                    np.mean(values)
                except Exception as exc:
                    raise RuntimeError(
                        f"[preflight] log_env would fail: key='{key}' "
                        f"values={values!r} — {exc}"
                    ) from exc

        if self.use_eval:
            self.eval_envs.reset()
            dummy_eval = _random_actions(self.n_eval_rollout_threads, self.num_agents)
            self.eval_envs.step(dummy_eval)

        print("[preflight] All checks passed.\n")

    # ------------------------------------------------------------------
    # render_gif: VMAS-compatible GIF rendering
    # ------------------------------------------------------------------

    @torch.no_grad()
    def render_gif(self, episode, total_num_steps):
        from mat.envs.vmas import VMASEnv

        try:
            from pyvirtualdisplay import Display
            _display = Display(visible=False, size=(800, 800))
            _display.start()
        except Exception:
            _display = None

        render_args = copy.copy(self.all_args)
        env = VMASEnv(render_args)
        env.seed(self.all_args.seed + episode * 7919)

        obs = env.reset()                            # (n_agents, obs_dim)
        obs_in = obs[np.newaxis]                     # (1, n_agents, obs_dim)

        if self.use_centralized_V:
            share_obs = np.expand_dims(obs_in.reshape(1, -1), 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs_in

        rnn_states = np.zeros(
            (1, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32
        )
        masks = np.ones((1, self.num_agents, 1), dtype=np.float32)
        action_space = env.action_space[0]
        all_frames = []

        for _ in range(self.episode_length):
            self.trainer.prep_rollout()

            if self.algorithm_name == "r_mappo":
                action, rnn_states = self.trainer.policy.act(
                    np.concatenate(obs_in),
                    np.concatenate(rnn_states),
                    np.concatenate(masks),
                    deterministic=True,
                )
            else:
                action, rnn_states = self.trainer.policy.act(
                    np.concatenate(share_obs),
                    np.concatenate(obs_in),
                    np.concatenate(rnn_states),
                    np.concatenate(masks),
                    deterministic=True,
                )

            actions = np.array(np.split(_t2n(action), 1))        # (1, n_agents, act_dim)
            rnn_states = np.array(np.split(_t2n(rnn_states), 1))

            if action_space.__class__.__name__ == "Discrete":
                actions_env = np.squeeze(np.eye(action_space.n)[actions], 2)
            else:
                actions_env = np.clip(actions, action_space.low, action_space.high)

            obs, _, dones, _ = env.step(actions_env[0])  # env expects (n_agents, act_dim)

            frame = env.render(mode="rgb_array")
            if frame is not None:
                all_frames.append(frame)

            obs_in = obs[np.newaxis]
            if self.use_centralized_V:
                share_obs = np.expand_dims(obs_in.reshape(1, -1), 1).repeat(self.num_agents, axis=1)
            else:
                share_obs = obs_in

            if np.any(dones):
                rnn_states[:] = 0.0

        env.close()
        if _display is not None:
            try:
                _display.stop()
            except Exception:
                pass

        if all_frames:
            gif_dir = Path(self.run_dir) / "gifs"
            gif_dir.mkdir(exist_ok=True)
            gif_path = gif_dir / f"step{total_num_steps:08d}.gif"
            imageio.mimsave(str(gif_path), all_frames, fps=10)
            print(f"[gif] saved {gif_path}")
