import copy
import time
from pathlib import Path

import imageio
import numpy as np
import torch
import wandb

from mat.runner.shared.mpe_runner import MPERunner, _t2n


class JaxMARLRobotariumRunner(MPERunner):
    """Runner for JaxMARL Robotarium wrappers.

    It reuses MAT/MPE rollout mechanics, but keeps environment logging generic
    instead of relying on `env_name == "MPE"`.
    """

    def run(self):
        self._preflight_check()
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        previous_obs = self.envs.reset()
        self.noise_rate = 0

        gif_num_snapshots = getattr(self.all_args, "gif_num_snapshots", 0)
        gif_interval = max(1, episodes // gif_num_snapshots) if gif_num_snapshots > 0 else None

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)

                obs, rewards, dones, infos = self.envs.step(actions_env)
                self.noise_rate = self.calculate_noise_rate(episode, episodes)
                if step > 0:
                    for i in range(len(obs)):
                        if np.random.random() < self.noise_rate:
                            obs[i] = previous_obs[i]
                if self.use_wandb:
                    wandb.log({"noise_rate": self.noise_rate})

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
                previous_obs = obs.copy()
                self.insert(data)

            self.compute()
            train_infos = self.train()

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save(episode)

            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                        self.all_args.scenario_name,
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                    )
                )

                env_infos = self._collect_env_infos(infos)
                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)
                if getattr(self.trainer, "use_distillation", False):
                    self.eval_student(total_num_steps)

            if gif_interval is not None and episode % gif_interval == 0:
                self.render_gif(episode, total_num_steps)

    def _preflight_check(self):
        """Smoke-test the full pipeline before training starts.

        Runs a few random-action steps through train envs, eval envs, and the
        render-gif env, exercising every path that would otherwise fail mid-run.
        Raises RuntimeError with a clear message if anything is broken.
        """
        print("[preflight] Checking pipelines before training...")
        n_actions = self.envs.action_space[0].n

        # --- train env: step + info logging ---
        self.envs.reset()
        for _ in range(2):
            dummy = np.eye(n_actions)[
                np.random.randint(n_actions, size=(self.n_rollout_threads, self.num_agents))
            ]
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

        # --- eval env ---
        if self.use_eval:
            self.eval_envs.reset()
            dummy_eval = np.eye(n_actions)[
                np.random.randint(n_actions, size=(self.n_eval_rollout_threads, self.num_agents))
            ]
            self.eval_envs.step(dummy_eval)

        # --- render-gif env (1 step, verify frames returned) ---
        if getattr(self.all_args, "gif_num_snapshots", 0) > 0:
            from mat.envs.marbler import MARBLEREnv
            render_args = copy.copy(self.all_args)
            render_args.marbler_save_gif = True
            render_args.marbler_show_figure_frequency = 1
            env = MARBLEREnv(render_args)
            env.seed(0)
            env.reset()
            dummy_render = np.eye(n_actions)[np.zeros(self.num_agents, dtype=int)]
            _, _, _, render_infos = env.step(dummy_render)
            frames = next(
                (info.get("frames", []) for info in render_infos if info.get("frames")),
                [],
            )
            env.close()
            if not frames:
                raise RuntimeError(
                    "[preflight] render_gif: env returned no frames. "
                    "Check marbler_save_gif / marbler_show_figure_frequency."
                )

        print("[preflight] All checks passed.\n")

    def _collect_env_infos(self, infos):
        env_infos = {}
        for agent_id in range(self.num_agents):
            for env_info in infos:
                if agent_id >= len(env_info):
                    continue
                for key, value in env_info[agent_id].items():
                    if isinstance(value, (str, bytes, list)):
                        continue
                    metric_key = f"agent{agent_id}/{key}"
                    env_infos.setdefault(metric_key, []).append(value)
        return env_infos

    @torch.no_grad()
    def render_gif(self, episode, total_num_steps):
        """Run one episode with a render-enabled env and save as a GIF."""
        from mat.envs.marbler import MARBLEREnv

        render_args = copy.copy(self.all_args)
        render_args.marbler_save_gif = True
        render_args.marbler_show_figure_frequency = 1

        env = MARBLEREnv(render_args)
        env.seed(self.all_args.seed + episode * 7919)

        obs = env.reset()  # (n_agents, obs_dim)
        obs_in = obs[np.newaxis]  # (1, n_agents, obs_dim)

        if self.use_centralized_V:
            share_obs = np.expand_dims(obs_in.reshape(1, -1), 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs_in

        rnn_states = np.zeros(
            (1, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32
        )
        masks = np.ones((1, self.num_agents, 1), dtype=np.float32)
        n_actions = env.action_space[0].n
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

            actions = np.array(np.split(_t2n(action), 1))         # (1, n_agents, 1)
            rnn_states = np.array(np.split(_t2n(rnn_states), 1))  # (1, n_agents, rN, hS)
            actions_env = np.squeeze(np.eye(n_actions)[actions], 2)  # (1, n_agents, n_actions)

            obs, rewards, dones, infos = env.step(actions_env[0])

            for agent_info in infos:
                frames = agent_info.get("frames", [])
                if frames:
                    all_frames.extend(frames)
                    break

            if np.any(dones):
                rnn_states[:] = 0.0
                masks[:] = 0.0
            else:
                masks = np.ones((1, self.num_agents, 1), dtype=np.float32)

            obs_in = obs[np.newaxis]
            if self.use_centralized_V:
                share_obs = np.expand_dims(obs_in.reshape(1, -1), 1).repeat(self.num_agents, axis=1)
            else:
                share_obs = obs_in

        env.close()

        if not all_frames:
            print("[render_gif] No frames collected.")
            return

        gif_dir = Path(self.run_dir) / "gifs"
        gif_dir.mkdir(parents=True, exist_ok=True)
        gif_path = gif_dir / f"step{total_num_steps:08d}.gif"

        rgb_frames = [frame[:, :, :3] for frame in all_frames]
        imageio.mimsave(str(gif_path), rgb_frames, fps=8)
        print(f"Saved GIF ({len(rgb_frames)} frames): {gif_path}")
