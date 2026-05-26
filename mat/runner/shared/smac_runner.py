import time
import wandb
import numpy as np
from functools import reduce
import torch
from mat.runner.shared.base_runner import Runner
import random
import copy
import os
import imageio
def _t2n(x):
    return x.detach().cpu().numpy()

class SMACRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        super(SMACRunner, self).__init__(config)
        self.rollout_save_interval = 1
        self._rollout_cache = []
        self._rollout_chunk_idx = 0
        self.collect_eval_rollouts = getattr(self.all_args, "collect_eval_rollouts", False)
        self._eval_rollout_idx = 0

    def run2(self):
        eval_runs = getattr(self.all_args, "eval_runs", 100)
        if self.collect_eval_rollouts and eval_runs > 1:
            original_eval_episodes = self.all_args.eval_episodes
            self.all_args.eval_episodes = original_eval_episodes * eval_runs
            self.eval(0)
            self.all_args.eval_episodes = original_eval_episodes
            return
        for _ in range(eval_runs):
            self.eval(0)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)

        # Use the same reset done in warmup to keep buffer and env in sync.
        obs = self.buffer.obs[0].copy()
        share_obs = self.buffer.share_obs[0].copy()
        available_actions = self.buffer.available_actions[0].copy()
        previous_obs = obs.copy()
        self.noise_rate = 0
        #available_actions = copy.deepcopy(available_now)
        # available_actions already set from warmup

        
        

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                # ensure actions respect availability; fallback to first available if invalid
                actions = actions.astype(np.int64)
                for i in range(len(actions)):
                    for j in range(len(actions[i])):
                        act_idx = actions[i][j][0] if actions[i][j].shape else actions[i][j]
                        avail = available_actions[i][j]
                        if avail[act_idx] == 0:
                            # pick the first valid action to avoid env assertion
                            fallback = int(np.nonzero(avail)[0][0]) if np.any(avail) else 0
                            actions[i][j] = fallback
                    
                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)

                # wandb.log({"rewards": np.mean(rewards)})
                # print("rewards: ", rewards.shape)
                # print shapoes
                # print("obs: ", obs.shape)
                # print("share_obs: ", share_obs.shape)
                # print("available_actions: ", available_actions.shape)
                self.noise_rate = self.calculate_noise_rate(episode, episodes)
                if step >0:
                    for i in range(len(obs)):
                        # if random.random() < self.noise_rate:
                        pass
                            # obs[i] = previous_obs[i] 

                # wandb.log({"noise_rate": self.noise_rate})
                
                # previous_obs = copy.deepcopy(obs)
                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic 
       
                 
                
                # insert data into buffer
                self.insert(data)

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            if getattr(self.trainer, "use_distillation", False):
                scheduled_coef = self.trainer.schedule_student_rl_coef(total_num_steps)
                if self.use_wandb and getattr(self.trainer, "student_rl_linear_schedule", False):
                    wandb.log({"student_rl_coef": scheduled_coef}, step=total_num_steps)

            # compute return and update network
            self.compute()
            if getattr(self.all_args, "save_train_rollouts", False):
                self.dump_rollout_npz(total_num_steps, episode)
            train_infos = self.train()
            
            # post process
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save(episode)

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Map {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.map_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                battles_won = []
                battles_game = []
                incre_battles_won = []
                incre_battles_game = []

                for i, info in enumerate(infos):
                    if 'battles_won' in info[0].keys():
                        battles_won.append(info[0]['battles_won'])
                        incre_battles_won.append(info[0]['battles_won']-last_battles_won[i])
                    if 'battles_game' in info[0].keys():
                        battles_game.append(info[0]['battles_game'])
                        incre_battles_game.append(info[0]['battles_game']-last_battles_game[i])

                incre_win_rate = np.sum(incre_battles_won)/np.sum(incre_battles_game) if np.sum(incre_battles_game)>0 else 0.0
                print("incre win rate is {}.".format(incre_win_rate))
                if self.use_wandb:
                    wandb.log({"incre_win_rate": incre_win_rate}, step=total_num_steps)
                else:
                    self.writter.add_scalars("incre_win_rate", {"incre_win_rate": incre_win_rate}, total_num_steps)

                last_battles_game = battles_game
                last_battles_won = battles_won

                train_infos['dead_ratio'] = 1 - self.buffer.active_masks.sum() / reduce(lambda x, y: x*y, list(self.buffer.active_masks.shape)) 
                
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)
                if getattr(self.trainer, "use_distillation", False):
                    self.eval_student(total_num_steps)

    def calculate_noise_rate(self, episode,episodes):
        """Calculate the noise rate for exploration."""
        if not self.gradual:
            return self.final_noise_rate
        else:
            return min(self.final_noise_rate * (episode/episodes*2),self.final_noise_rate)

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()

        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                            np.concatenate(self.buffer.obs[step]),
                                            np.concatenate(self.buffer.rnn_states[step]),
                                            np.concatenate(self.buffer.rnn_states_critic[step]),
                                            np.concatenate(self.buffer.masks[step]),
                                            np.concatenate(self.buffer.available_actions[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])
        
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards, masks, bad_masks, active_masks, available_actions)

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    @torch.no_grad()
    def dump_rollout_npz(self, total_num_steps, episode):
        rollout_dir = os.path.join(str(self.run_dir), "rollouts")
        os.makedirs(rollout_dir, exist_ok=True)

        buffer = self.buffer
        data = {
            "obs": buffer.obs[:-1],
            "share_obs": buffer.share_obs[:-1],
            "actions": buffer.actions,
            "action_log_probs": buffer.action_log_probs,
            "values": buffer.value_preds[:-1],
            "rewards": buffer.rewards,
            "returns": buffer.returns[:-1],
            "advantages": buffer.advantages,
            "masks": buffer.masks[:-1],
            "masks_next": buffer.masks[1:],
            "active_masks": buffer.active_masks[1:],
            "bad_masks": buffer.bad_masks[1:],
            "rnn_states": buffer.rnn_states[:-1],
            "rnn_states_critic": buffer.rnn_states_critic[:-1],
        }

        if buffer.available_actions is not None:
            data["available_actions"] = buffer.available_actions[:-1]

        if self.algorithm_name in ["mat", "mat_dec"] and self.trainer.policy.action_type == "Discrete":
            obs = buffer.obs[:-1].reshape(-1, self.num_agents, *buffer.obs.shape[3:])
            actions = buffer.actions.reshape(-1, self.num_agents, *buffer.actions.shape[3:])
            available_actions = None
            if buffer.available_actions is not None:
                available_actions = buffer.available_actions[:-1].reshape(
                    -1, self.num_agents, buffer.available_actions.shape[-1]
                )
            logits = self.trainer.policy.transformer.compute_discrete_logits(
                obs, actions, available_actions
            )
            logits = _t2n(logits).reshape(
                buffer.actions.shape[0],
                self.n_rollout_threads,
                self.num_agents,
                -1
            )
            data["action_logits"] = logits

        self._rollout_cache.append(data)
        if len(self._rollout_cache) < self.rollout_save_interval:
            return

        chunk = self._rollout_cache
        stacked = {}
        for key in chunk[0].keys():
            stacked[key] = np.stack([c[key] for c in chunk], axis=0)

        file_path = os.path.join(
            rollout_dir,
            f"smac_rollout_chunk_{self._rollout_chunk_idx}_steps_{total_num_steps}.npz"
        )
        np.savez_compressed(file_path, **stacked)
        self._rollout_cache = []
        self._rollout_chunk_idx += 1

    def _compute_gae(self, rewards, values, masks_next, next_value):
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        gae = 0
        for step in reversed(range(rewards.shape[0])):
            delta = rewards[step] + self.all_args.gamma * next_value * masks_next[step] - values[step]
            gae = delta + self.all_args.gamma * self.all_args.gae_lambda * masks_next[step] * gae
            advantages[step] = gae
            returns[step] = gae + values[step]
            next_value = values[step]
        return advantages, returns

    def _debug_render_path(self, total_num_steps):
        render_dir = getattr(self.all_args, "debug_render_dir", "debug_renders")
        if not os.path.isabs(render_dir):
            render_dir = os.path.join(str(self.run_dir), render_dir)
        os.makedirs(render_dir, exist_ok=True)
        filename = "eval_steps_{}.gif".format(total_num_steps)
        return os.path.join(render_dir, filename)

    def _capture_debug_frame(self):
        frames = self.eval_envs.render("rgb_array")
        frames = np.asarray(frames)
        if frames.ndim == 5:
            return frames[0, 0]
        if frames.ndim == 4:
            return frames[0]
        if frames.ndim == 3:
            return frames
        raise ValueError("Unexpected debug render frame shape: {}".format(frames.shape))

    def _save_debug_render(self, frames, total_num_steps):
        if len(frames) == 0:
            return
        fps = max(int(getattr(self.all_args, "debug_render_fps", 8)), 1)
        path = self._debug_render_path(total_num_steps)
        imageio.mimsave(path, frames, duration=1.0 / fps)
        print("debug render saved at {}.".format(path))
        if self.use_wandb:
            wandb.save(path)
    
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        save_debug_render = getattr(self.all_args, "save_debug_render", False)
        debug_render_frames = []
        debug_render_step = 0
        debug_render_done = False
        debug_render_episode_limit = max(int(getattr(self.all_args, "debug_render_episodes", 1)), 0)
        debug_render_interval = max(int(getattr(self.all_args, "debug_render_interval", 1)), 1)
        if save_debug_render and debug_render_episode_limit > 0:
            debug_render_frames.append(self._capture_debug_frame())

        if self.collect_eval_rollouts and self.algorithm_name not in ["mat", "mat_dec"]:
            raise NotImplementedError("eval rollout collection is only supported for MAT.")

        eval_obs_list = []
        eval_share_obs_list = []
        eval_actions_list = []
        eval_action_log_probs_list = []
        eval_values_list = []
        eval_rewards_list = []
        eval_dones_list = []
        eval_masks_next_list = []
        eval_available_actions_list = []

        while True:
            eval_obs_for_policy = eval_obs
            if self.noise_std > 0:
                noise = np.random.normal(0.0, self.noise_std, size=eval_obs.shape).astype(
                    eval_obs.dtype, copy=False
                )
                # Avoid in-place modification in case the env reuses the same array buffer internally.
                eval_obs_for_policy = eval_obs + noise
                
            self.trainer.prep_rollout()
            if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec" or self.algorithm_name == "pld":
                if self.collect_eval_rollouts:
                    eval_values, eval_actions, eval_action_log_probs, eval_rnn_states, _ = \
                        self.trainer.policy.get_actions(np.concatenate(eval_share_obs),
                                                        np.concatenate(eval_obs_for_policy),
                                                        np.concatenate(eval_rnn_states),
                                                        np.concatenate(eval_rnn_states),
                                                        np.concatenate(eval_masks),
                                                        np.concatenate(eval_available_actions),
                                                        deterministic=True)
                    eval_values = np.array(np.split(_t2n(eval_values), self.n_eval_rollout_threads))
                    eval_action_log_probs = np.array(np.split(_t2n(eval_action_log_probs), self.n_eval_rollout_threads))
                    eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
                    eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
                else:
                    eval_actions, eval_rnn_states = \
                        self.trainer.policy.act(np.concatenate(eval_share_obs),
                                                np.concatenate(eval_obs_for_policy),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                np.concatenate(eval_available_actions),
                                                deterministic=True)
                    eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
                    eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            elif self.algorithm_name == "r_mappo":
                eval_actions, eval_rnn_states = \
                    self.trainer.policy.act(np.concatenate(eval_obs_for_policy),
                                            np.concatenate(eval_rnn_states),
                                            np.concatenate(eval_masks),
                                            np.concatenate(eval_available_actions),
                                            deterministic=True)
                eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
                eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            else:
                raise NotImplementedError
            
            for i in range(len(eval_actions)):
                for j in range(len(eval_actions[i])):
                    if eval_available_actions[i][j][eval_actions[i][j]] == 0:
                        valid = np.nonzero(eval_available_actions[i][j])[0]
                        eval_actions[i][j] = valid[0] if len(valid) > 0 else 0
        
            if self.collect_eval_rollouts:
                eval_obs_list.append(eval_obs.copy())
                eval_share_obs_list.append(eval_share_obs.copy())
                eval_available_actions_list.append(eval_available_actions.copy())
                eval_actions_list.append(eval_actions.copy())
                eval_action_log_probs_list.append(eval_action_log_probs.copy())
                eval_values_list.append(eval_values.copy())
            
            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
            one_episode_rewards.append(eval_rewards)
            debug_render_step += 1
            if (
                save_debug_render
                and not debug_render_done
                and debug_render_episode_limit > 0
                and debug_render_step % debug_render_interval == 0
            ):
                debug_render_frames.append(self._capture_debug_frame())

            

            # for i in range(len(eval_obs)):
            #     if random.random() < self.eval_noise_rate:
                     
            #         eval_obs[i] = previous_eval_obs[i]
            # previous_eval_obs =  eval_obs.copy()

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            if self.collect_eval_rollouts:
                eval_rewards_list.append(eval_rewards.copy())
                eval_dones_list.append(eval_dones.copy())
                eval_masks_next_list.append(eval_masks.copy())

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    one_episode_rewards = []
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1
                    if save_debug_render and eval_episode >= debug_render_episode_limit:
                        debug_render_done = True

            if eval_episode >= self.all_args.eval_episodes:
                if save_debug_render:
                    self._save_debug_render(debug_render_frames, total_num_steps)
                if self.collect_eval_rollouts:
                    rollout_dir = os.path.join(str(self.run_dir), "rollouts")
                    os.makedirs(rollout_dir, exist_ok=True)

                    obs_arr = np.stack(eval_obs_list, axis=0)
                    share_obs_arr = np.stack(eval_share_obs_list, axis=0)
                    actions_arr = np.stack(eval_actions_list, axis=0)
                    action_log_probs_arr = np.stack(eval_action_log_probs_list, axis=0)
                    values_arr = np.stack(eval_values_list, axis=0)
                    rewards_arr = np.stack(eval_rewards_list, axis=0)
                    dones_arr = np.stack(eval_dones_list, axis=0)
                    masks_next_arr = np.stack(eval_masks_next_list, axis=0)
                    available_actions_arr = np.stack(eval_available_actions_list, axis=0)

                    next_value = self.trainer.policy.get_values(np.concatenate(eval_share_obs),
                                                                np.concatenate(eval_obs),
                                                                np.concatenate(eval_rnn_states),
                                                                np.concatenate(eval_masks),
                                                                np.concatenate(eval_available_actions))
                    next_value = np.array(np.split(_t2n(next_value), self.n_eval_rollout_threads))
                    advantages, returns = self._compute_gae(rewards_arr, values_arr, masks_next_arr, next_value)

                    data = {
                        "obs": obs_arr,
                        "share_obs": share_obs_arr,
                        "actions": actions_arr,
                        "action_log_probs": action_log_probs_arr,
                        "values": values_arr,
                        "rewards": rewards_arr,
                        "dones": dones_arr,
                        "masks_next": masks_next_arr,
                        "advantages": advantages,
                        "returns": returns,
                        "available_actions": available_actions_arr,
                    }

                    if self.trainer.policy.action_type == "Discrete":
                        logits = self.trainer.policy.transformer.compute_discrete_logits(
                            obs_arr.reshape(-1, self.num_agents, obs_arr.shape[-1]),
                            actions_arr.reshape(-1, self.num_agents, actions_arr.shape[-1]),
                            available_actions_arr.reshape(-1, self.num_agents, available_actions_arr.shape[-1])
                        )
                        logits = _t2n(logits).reshape(
                            obs_arr.shape[0],
                            self.n_eval_rollout_threads,
                            self.num_agents,
                            -1
                        )
                        data["action_logits"] = logits

                    file_path = os.path.join(
                        rollout_dir,
                        f"smac_eval_rollout_{self._eval_rollout_idx}_steps_{total_num_steps}.npz"
                    )
                    np.savez_compressed(file_path, **data)
                    self._eval_rollout_idx += 1

                if getattr(self.all_args, "save_replay", False):
                    self.eval_envs.save_replay()
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards}                
                self.log_env(eval_env_infos, total_num_steps)
                eval_win_rate = eval_battles_won/eval_episode
                print("eval win rate is {}.".format(eval_win_rate))
                if self.use_wandb:
                    wandb.log({"eval_win_rate": eval_win_rate}, step=total_num_steps)
                else:
                    self.writter.add_scalars("eval_win_rate", {"eval_win_rate": eval_win_rate}, total_num_steps)
                break

    @torch.no_grad()
    def eval_student(self, total_num_steps):
        """Evaluate student policies under SMAC evaluation settings."""
        if not getattr(self.trainer, "use_distillation", False):
            return
        if self.eval_envs is None:
            return

        eval_episode = 0
        eval_episode_rewards = []
        one_episode_rewards = []
        student_battles_won = 0

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        student_hidden_size = 0
        if hasattr(self.trainer.student_policy[0], "actor_rnn"):
            student_hidden_size = self.trainer.student_policy[0].actor_rnn.hidden_size
        
        student_hidden_states = [
            torch.zeros(
                1,
                self.n_eval_rollout_threads,
                student_hidden_size,
                device=self.device
            )
            for student_policy in self.trainer.student_policy
        ]

        while True:
            eval_obs_for_policy = eval_obs
            if self.noise_std > 0:
                noise = np.random.normal(0.0, self.noise_std, size=eval_obs.shape).astype(
                    eval_obs.dtype, copy=False
                )
                eval_obs_for_policy = eval_obs + noise

            actions_per_agent = []
            for agent_id, student_policy in enumerate(self.trainer.student_policy):
                agent_obs = torch.as_tensor(
                    eval_obs_for_policy[:, agent_id, :],
                    dtype=torch.float32,
                    device=self.device
                )
                action_tensor, _, hidden_state = student_policy.act(
                    agent_obs,
                    student_hidden_states[agent_id],
                    deterministic=True
                )
                student_hidden_states[agent_id] = hidden_state
                actions_per_agent.append(action_tensor.cpu().numpy())

            eval_actions = np.stack(actions_per_agent, axis=1).astype(np.int32)

            for env_idx in range(len(eval_actions)):
                for agent_idx in range(len(eval_actions[env_idx])):
                    if eval_available_actions[env_idx][agent_idx][eval_actions[env_idx][agent_idx]] == 0:
                        valid = np.nonzero(eval_available_actions[env_idx][agent_idx])[0]
                        eval_actions[env_idx][agent_idx] = valid[0] if valid.size > 0 else 0

            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
            one_episode_rewards.append(eval_rewards)

            eval_dones_env = np.all(eval_dones, axis=1)

            for agent_id in range(self.num_agents):
                agent_done_np = eval_dones[:, agent_id].astype(bool)
                if agent_done_np.any():
                    student_hidden_states[agent_id][:, agent_done_np, :] = 0

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    one_episode_rewards = []
                    if eval_infos[eval_i][0]['won']:
                        student_battles_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.array(eval_episode_rewards)
                student_env_infos = {'student_eval_average_episode_rewards': eval_episode_rewards}
                self.log_env(student_env_infos, total_num_steps)
                student_win_rate = student_battles_won / eval_episode
                if self.use_wandb:
                    wandb.log({"student_eval_win_rate": student_win_rate}, step=total_num_steps)
                else:
                    self.writter.add_scalars("student_eval_win_rate", {"student_eval_win_rate": student_win_rate}, total_num_steps)
                break


class PLDSMACHRunner(SMACRunner):
    """Offline PLD training runner using dataset rollouts; env interaction only for eval."""
    def run(self):
        start = time.time()
        updates = self.all_args.pld_updates

        for update in range(updates):
            train_infos = self.train()
            total_num_steps = update + 1

            if update % self.log_interval == 0:
                end = time.time()
                print("\n PLD updates {}/{} total, elapsed {:.2f}s.\n"
                      .format(update, updates, end - start))
                self.log_train(train_infos, total_num_steps)

            if update % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def log_train(self, train_infos, total_num_steps):
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)
