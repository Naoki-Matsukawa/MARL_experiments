import time
import wandb
import numpy as np
from functools import reduce
import torch
from mat.runner.shared.base_runner import Runner
import random
import copy
def _t2n(x):
    return x.detach().cpu().numpy()

class SMACRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        super(SMACRunner, self).__init__(config)

    def run2(self):
        for episode in range(1):
            self.eval(episode)

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
    def eval(self, total_num_steps):
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            eval_obs_for_policy = eval_obs
            if self.noise_std > 0:
                noise = np.random.normal(0.0, self.noise_std, size=eval_obs.shape).astype(
                    eval_obs.dtype, copy=False
                )
                # Avoid in-place modification in case the env reuses the same array buffer internally.
                eval_obs_for_policy = eval_obs + noise
                
            self.trainer.prep_rollout()
            if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
                eval_actions, eval_rnn_states = \
                    self.trainer.policy.act(np.concatenate(eval_share_obs),
                                            np.concatenate(eval_obs_for_policy),
                                            np.concatenate(eval_rnn_states),
                                            np.concatenate(eval_masks),
                                            np.concatenate(eval_available_actions),
                                            deterministic=True)
            elif self.algorithm_name == "r_mappo":
                eval_actions, eval_rnn_states = \
                    self.trainer.policy.act(np.concatenate(eval_obs_for_policy),
                                            np.concatenate(eval_rnn_states),
                                            np.concatenate(eval_masks),
                                            np.concatenate(eval_available_actions),
                                            deterministic=True)
            else:
                raise NotImplementedError
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            for i in range(len(eval_actions)):
                for j in range(len(eval_actions[i])):
                    if eval_available_actions[i][j][eval_actions[i][j]] == 0:
                        valid = np.nonzero(eval_available_actions[i][j])[0]
                        eval_actions[i][j] = valid[0] if len(valid) > 0 else 0
        
            
            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
            one_episode_rewards.append(eval_rewards)

            

            # for i in range(len(eval_obs)):
            #     if random.random() < self.eval_noise_rate:
                     
            #         eval_obs[i] = previous_eval_obs[i]
            # previous_eval_obs =  eval_obs.copy()

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    one_episode_rewards = []
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                # self.eval_envs.save_replay()
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
