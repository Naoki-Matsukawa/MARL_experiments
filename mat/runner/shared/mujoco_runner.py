import time
import wandb
import numpy as np
from functools import reduce
import torch
from mat.runner.shared.base_runner import Runner
import random

def _t2n(x):
    return x.detach().cpu().numpy()


def faulty_action(action, faulty_node):
    action_fault = action.copy()
    if faulty_node >= 0:
        action_fault[:, faulty_node, :] = 0.
        # action[:, faulty_node, :] = 0.
    # return action
    return action_fault


class MujocoRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        super(MujocoRunner, self).__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        train_episode_rewards = [0 for _ in range(self.n_rollout_threads)]
        done_episodes_rewards = []
        previous_obs = self.envs.reset()[0]
        self.noise_rate = 0

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                # actions = faulty_action(actions, self.all_args.faulty_node)
                #
                # # Obser reward and next obs
                # obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)

                actions_fault = faulty_action(actions, self.all_args.faulty_node)

                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions_fault)

                self.noise_rate = self.calculate_noise_rate(episode, episodes)
                if step >0:
                    for i in range(len(obs)):
                        if random.random() < self.noise_rate:
                            obs[i] = previous_obs[i] 
                wandb.log({"noise_rate": self.noise_rate})

                dones_env = np.all(dones, axis=1)
                reward_env = np.mean(rewards, axis=1).flatten()
                train_episode_rewards += reward_env
                for t in range(self.n_rollout_threads):
                    if dones_env[t]:
                        done_episodes_rewards.append(train_episode_rewards[t])
                        train_episode_rewards[t] = 0

                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic
                previous_obs = obs.copy()
                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save(episode)

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                self.log_train(train_infos, total_num_steps)

                if len(done_episodes_rewards) > 0:
                    aver_episode_rewards = np.mean(done_episodes_rewards)
                    print("some episodes done, average rewards: ", aver_episode_rewards)
                    #self.writter.add_scalars("train_episode_rewards", {"aver_rewards": aver_episode_rewards}, total_num_steps)
                    wandb.log({"train_episode_rewards": aver_episode_rewards}, step=total_num_steps)
                    done_episodes_rewards = []

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                faulty_nodes = self.all_args.eval_faulty_node
                for node in faulty_nodes:
                    self.eval(total_num_steps, node)
                    if getattr(self.trainer, "use_distillation", False):
                        self.eval_student(total_num_steps, node)

    def calculate_noise_rate(self, episode,episodes):
     
        if not self.gradual:
            return self.final_noise_rate
        else:
            return min(self.final_noise_rate * (episode/episodes*2),self.final_noise_rate)


    def warmup(self):
        # reset env
        obs, share_obs, _ = self.envs.reset()

        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                            np.concatenate(self.buffer.obs[step]),
                                            np.concatenate(self.buffer.rnn_states[step]),
                                            np.concatenate(self.buffer.rnn_states_critic[step]),
                                            np.concatenate(self.buffer.masks[step]))
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

        # bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])

        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards, masks, None, active_masks,
                           None)

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
        print("average_step_rewards is {}.".format(train_infos["average_step_rewards"]))
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps, faulty_node):
        eval_episode = 0
        eval_episode_rewards = []
        one_episode_rewards = [0 for _ in range(self.all_args.eval_episodes)]

        eval_obs, eval_share_obs, _ = self.eval_envs.reset()
        eval_rnn_states = np.zeros((self.all_args.eval_episodes, self.num_agents, self.recurrent_N,
                                    self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.all_args.eval_episodes, self.num_agents, 1), dtype=np.float32)
        previous_eval_obs = eval_obs.copy()
        while True:
            self.trainer.prep_rollout()
            eval_actions, eval_rnn_states = \
                self.trainer.policy.act(np.concatenate(eval_share_obs),
                                        np.concatenate(eval_obs),
                                        np.concatenate(eval_rnn_states),
                                        np.concatenate(eval_masks),
                                        deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.all_args.eval_episodes))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.all_args.eval_episodes))

            # Obser reward and next obs
            eval_actions = faulty_action(eval_actions, faulty_node)
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, _ = self.eval_envs.step(eval_actions)
            
            for i in range(len(eval_obs)):
                if random.random() < self.eval_noise_rate:
                    eval_obs[i] = previous_eval_obs[i]
            previous_eval_obs = eval_obs.copy()
            
            eval_rewards = np.mean(eval_rewards, axis=1).flatten()
            one_episode_rewards += eval_rewards

            eval_dones_env = np.all(eval_dones, axis=1)
            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents,
                                                                self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.all_args.eval_episodes, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1),
                                                          dtype=np.float32)

            for eval_i in range(self.all_args.eval_episodes):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(one_episode_rewards[eval_i])
                    one_episode_rewards[eval_i] = 0

            if eval_episode >= self.all_args.eval_episodes:
                key_average = 'faulty_node_' + str(faulty_node) + '/eval_average_episode_rewards'
                key_max = 'faulty_node_' + str(faulty_node) + '/eval_max_episode_rewards'
                eval_env_infos = {key_average: eval_episode_rewards,
                                  key_max: [np.max(eval_episode_rewards)]}                 
                self.log_env(eval_env_infos, total_num_steps)
                print("faulty_node {} eval_average_episode_rewards is {}."
                      .format(faulty_node, np.mean(eval_episode_rewards)))
                break

    @torch.no_grad()
    def eval_student(self, total_num_steps, faulty_node):
        """Evaluate student policies with Mujoco environments."""
        if not getattr(self.trainer, "use_distillation", False):
            return
        if self.eval_envs is None:
            return

        eval_episode = 0
        eval_episode_rewards = []
        one_episode_rewards = [0 for _ in range(self.all_args.eval_episodes)]

        eval_obs, eval_share_obs, _ = self.eval_envs.reset()

        student_hidden_states = [
            torch.zeros(
                1,
                self.all_args.eval_episodes,
                student_policy.rnn.hidden_size,
                device=self.device
            )
            for student_policy in self.trainer.student_policy
        ]

        while True:
            actions_per_agent = []
            for agent_id, student_policy in enumerate(self.trainer.student_policy):
                agent_obs = torch.as_tensor(
                    eval_obs[:, agent_id, :],
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

            eval_actions = np.stack(actions_per_agent, axis=1)
            action_space = self.eval_envs.action_space[0]
            if action_space.__class__.__name__ == 'Discrete':
                eval_actions = eval_actions.astype(np.int32)
                eval_actions_env = np.eye(action_space.n)[eval_actions]
            else:
                eval_actions_env = np.clip(
                    eval_actions,
                    action_space.low,
                    action_space.high
                )
            eval_actions_env = faulty_action(eval_actions_env, faulty_node)

            eval_obs, eval_share_obs, eval_rewards, eval_dones, _, _ = self.eval_envs.step(eval_actions_env)

            eval_rewards = np.mean(eval_rewards, axis=1).flatten()
            one_episode_rewards += eval_rewards

            eval_dones_env = np.all(eval_dones, axis=1)
            for agent_id in range(self.num_agents):
                agent_done_np = eval_dones[:, agent_id].astype(bool)
                if agent_done_np.any():
                    agent_done = torch.as_tensor(agent_done_np, device=self.device, dtype=torch.bool)
                    student_hidden_states[agent_id][:, agent_done, :] = 0

            for eval_i in range(self.all_args.eval_episodes):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(one_episode_rewards[eval_i])
                    one_episode_rewards[eval_i] = 0

            if eval_episode >= self.all_args.eval_episodes:
                key_average = f'faulty_node_{faulty_node}/student_eval_average_episode_rewards'
                key_max = f'faulty_node_{faulty_node}/student_eval_max_episode_rewards'
                student_env_infos = {
                    key_average: eval_episode_rewards,
                    key_max: [np.max(eval_episode_rewards)]
                }
                self.log_env(student_env_infos, total_num_steps)
                print("faulty_node {} student_eval_average_episode_rewards is {}."
                      .format(faulty_node, np.mean(eval_episode_rewards)))
                break
