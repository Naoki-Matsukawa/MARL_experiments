import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from mat.utils.util import get_gard_norm, huber_loss, mse_loss
from mat.utils.valuenorm import ValueNorm
from mat.algorithms.utils.util import check
from mat.algorithms.mat.rnn import DiscreteRecurrentPolicy, ContinuousRecurrentPolicy

class MATTrainer:
    """
    Trainer class for MAT to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 num_agents,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.num_agents = num_agents

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        self.dec_actor = args.dec_actor

        student_lr = getattr(args, "student_lr", getattr(policy.optimizer, "defaults", {}).get("lr", 1e-3))
        self.use_distillation = getattr(args, "distillation", False)
        self.student_value_coef = getattr(args, "student_value_coef", 1.0)
        if self.policy.action_type == 'Discrete':
            policy_cls = DiscreteRecurrentPolicy
            student_kwargs = {}
        else:
            policy_cls = ContinuousRecurrentPolicy
            student_kwargs = {"log_std_init": getattr(args, "student_log_std_init", 0.0)}

        if self.use_distillation:
            value_dim = getattr(getattr(self.policy.transformer, "encoder", None), "n_embd", None)
            student_kwargs = {**student_kwargs, "value_dim": value_dim}
            self.student_policy = [
                policy_cls(self.policy.obs_dim, self.policy.act_dim, lr=student_lr, **student_kwargs).to(device)
                for _ in range(num_agents)
            ]
            self.student_optimizers = [student.optimizer for student in self.student_policy]
            self.student_hidden_state = [None for _ in range(num_agents)]
        else:
            self.student_policy = None
            self.student_optimizers = []
            self.student_hidden_state = None
        self.student_kl_coef = getattr(args, "student_kl_coef", 1.0)
        self.student_rl_coef = getattr(args, "student_rl_coef", 0.1)
        self.student_clip_param = getattr(args, "student_clip_param", self.clip_param)

        if self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """

        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                    self.clip_param)

        if self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        # if self._use_value_active_masks and not self.dec_actor:
        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch, 
                                                                              rnn_states_batch, 
                                                                              rnn_states_critic_batch, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_loss = (-torch.sum(torch.min(surr1, surr2),
                                      dim=-1,
                                      keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        loss = policy_loss - dist_entropy * self.entropy_coef + value_loss * self.value_loss_coef

        self.policy.optimizer.zero_grad()
        loss.backward()

        if self._use_max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(self.policy.transformer.parameters(), self.max_grad_norm)
        else:
            grad_norm = get_gard_norm(self.policy.transformer.parameters())

        self.policy.optimizer.step()

        return value_loss, grad_norm, policy_loss, dist_entropy, grad_norm, imp_weights

    def train(self, buffer):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        advantages_copy = buffer.advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (buffer.advantages - mean_advantages) / (std_advantages + 1e-5)
        

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        if self.use_distillation:
            train_info['student_kl_loss'] = 0
            train_info['student_rl_loss'] = 0
            train_info['student_value_loss'] = 0


        # share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        # value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        # adv_targ, available_actions_batch = sample

        for _ in range(self.ppo_epoch):
            data_generator = buffer.feed_forward_generator_transformer(advantages, self.num_mini_batch)

            for sample in data_generator:

                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample)

                share_obs_batch, obs_batch, _, _, actions_batch, _, _, _, active_masks_batch, _, adv_batch, available_actions_batch = sample

                if self.use_distillation and self.policy.action_type == 'Discrete':
                    dist_info = self.policy.get_action_distribution(
                        share_obs_batch,
                        obs_batch,
                        actions_batch,
                        available_actions_batch
                    )
                    teacher_probs = dist_info['probs'].detach()
                    teacher_encoder = dist_info.get('encoder_rep')
                    teacher_encoder = teacher_encoder.detach() if teacher_encoder is not None else None

                    obs_agent_view = self._reshape_agent_view(obs_batch)
                    actions_agent_view = self._reshape_agent_view(actions_batch)
                    active_masks_agent_view = self._reshape_agent_view(active_masks_batch)
                    adv_agent_view = self._reshape_agent_view(adv_batch) if adv_batch is not None else None
                    teacher_encoder_agent_view = self._reshape_agent_view(teacher_encoder) if teacher_encoder is not None else None

                    for agent_id in range(self.num_agents):
                        agent_obs = obs_agent_view[:, agent_id, ...]
                        agent_teacher_probs = teacher_probs[:, agent_id, ...]
                        agent_actions = actions_agent_view[:, agent_id, ...]
                        agent_active_masks = active_masks_agent_view[:, agent_id, ...] if active_masks_agent_view is not None else None
                        agent_advantages = adv_agent_view[:, agent_id, ...] if adv_agent_view is not None else None
                        agent_teacher_encoder = teacher_encoder_agent_view[:, agent_id, ...] if teacher_encoder_agent_view is not None else None

                        student_loss, hidden_state, student_info = self._calculate_discrete_student_loss(
                            self.student_policy[agent_id],
                            agent_obs,
                            agent_teacher_probs,
                            agent_actions,
                            agent_advantages,
                            agent_active_masks,
                            agent_teacher_encoder,
                            self.student_hidden_state[agent_id]
                        )
                        train_info['student_kl_loss'] += student_info['kl_loss']
                        train_info['student_rl_loss'] += student_info['rl_loss']
                        train_info['student_value_loss'] += student_info['value_loss']
                        self.student_hidden_state[agent_id] = self._detach_hidden(hidden_state)

                        self.student_optimizers[agent_id].zero_grad()
                        student_loss.backward()
                        self.student_optimizers[agent_id].step()
                elif self.use_distillation and self.policy.action_type != 'Discrete':
                    dist_info = self.policy.get_action_distribution(
                        share_obs_batch,
                        obs_batch,
                        actions_batch,
                        available_actions_batch
                    )
                    teacher_means = dist_info['means'].detach()
                    teacher_log_stds = dist_info['log_stds'].detach()
                    teacher_encoder = dist_info.get('encoder_rep')
                    teacher_encoder = teacher_encoder.detach() if teacher_encoder is not None else None

                    obs_agent_view = self._reshape_agent_view(obs_batch)
                    actions_agent_view = self._reshape_agent_view(actions_batch)
                    active_masks_agent_view = self._reshape_agent_view(active_masks_batch)
                    adv_agent_view = self._reshape_agent_view(adv_batch) if adv_batch is not None else None
                    teacher_encoder_agent_view = self._reshape_agent_view(teacher_encoder) if teacher_encoder is not None else None
                    for agent_id in range(self.num_agents):
                        agent_obs = obs_agent_view[:, agent_id, ...]
                        agent_actions = actions_agent_view[:, agent_id, ...]
                        agent_active_masks = active_masks_agent_view[:, agent_id, ...] if active_masks_agent_view is not None else None
                        agent_advantages = adv_agent_view[:, agent_id, ...] if adv_agent_view is not None else None
                        agent_teacher_mean = teacher_means[:, agent_id, ...]
                        agent_teacher_log_std = teacher_log_stds[:, agent_id, ...]
                        agent_teacher_encoder = teacher_encoder_agent_view[:, agent_id, ...] if teacher_encoder_agent_view is not None else None

                        student_loss, hidden_state, student_info = self._calculate_continuous_student_loss(
                            self.student_policy[agent_id],
                            agent_obs,
                            agent_teacher_mean,
                            agent_teacher_log_std,
                            agent_actions,
                            agent_advantages,
                            agent_active_masks,
                            agent_teacher_encoder,
                            self.student_hidden_state[agent_id]
                        )
                        train_info['student_kl_loss'] += student_info['kl_loss']
                        train_info['student_rl_loss'] += student_info['rl_loss']
                        train_info['student_value_loss'] += student_info['value_loss']
                        self.student_hidden_state[agent_id] = self._detach_hidden(hidden_state)

                        self.student_optimizers[agent_id].zero_grad()
                        student_loss.backward()
                        self.student_optimizers[agent_id].step()
                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def prep_training(self):
        self.policy.train()

    def prep_rollout(self):
        self.policy.eval()
    
    def _reshape_agent_view(self, tensor):
        """
        Reshape flattened (mini_batch * num_agents, ...) tensors back to (mini_batch, num_agents, ...).
        """
        if tensor is None:
            return None
        batch_size = tensor.shape[0]
        if batch_size % self.num_agents != 0:
            # already shaped as (mini_batch, num_agents, ...)
            return tensor
        mini_batch = batch_size // self.num_agents
        if torch.is_tensor(tensor):
            return tensor.contiguous().view(mini_batch, self.num_agents, *tensor.shape[1:])
        elif isinstance(tensor, np.ndarray):
            return tensor.reshape(mini_batch, self.num_agents, *tensor.shape[1:])
        else:
            raise TypeError(f"Unsupported tensor type for reshape: {type(tensor)}")

    def _detach_hidden(self, hidden_state):
        if hidden_state is None:
            return None
        if isinstance(hidden_state, tuple):
            return tuple(h.detach() if h is not None else None for h in hidden_state)
        return hidden_state.detach()

    def _calculate_discrete_student_loss(self, student_policy, obs_batch, teacher_probs_batch, actions_batch,
                                         advantages_batch=None, active_masks_batch=None, teacher_encoder_batch=None,
                                         hidden_state=None):
        student_info = {'kl_loss': 0.0, 'rl_loss': 0.0, 'value_loss': 0.0}

        obs_batch = check(obs_batch).to(**self.tpdv)
        teacher_probs_batch = teacher_probs_batch.to(**self.tpdv)
        actions_batch = check(actions_batch).to(**self.tpdv)

        if obs_batch.dim() == 2:
            obs_batch = obs_batch.unsqueeze(1)
        actor_hidden, critic_hidden = hidden_state if isinstance(hidden_state, tuple) else (hidden_state, None)

        student_probs, new_actor_hidden = student_policy(obs_batch, actor_hidden)
        student_probs = student_probs.squeeze(1)
        student_values, new_critic_hidden = student_policy.forward_value(obs_batch, critic_hidden)
        student_values = student_values.squeeze(1)

        kl_elementwise = F.kl_div(
            torch.log(student_probs + 1e-8),
            teacher_probs_batch,
            reduction='none'
        ).sum(-1, keepdim=True)

        if active_masks_batch is not None:
            active_masks_batch = check(active_masks_batch).to(**self.tpdv)
            kl_loss = (kl_elementwise * active_masks_batch).sum() / (active_masks_batch.sum() + 1e-8)
        else:
            kl_loss = kl_elementwise.mean()

        rl_loss = torch.tensor(0.0, device=self.device)
        if self.student_rl_coef > 0 and advantages_batch is not None:
            advantages_batch = check(advantages_batch).to(**self.tpdv)
            if advantages_batch.dim() > 1:
                advantages_batch = advantages_batch.squeeze(-1)
            if actions_batch.dim() > 1:
                actions_batch = actions_batch.squeeze(-1)

            actions_long = actions_batch.long()
            student_selected = torch.gather(student_probs, -1, actions_long.unsqueeze(-1)).squeeze(-1)
            log_pi = torch.log(student_selected + 1e-8)
            pg_terms = -(log_pi * advantages_batch)

            if active_masks_batch is not None:
                active_flat = active_masks_batch.squeeze(-1)
                rl_loss = (pg_terms * active_flat).sum() / (active_flat.sum() + 1e-8)
            else:
                rl_loss = pg_terms.mean()

        value_loss = torch.tensor(0.0, device=self.device)
        if teacher_encoder_batch is not None:
            teacher_encoder_batch = teacher_encoder_batch.to(**self.tpdv)
            if teacher_encoder_batch.dim() == 2:
                teacher_encoder_batch = teacher_encoder_batch.unsqueeze(1)
            teacher_encoder_batch = teacher_encoder_batch.squeeze(1).detach()
            value_elementwise = F.mse_loss(student_values, teacher_encoder_batch, reduction='none').sum(-1, keepdim=True)
            if active_masks_batch is not None:
                active_masks_batch = check(active_masks_batch).to(**self.tpdv)
                value_loss = (value_elementwise * active_masks_batch).sum() / (active_masks_batch.sum() + 1e-8)
            else:
                value_loss = value_elementwise.mean()

        total_loss = self.student_kl_coef * kl_loss + self.student_rl_coef * rl_loss + self.student_value_coef * value_loss
        student_info['kl_loss'] = kl_loss.item()
        student_info['rl_loss'] = rl_loss.item()
        student_info['value_loss'] = value_loss.item()
        new_hidden = (new_actor_hidden, new_critic_hidden)
        return total_loss, new_hidden, student_info

    def _calculate_continuous_student_loss(self, student_policy, obs_batch, teacher_mean_batch, teacher_log_std_batch,
                                           actions_batch, advantages_batch=None, active_masks_batch=None,
                                           teacher_encoder_batch=None, hidden_state=None):
        student_info = {'kl_loss': 0.0, 'rl_loss': 0.0, 'value_loss': 0.0}

        obs_batch = check(obs_batch).to(**self.tpdv)
        teacher_mean_batch = check(teacher_mean_batch).to(**self.tpdv)
        teacher_log_std_batch = check(teacher_log_std_batch).to(**self.tpdv)
        actions_batch = check(actions_batch).to(**self.tpdv)

        if obs_batch.dim() == 2:
            obs_batch = obs_batch.unsqueeze(1)

        actor_hidden, critic_hidden = hidden_state if isinstance(hidden_state, tuple) else (hidden_state, None)

        student_mean, student_log_std, new_actor_hidden = student_policy(obs_batch, actor_hidden)
        student_mean = student_mean.squeeze(1)
        student_log_std = student_log_std.squeeze(1)
        student_values, new_critic_hidden = student_policy.forward_value(obs_batch, critic_hidden)
        student_values = student_values.squeeze(1)

        teacher_mean = teacher_mean_batch
        teacher_log_std = teacher_log_std_batch

        var_teacher = torch.exp(2 * teacher_log_std)
        var_student = torch.exp(2 * student_log_std)

        kl_terms = (student_log_std - teacher_log_std) + \
            (var_teacher + (teacher_mean - student_mean) ** 2) / (2 * var_student) - 0.5
        kl_loss = kl_terms.sum(-1, keepdim=True)

        if active_masks_batch is not None:
            active_masks_batch = check(active_masks_batch).to(**self.tpdv)
            kl_loss = (kl_loss * active_masks_batch).sum() / (active_masks_batch.sum() + 1e-8)
        else:
            kl_loss = kl_loss.mean()

        rl_loss = torch.tensor(0.0, device=self.device)
        if self.student_rl_coef > 0 and advantages_batch is not None:
            advantages_batch = check(advantages_batch).to(**self.tpdv)
            if advantages_batch.dim() > 1:
                advantages_batch = advantages_batch.squeeze(-1)

            student_dist = Normal(student_mean, torch.exp(student_log_std))
            log_probs = student_dist.log_prob(actions_batch).sum(-1)
            pg_terms = -(log_probs * advantages_batch)

            if active_masks_batch is not None:
                active_flat = active_masks_batch.squeeze(-1)
                rl_loss = (pg_terms * active_flat).sum() / (active_flat.sum() + 1e-8)
            else:
                rl_loss = pg_terms.mean()

        value_loss = torch.tensor(0.0, device=self.device)
        if teacher_encoder_batch is not None:
            teacher_encoder_batch = check(teacher_encoder_batch).to(**self.tpdv)
            if teacher_encoder_batch.dim() == 2:
                teacher_encoder_batch = teacher_encoder_batch.unsqueeze(1)
            teacher_encoder_batch = teacher_encoder_batch.squeeze(1).detach()
            value_elementwise = F.mse_loss(student_values, teacher_encoder_batch, reduction='none').sum(-1, keepdim=True)
            if active_masks_batch is not None:
                active_flat = check(active_masks_batch).to(**self.tpdv)
                value_loss = (value_elementwise * active_flat).sum() / (active_flat.sum() + 1e-8)
            else:
                value_loss = value_elementwise.mean()

        total_loss = self.student_kl_coef * kl_loss + self.student_rl_coef * rl_loss + self.student_value_coef * value_loss
        student_info['kl_loss'] = kl_loss.item()
        student_info['rl_loss'] = rl_loss.item()
        student_info['value_loss'] = value_loss.item()
        new_hidden = (new_actor_hidden, new_critic_hidden)
        return total_loss, new_hidden, student_info
