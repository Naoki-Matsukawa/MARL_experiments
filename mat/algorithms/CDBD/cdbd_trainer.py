import numpy as np
import torch
import torch.nn as nn

from mat.algorithms.utils.util import check
from mat.utils.util import get_gard_norm


class CDBDTrainer:
    """
    PPO student update with frozen teacher critic values and latent distillation.
    """

    def __init__(self, args, policy, num_agents=None, device=torch.device("cpu")):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.num_agents = num_agents

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.belief_coef = float(getattr(args, "cdbd_belief_coef", 1.0))

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_policy_active_masks = args.use_policy_active_masks
        self.value_normalizer = None

    def ppo_update(self, sample, update_actor=True):
        if len(sample) == 12:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                adv_targ, available_actions_batch = sample
        else:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                adv_targ, available_actions_batch, _ = sample

        del value_preds_batch, return_batch

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        _, action_log_probs, dist_entropy, student_latent, teacher_latent = self.policy.evaluate_actions(
            share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )

        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        belief_error = (student_latent - teacher_latent.detach()).pow(2).mean(dim=-1, keepdim=True)
        if self._use_policy_active_masks:
            belief_loss = (belief_error * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            belief_loss = belief_error.mean()

        total_loss = policy_loss - dist_entropy * self.entropy_coef + self.belief_coef * belief_loss

        self.policy.actor_optimizer.zero_grad()
        if update_actor:
            total_loss.backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        return policy_loss, belief_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, buffer, update_actor=True):
        advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {
            "policy_loss": 0,
            "belief_loss": 0,
            "dist_entropy": 0,
            "actor_grad_norm": 0,
            "ratio": 0,
        }

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(
                    advantages,
                    self.num_mini_batch,
                    self.data_chunk_length,
                )
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                policy_loss, belief_loss, dist_entropy, actor_grad_norm, imp_weights = self.ppo_update(
                    sample,
                    update_actor,
                )
                train_info["policy_loss"] += policy_loss.item()
                train_info["belief_loss"] += belief_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += float(actor_grad_norm)
                train_info["ratio"] += imp_weights.mean().item()

        num_updates = self.ppo_epoch * self.num_mini_batch
        for key in train_info:
            train_info[key] /= num_updates
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.teacher.eval()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.teacher.eval()
