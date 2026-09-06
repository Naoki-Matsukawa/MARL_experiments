"""JAX/Flax PPO trainer for the MAT policy.

This module mirrors the responsibilities of the PyTorch ``MATTrainer`` but targets the
``TransformerPolicyFlax`` interface.  It is intentionally lightweight so users can
experiment with PPO-style updates without re-using the PyTorch autograd stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import jax
import jax.numpy as jnp


def _to_jnp(array, dtype=jnp.float32):
    return jnp.asarray(array, dtype=dtype)


@dataclass
class PPOConfig:
    clip_param: float
    ppo_epoch: int
    num_mini_batch: int
    value_loss_coef: float
    entropy_coef: float
    max_grad_norm: float
    use_clipped_value_loss: bool
    use_value_active_masks: bool
    use_policy_active_masks: bool


class MATTrainerFlax:
    """Simple PPO trainer working with :class:`TransformerPolicyFlax`."""

    def __init__(self, config: PPOConfig, policy):
        self.cfg = config
        self.policy = policy

    def _value_loss(self, values, value_preds, returns, active_masks):
        value_pred_clipped = value_preds + jnp.clip(
            values - value_preds, -self.cfg.clip_param, self.cfg.clip_param
        )

        error_clipped = returns - value_pred_clipped
        error_original = returns - values

        value_loss_clipped = jnp.mean(error_clipped**2)
        value_loss_original = jnp.mean(error_original**2)
        value_loss = (
            jnp.maximum(value_loss_original, value_loss_clipped)
            if self.cfg.use_clipped_value_loss
            else value_loss_original
        )

        if self.cfg.use_value_active_masks:
            denom = jnp.maximum(jnp.sum(active_masks), 1.0)
            value_loss = jnp.sum(value_loss * active_masks) / denom

        return value_loss

    def _ppo_loss(self, params, batch):
        values, log_probs, entropy = self.policy.evaluate_actions(
            params,
            batch["share_obs"],
            batch["obs"],
            batch["actions"],
            batch.get("available_actions"),
            deterministic=False,
        )

        imp_weights = jnp.exp(log_probs - batch["old_action_log_probs"])
        surr1 = imp_weights * batch["advantages"]
        surr2 = jnp.clip(
            imp_weights, 1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param
        ) * batch["advantages"]

        if self.cfg.use_policy_active_masks:
            mask = batch["active_masks"]
            denom = jnp.maximum(jnp.sum(mask), 1.0)
            policy_loss = -jnp.sum(jnp.minimum(surr1, surr2) * mask) / denom
        else:
            policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

        value_loss = self._value_loss(
            values, batch["value_preds"], batch["returns"], batch["active_masks"]
        )
        entropy_loss = jnp.mean(entropy)

        total_loss = (
            policy_loss
            - entropy_loss * self.cfg.entropy_coef
            + value_loss * self.cfg.value_loss_coef
        )

        metrics = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy_loss,
        }

        return total_loss, metrics

    def train_step(self, batch: Dict[str, jnp.ndarray]):
        batch = {k: _to_jnp(v) for k, v in batch.items()}

        loss_fn = lambda params: self._ppo_loss(params, batch)
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(self.policy.params)

        if self.cfg.max_grad_norm is not None and self.cfg.max_grad_norm > 0:
            grad_norm = jnp.sqrt(
                sum([jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)])
            )
            scale = jnp.where(
                grad_norm > self.cfg.max_grad_norm,
                self.cfg.max_grad_norm / (grad_norm + 1e-6),
                1.0,
            )
            grads = jax.tree_util.tree_map(lambda g: g * scale, grads)

        self.policy.apply_gradients(grads)
        metrics["loss"] = loss
        return {k: float(v) for k, v in metrics.items()}
