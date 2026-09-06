"""All-in-one Flax implementation of MAT (model, policy wrapper, PPO trainer).

This module co-locates the previously separate components:
    - Multi-agent Transformer encoder/decoder stack.
    - Policy helper that handles parameter/optimizer state plus action sampling.
    - PPO-style trainer that optimizes the policy with clipped objectives.

Having them in a single file simplifies experimentation when working entirely
in the Flax/JAX stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax


# -----------------------------------------------------------------------------
# Transformer core
# -----------------------------------------------------------------------------


def orthogonal_init(scale: float = 1.0):
    return nn.initializers.orthogonal(scale)


@dataclass
class TransformerConfig:
    state_dim: int
    obs_dim: int
    action_dim: int
    n_agent: int
    n_block: int = 1
    n_embd: int = 64
    n_head: int = 1
    encode_state: bool = False
    action_type: str = "Discrete"
    dec_actor: bool = False
    share_actor: bool = False


class SelfAttention(nn.Module):
    n_embd: int
    n_head: int
    masked: bool = False

    @nn.compact
    def __call__(self, query, key, value, *, deterministic: bool = True):
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.n_head,
            kernel_init=orthogonal_init(),
            deterministic=deterministic,
        )
        mask = None
        if self.masked:
            q_len = query.shape[1]
            kv_len = key.shape[1]
            mask = jnp.tril(jnp.ones((q_len, kv_len), dtype=bool)).reshape(1, 1, q_len, kv_len)
        return attn(query, key, value, mask=mask)


class FeedForward(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat, kernel_init=orthogonal_init())(x)
            if i < len(self.features) - 1:
                x = nn.gelu(x)
        return x


class EncodeBlock(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, x, *, deterministic: bool = True):
        attn = SelfAttention(
            self.config.n_embd, self.config.n_head, masked=False
        )(x, x, x, deterministic=deterministic)
        x = nn.LayerNorm()(x + attn)
        mlp = FeedForward([self.config.n_embd, self.config.n_embd])(x)
        x = nn.LayerNorm()(x + mlp)
        return x


class DecodeBlock(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, actions, enc_rep, *, deterministic: bool = True):
        masked = SelfAttention(
            self.config.n_embd, self.config.n_head, masked=True
        )(actions, actions, actions, deterministic=deterministic)
        actions = nn.LayerNorm()(actions + masked)

        cross = SelfAttention(
            self.config.n_embd, self.config.n_head, masked=False
        )(enc_rep, actions, actions, deterministic=deterministic)
        rep = nn.LayerNorm()(enc_rep + cross)

        mlp = FeedForward([self.config.n_embd, self.config.n_embd])(rep)
        rep = nn.LayerNorm()(rep + mlp)
        return rep


class Encoder(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, state, obs, *, deterministic: bool = True):
        inputs = state if self.config.encode_state else obs
        x = nn.LayerNorm()(inputs)
        x = nn.Dense(self.config.n_embd, kernel_init=orthogonal_init())(x)
        x = nn.gelu(x)
        x = nn.LayerNorm()(x)

        for idx in range(self.config.n_block):
            x = EncodeBlock(self.config, name=f"enc_{idx}")(x, deterministic=deterministic)

        value = FeedForward([self.config.n_embd, 1])(x)
        return value, x


class AgentMLP(nn.Module):
    obs_dim: int
    n_embd: int
    action_dim: int

    @nn.compact
    def __call__(self, obs):
        x = nn.LayerNorm()(obs)
        x = nn.Dense(self.n_embd, kernel_init=orthogonal_init())(x)
        x = nn.gelu(x)
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.n_embd, kernel_init=orthogonal_init())(x)
        x = nn.gelu(x)
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.action_dim, kernel_init=orthogonal_init())(x)
        return x


class Decoder(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, actions, enc_rep, obs, *, deterministic: bool = True):
        if self.config.dec_actor:
            batch = obs.shape[0]
            if self.config.share_actor:
                obs_flat = obs.reshape(batch * self.config.n_agent, self.config.obs_dim)
                logits = AgentMLP(
                    self.config.obs_dim, self.config.n_embd, self.config.action_dim, name="shared_actor"
                )(obs_flat)
                logits = logits.reshape(batch, self.config.n_agent, self.config.action_dim)
                return logits
            logits = []
            for agent in range(self.config.n_agent):
                logits.append(
                    AgentMLP(
                        self.config.obs_dim, self.config.n_embd, self.config.action_dim, name=f"agent_{agent}"
                    )(obs[:, agent, :])
                )
            return jnp.stack(logits, axis=1)

        feature_layer = nn.Dense(
            self.config.n_embd, kernel_init=orthogonal_init(), use_bias=self.config.action_type != "Discrete"
        )
        x = feature_layer(actions)
        x = nn.gelu(x)
        x = nn.LayerNorm()(x)

        for idx in range(self.config.n_block):
            x = DecodeBlock(self.config, name=f"dec_{idx}")(x, enc_rep, deterministic=deterministic)

        logits = FeedForward([self.config.n_embd, self.config.action_dim])(x)
        if self.config.action_type != "Discrete":
            log_std = self.param(
                "log_std",
                lambda rng, shape: jnp.zeros(shape, dtype=jnp.float32),
                (self.config.action_dim,),
            )
            std = jnp.exp(jnp.clip(log_std, a_min=-5.0, a_max=2.0))
            return logits, std
        return logits


class MultiAgentTransformerFlax(nn.Module):
    config: TransformerConfig

    def setup(self):
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)

    def __call__(self, state, obs, actions, *, deterministic: bool = True):
        value, rep = self.encoder(state, obs, deterministic=deterministic)
        decoder_out = self.decoder(actions, rep, obs, deterministic=deterministic)
        return decoder_out, value


# -----------------------------------------------------------------------------
# Policy wrapper
# -----------------------------------------------------------------------------


LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def _to_jnp(array, dtype=jnp.float32):
    return jnp.asarray(array, dtype=dtype)


class TransformerPolicyFlax:
    """Handles parameter state, optimizer, and action/value queries."""

    def __init__(self, config: TransformerConfig, lr: float, weight_decay: float, num_agents: int, seed: int = 0):
        self.config = config
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_agents = num_agents

        rng = jax.random.PRNGKey(seed)
        rng, init_key = jax.random.split(rng)
        dummy_state = jnp.zeros((1, num_agents, config.state_dim))
        dummy_obs = jnp.zeros((1, num_agents, config.obs_dim))
        dummy_actions = jnp.zeros((1, num_agents, config.action_dim))

        self.model = MultiAgentTransformerFlax(config)
        variables = self.model.init(init_key, dummy_state, dummy_obs, dummy_actions)
        self.params = variables["params"]

        self.tx = optax.adam(learning_rate=lr, weight_decay=weight_decay)
        self.opt_state = self.tx.init(self.params)
        self.rng = rng

    # internal helpers --------------------------------------------------------
    def _forward(self, params, state, obs, actions, deterministic: bool = True):
        return self.model.apply({"params": params}, state, obs, actions, deterministic=deterministic)

    def _mask_logits(self, logits, available_actions):
        if available_actions is None:
            return logits
        mask = jnp.where(available_actions > 0, 0.0, -1e9)
        return logits + mask

    def _discrete_log_probs(self, logits, actions):
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        squeezed = jnp.squeeze(actions, axis=-1) if actions.ndim == 3 else actions
        gathered = jnp.take_along_axis(
            log_probs, jnp.expand_dims(squeezed, axis=-1), axis=-1
        )
        return gathered

    def _continuous_log_probs(self, mean, std, actions):
        var = std**2
        log_scale = jnp.log(std)
        return -0.5 * (((actions - mean) ** 2) / var + 2 * log_scale + jnp.log(2 * jnp.pi))

    # public API --------------------------------------------------------------
    def evaluate_actions(
        self,
        params,
        state,
        obs,
        actions,
        available_actions: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ):
        state = _to_jnp(state)
        obs = _to_jnp(obs)
        actions = _to_jnp(actions)
        if available_actions is not None:
            available_actions = _to_jnp(available_actions)
        decoder_out, values = self._forward(params, state, obs, actions, deterministic=deterministic)

        if self.config.action_type == "Discrete":
            logits = self._mask_logits(decoder_out, available_actions)
            log_probs = self._discrete_log_probs(logits, actions)
            probs = jax.nn.softmax(logits, axis=-1)
            entropy = -jnp.sum(probs * jnp.log(jnp.clip(probs, a_min=1e-12)), axis=-1, keepdims=True)
        else:
            means, std = decoder_out
            std = jnp.clip(std, a_min=jnp.exp(LOG_STD_MIN), a_max=jnp.exp(LOG_STD_MAX))
            log_probs = self._continuous_log_probs(means, std, actions)
            entropy = 0.5 + 0.5 * jnp.log(2 * jnp.pi) + jnp.log(std)

        return values, log_probs, entropy

    def sample_actions(
        self,
        params,
        state,
        obs,
        available_actions: Optional[jnp.ndarray] = None,
        deterministic: bool = False,
    ):
        state = _to_jnp(state)
        obs = _to_jnp(obs)
        if available_actions is not None:
            available_actions = _to_jnp(available_actions)
        decoder_out, values = self._forward(
            params,
            state,
            obs,
            jnp.zeros((state.shape[0], self.num_agents, self.config.action_dim)),
            deterministic=deterministic,
        )
        rng, sample_key = jax.random.split(self.rng)
        self.rng = rng

        if self.config.action_type == "Discrete":
            logits = self._mask_logits(decoder_out, available_actions)
            if deterministic:
                actions = jnp.argmax(logits, axis=-1, keepdims=True)
            else:
                sample = jax.random.categorical(sample_key, logits=logits, axis=-1)
                actions = jnp.expand_dims(sample, axis=-1)
            log_probs = self._discrete_log_probs(logits, actions)
        else:
            means, std = decoder_out
            std = jnp.clip(std, a_min=jnp.exp(LOG_STD_MIN), a_max=jnp.exp(LOG_STD_MAX))
            if deterministic:
                actions = means
            else:
                sample = jax.random.normal(sample_key, means.shape)
                actions = means + sample * std
            log_probs = self._continuous_log_probs(means, std, actions)

        return values, actions, log_probs

    def apply_gradients(self, grads):
        updates, self.opt_state = self.tx.update(grads, self.opt_state, self.params)
        self.params = optax.apply_updates(self.params, updates)


# -----------------------------------------------------------------------------
# PPO trainer
# -----------------------------------------------------------------------------


@dataclass
class PPOConfig:
    clip_param: float = 0.05
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    use_clipped_value_loss: bool = True
    use_value_active_masks: bool = False
    use_policy_active_masks: bool = False


class MATTrainerFlax:
    def __init__(self, config: PPOConfig, policy: TransformerPolicyFlax):
        self.cfg = config
        self.policy = policy

    def _value_loss(self, values, value_preds, returns, active_masks):
        clipped = value_preds + jnp.clip(values - value_preds, -self.cfg.clip_param, self.cfg.clip_param)
        loss1 = (returns - values) ** 2
        loss2 = (returns - clipped) ** 2
        base_loss = jnp.maximum(loss1, loss2) if self.cfg.use_clipped_value_loss else loss1
        if self.cfg.use_value_active_masks:
            denom = jnp.maximum(jnp.sum(active_masks), 1.0)
            return jnp.sum(base_loss * active_masks) / denom
        return jnp.mean(base_loss)

    def _ppo_loss(self, params, batch):
        values, log_probs, entropy = self.policy.evaluate_actions(
            params,
            batch["state"],
            batch["obs"],
            batch["actions"],
            batch.get("available_actions"),
            deterministic=False,
        )
        imp_weights = jnp.exp(log_probs - batch["old_log_probs"])
        surr1 = imp_weights * batch["advantages"]
        surr2 = jnp.clip(imp_weights, 1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param) * batch["advantages"]

        if self.cfg.use_policy_active_masks:
            mask = batch["active_masks"]
            denom = jnp.maximum(jnp.sum(mask), 1.0)
            policy_loss = -jnp.sum(jnp.minimum(surr1, surr2) * mask) / denom
        else:
            policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

        value_loss = self._value_loss(values, batch["value_preds"], batch["returns"], batch["active_masks"])
        entropy_loss = jnp.mean(entropy)

        total_loss = policy_loss - entropy_loss * self.cfg.entropy_coef + value_loss * self.cfg.value_loss_coef
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
            scale = jnp.where(grad_norm > self.cfg.max_grad_norm, self.cfg.max_grad_norm / (grad_norm + 1e-6), 1.0)
            grads = jax.tree_util.tree_map(lambda g: g * scale, grads)

        self.policy.apply_gradients(grads)
        metrics["loss"] = loss
        return {k: float(v) for k, v in metrics.items()}
