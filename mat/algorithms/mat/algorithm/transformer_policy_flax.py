"""Flax/JAX reimplementation of the MAT Transformer policy.

This module mirrors the PyTorch ``TransformerPolicy`` API at a high level but uses
``MultiAgentTransformerFlax`` under the hood.  The resulting class lets you
instantiate the encoder/decoder stack, keep track of Flax parameters, and compute
action/value outputs that can be consumed by a JAX-based PPO trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax

from mat.algorithms.mat.algorithm.ma_transformer_flax import (
    MultiAgentTransformerFlax,
    TransformerConfig,
)


LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def _to_jnp(array, dtype=jnp.float32):
    return jnp.asarray(array, dtype=dtype)


@dataclass
class PolicyState:
    """Container that mirrors the attributes we mutate during training."""

    params: Any
    opt_state: optax.OptState
    rng: jax.Array


class TransformerPolicyFlax:
    """JAX/Flax analogue of :class:`TransformerPolicy`."""

    def __init__(self, args, obs_space, cent_obs_space, act_space, num_agents, seed: int = 0):
        self.algorithm_name = args.algorithm_name
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.num_agents = num_agents
        self.encode_state = args.encode_state
        self.dec_actor = args.dec_actor
        self.share_actor = args.share_actor
        self.n_block = args.n_block
        self.n_embd = args.n_embd
        self.n_head = args.n_head
        self.action_type = "Continuous" if act_space.__class__.__name__ == "Box" else "Discrete"

        self.obs_dim = obs_space.shape[0] if hasattr(obs_space, "shape") else obs_space.n
        self.share_obs_dim = (
            cent_obs_space.shape[0] if hasattr(cent_obs_space, "shape") else cent_obs_space.n
        )
        if self.action_type == "Discrete":
            self.act_dim = act_space.n
            self.act_num = 1
        else:
            self.act_dim = act_space.shape[0]
            self.act_num = self.act_dim

        config = TransformerConfig(
            state_dim=self.share_obs_dim,
            obs_dim=self.obs_dim,
            action_dim=self.act_dim,
            n_agent=num_agents,
            n_block=self.n_block,
            n_embd=self.n_embd,
            n_head=self.n_head,
            encode_state=self.encode_state,
            action_type=self.action_type,
            dec_actor=self.dec_actor,
            share_actor=self.share_actor,
        )
        self.model = MultiAgentTransformerFlax(config)

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        dummy_state = jnp.zeros((1, num_agents, self.share_obs_dim), dtype=jnp.float32)
        dummy_obs = jnp.zeros((1, num_agents, self.obs_dim), dtype=jnp.float32)
        dummy_action_tokens = jnp.zeros((1, num_agents, self.act_dim), dtype=jnp.float32)
        variables = self.model.init(init_rng, dummy_state, dummy_obs, dummy_action_tokens)

        self.params = variables["params"]
        self.tx = optax.adam(self.lr, weight_decay=self.weight_decay)
        self.opt_state = self.tx.init(self.params)
        self.rng = rng

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _forward(self, params, state, obs, action_tokens=None, deterministic: bool = True):
        if action_tokens is None:
            action_tokens = jnp.zeros((state.shape[0], self.num_agents, self.act_dim), dtype=jnp.float32)
        decoder_out, values = self.model.apply(
            {"params": params}, state, obs, action_tokens, deterministic=deterministic
        )
        return decoder_out, values

    def _mask_logits(self, logits, available_actions):
        if available_actions is None:
            return logits
        mask = jnp.where(available_actions > 0, 0.0, -1e9)
        return logits + mask

    def _discrete_log_probs(self, logits, actions):
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        actions = jnp.squeeze(actions, axis=-1) if actions.ndim == 3 else actions
        gather = jnp.take_along_axis(
            log_probs, jnp.expand_dims(actions, axis=-1), axis=-1
        )
        return gather

    def _discrete_entropy(self, logits, available_actions):
        probs = jax.nn.softmax(self._mask_logits(logits, available_actions), axis=-1)
        log_probs = jnp.log(jnp.clip(probs, a_min=1e-12))
        entropy = -jnp.sum(probs * log_probs, axis=-1, keepdims=True)
        return entropy

    def _continuous_log_probs(self, mean, std, actions):
        var = std**2
        log_scale = jnp.log(std)
        return -0.5 * (((actions - mean) ** 2) / var + 2 * log_scale + jnp.log(2 * jnp.pi))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def policy_state(self) -> PolicyState:
        return PolicyState(params=self.params, opt_state=self.opt_state, rng=self.rng)

    def update_state(self, state: PolicyState):
        self.params = state.params
        self.opt_state = state.opt_state
        self.rng = state.rng

    def apply_gradients(self, grads):
        updates, self.opt_state = self.tx.update(grads, self.opt_state, self.params)
        self.params = optax.apply_updates(self.params, updates)

    def evaluate_actions(
        self,
        params,
        cent_obs,
        obs,
        actions,
        available_actions: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        state = _to_jnp(cent_obs)
        obs = _to_jnp(obs)
        actions = _to_jnp(actions)
        if available_actions is not None:
            available_actions = _to_jnp(available_actions)

        decoder_out, values = self._forward(params, state, obs, deterministic=deterministic)

        if self.action_type == "Discrete":
            logits = self._mask_logits(decoder_out, available_actions)
            log_probs = self._discrete_log_probs(logits, actions)
            entropy = self._discrete_entropy(decoder_out, available_actions)
        else:
            means, std = decoder_out
            std = jnp.clip(std, a_min=jnp.exp(LOG_STD_MIN), a_max=jnp.exp(LOG_STD_MAX))
            log_probs = self._continuous_log_probs(means, std, actions)
            entropy = 0.5 + 0.5 * jnp.log(2 * jnp.pi) + jnp.log(std)

        return values, log_probs, entropy

    def sample_actions(
        self,
        params,
        cent_obs,
        obs,
        available_actions: Optional[jnp.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        state = _to_jnp(cent_obs)
        obs = _to_jnp(obs)
        if available_actions is not None:
            available_actions = _to_jnp(available_actions)

        decoder_out, values = self._forward(params, state, obs, deterministic=deterministic)

        rng, sample_rng = jax.random.split(self.rng)
        self.rng = rng

        if self.action_type == "Discrete":
            logits = self._mask_logits(decoder_out, available_actions)
            if deterministic:
                actions = jnp.argmax(logits, axis=-1, keepdims=True)
            else:
                sample = jax.random.categorical(sample_rng, logits=logits, axis=-1)
                actions = jnp.expand_dims(sample, axis=-1)
            log_probs = self._discrete_log_probs(logits, actions)
        else:
            means, std = decoder_out
            std = jnp.clip(std, a_min=jnp.exp(LOG_STD_MIN), a_max=jnp.exp(LOG_STD_MAX))
            if deterministic:
                actions = means
            else:
                sample = jax.random.normal(sample_rng, means.shape)
                actions = means + sample * std
            log_probs = self._continuous_log_probs(means, std, actions)

        return values, actions, log_probs

    def optimizer_step(self, grads):
        self.apply_gradients(grads)

