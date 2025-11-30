"""Flax/JAX counterpart of the PyTorch Multi-Agent Transformer (MAT).

The goal of this module is not to be a drop-in replacement for the PyTorch
implementation, but rather to illustrate how the Transformer-style encoder and
decoder used in MAT can be implemented with Flax's functional API.  The module
keeps the high-level structure (per-agent encoder, masked decoder, optional
per-agent MLP actor) while exposing a simpler functional interface that returns
value estimates and policy logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import jax.numpy as jnp
from flax import linen as nn


def orthogonal_init(scale: float = 1.0):
    """Orthogonal kernel initializer matching the PyTorch version."""

    return nn.initializers.orthogonal(scale)


@dataclass
class TransformerConfig:
    """Configuration container shared by encoder/decoder modules."""

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
    """Multi-head attention with optional causal masking."""

    n_embd: int
    n_head: int
    n_agent: int
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
            causal = jnp.tril(jnp.ones((q_len, kv_len), dtype=bool))
            mask = causal.reshape(1, 1, q_len, kv_len)

        return attn(query, key, value, mask=mask)


class FeedForward(nn.Module):
    """Simple MLP used inside Transformer blocks."""

    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for i, feat in enumerate(self.features):
            x = nn.Dense(
                feat,
                kernel_init=orthogonal_init(),
                bias_init=nn.initializers.zeros,
            )(x)
            if i < len(self.features) - 1:
                x = nn.gelu(x)
        return x


class EncodeBlock(nn.Module):
    """Single encoder block (SA + MLP)."""

    n_embd: int
    n_head: int
    n_agent: int

    @nn.compact
    def __call__(self, x, *, deterministic: bool = True):
        attn = SelfAttention(
            self.n_embd, self.n_head, self.n_agent, masked=False, name="self_attn"
        )(x, x, x, deterministic=deterministic)
        x = nn.LayerNorm()(x + attn)

        mlp = FeedForward([self.n_embd, self.n_embd])(x)
        x = nn.LayerNorm()(x + mlp)

        return x


class DecodeBlock(nn.Module):
    """Decoder block mirroring the PyTorch MAT design."""

    n_embd: int
    n_head: int
    n_agent: int

    @nn.compact
    def __call__(self, actions, enc_rep, *, deterministic: bool = True):
        # Masked self-attention over previously generated actions.
        masked = SelfAttention(
            self.n_embd, self.n_head, self.n_agent, masked=True, name="masked_attn"
        )(actions, actions, actions, deterministic=deterministic)
        actions = nn.LayerNorm()(actions + masked)

        # Cross attention from encoder representation (queries) to decoder states.
        cross = SelfAttention(
            self.n_embd, self.n_head, self.n_agent, masked=False, name="cross_attn"
        )(enc_rep, actions, actions, deterministic=deterministic)
        rep = nn.LayerNorm()(enc_rep + cross)

        mlp = FeedForward([self.n_embd, self.n_embd])(rep)
        rep = nn.LayerNorm()(rep + mlp)

        return rep


class Encoder(nn.Module):
    """Agent-wise encoder producing shared value features."""

    config: TransformerConfig

    @nn.compact
    def __call__(self, state, obs, *, deterministic: bool = True):
        if self.config.encode_state:
            x = nn.LayerNorm()(state)
            x = nn.Dense(
                self.config.n_embd,
                kernel_init=orthogonal_init(),
                bias_init=nn.initializers.zeros,
            )(x)
        else:
            x = nn.LayerNorm()(obs)
            x = nn.Dense(
                self.config.n_embd,
                kernel_init=orthogonal_init(),
                bias_init=nn.initializers.zeros,
            )(x)

        x = nn.gelu(x)
        x = nn.LayerNorm()(x)

        for idx in range(self.config.n_block):
            x = EncodeBlock(
                self.config.n_embd, self.config.n_head, self.config.n_agent, name=f"enc_{idx}"
            )(x, deterministic=deterministic)

        value = FeedForward([self.config.n_embd, 1])(x)
        return value, x


class AgentMLP(nn.Module):
    """Utility MLP for the decoder-only policy."""

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
    """Masked decoder that can optionally fall back to per-agent MLPs."""

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
            else:
                logits = []
                for agent in range(self.config.n_agent):
                    logits.append(
                        AgentMLP(
                            self.config.obs_dim,
                            self.config.n_embd,
                            self.config.action_dim,
                            name=f"agent_{agent}",
                        )(obs[:, agent, :])
                    )
                logits = jnp.stack(logits, axis=1)
            return logits

        if self.config.action_type == "Discrete":
            action_features = nn.Dense(
                self.config.n_embd,
                kernel_init=orthogonal_init(),
                use_bias=False,
            )(actions)
        else:
            action_features = nn.Dense(
                self.config.n_embd,
                kernel_init=orthogonal_init(),
            )(actions)
        x = nn.LayerNorm()(nn.gelu(action_features))

        for idx in range(self.config.n_block):
            x = DecodeBlock(
                self.config.n_embd, self.config.n_head, self.config.n_agent, name=f"dec_{idx}"
            )(x, enc_rep, deterministic=deterministic)

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
    """High-level Flax policy that mirrors the PyTorch MAT API."""

    config: TransformerConfig

    def setup(self):
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)

    def encode(self, state, obs, *, deterministic: bool = True):
        return self.encoder(state, obs, deterministic=deterministic)

    def decode(self, actions, enc_rep, obs, *, deterministic: bool = True):
        return self.decoder(actions, enc_rep, obs, deterministic=deterministic)

    def __call__(self, state, obs, actions, *, deterministic: bool = True):
        values, enc_rep = self.encode(state, obs, deterministic=deterministic)
        decoder_out = self.decode(actions, enc_rep, obs, deterministic=deterministic)
        return decoder_out, values
