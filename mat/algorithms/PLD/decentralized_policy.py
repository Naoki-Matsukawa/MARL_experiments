# decentralized_models.py
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Agent local obs -> latent (match VAE latent)
# ---------------------------
class AgentObsEncoder(nn.Module):
    """
    o_i -> z_i_pred
    Input:  obs [B, N, obs_dim] or [B, T, N, obs_dim]
    Output: z   [B, N, latent_dim] or [B, T, N, latent_dim]
    """
    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        use_layernorm: bool = True,
    ):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 3:
            B, N, D = obs.shape
            x = obs.reshape(B * N, D)
            z = self.net(x).reshape(B, N, -1)
            return z
        elif obs.dim() == 4:
            B, T, N, D = obs.shape
            x = obs.reshape(B * T * N, D)
            z = self.net(x).reshape(B, T, N, -1)
            return z
        else:
            raise ValueError(f"Unexpected obs shape: {tuple(obs.shape)}")


# ---------------------------
# Policy: (raw obs + predicted latent) -> action dist
# ---------------------------
class AgentPolicy(nn.Module):
    """
    Input:  obs [B,N,obs_dim] + z [B,N,latent_dim]
    Output:
      discrete: logits [B,N,act_dim]
      continuous: mu/log_std [B,N,act_dim] (diag Gaussian)
    """
    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        act_dim: int,
        action_type: Literal["discrete", "continuous"] = "discrete",
        hidden_dim: int = 256,
        num_layers: int = 2,
        use_layernorm: bool = True,
        log_std_bounds: Tuple[float, float] = (-5.0, 2.0),
    ):
        super().__init__()
        self.action_type = action_type
        self.act_dim = act_dim
        self.log_std_bounds = log_std_bounds

        in_dim = obs_dim + latent_dim
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        if action_type == "discrete":
            self.head = nn.Linear(hidden_dim, act_dim)
        elif action_type == "continuous":
            self.mu_head = nn.Linear(hidden_dim, act_dim)
            self.log_std_head = nn.Linear(hidden_dim, act_dim)
        else:
            raise ValueError(action_type)

    def forward(self, obs: torch.Tensor, z_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        if obs.dim() != 3 or z_pred.dim() != 3:
            raise ValueError("AgentPolicy expects [B,N,*] tensors")

        x = torch.cat([obs, z_pred], dim=-1)  # [B,N,obs+latent]
        B, N, D = x.shape
        h = self.backbone(x.reshape(B * N, D)).reshape(B, N, -1)

        if self.action_type == "discrete":
            logits = self.head(h)
            return {"logits": logits}
        else:
            mu = self.mu_head(h)
            log_std = self.log_std_head(h)
            lo, hi = self.log_std_bounds
            log_std = torch.clamp(log_std, lo, hi)
            return {"mu": mu, "log_std": log_std}


# ---------------------------
# Losses
# ---------------------------
def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x.mean()
    # mask broadcast
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    x = x * mask
    denom = mask.sum().clamp_min(1.0)
    return x.sum() / denom


def latent_mimic_loss(
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    loss_type: Literal["mse", "cosine"] = "mse",
) -> torch.Tensor:
    """
    z_pred:   [B,T,N,latent] or [B,N,latent]
    z_target: same shape
    mask:     [B,T] or [B,T,N] or [B,N] (broadcast可)
    """
    if loss_type == "mse":
        per = (z_pred - z_target).pow(2).mean(dim=-1)  # [...,]
    elif loss_type == "cosine":
        per = 1.0 - F.cosine_similarity(z_pred, z_target, dim=-1)
    else:
        raise ValueError(loss_type)
    return masked_mean(per, mask)


def advantage_weighted_imitation_loss(
    policy_out: Dict[str, torch.Tensor],
    actions_teacher: torch.Tensor,
    advantages: torch.Tensor,
    action_type: Literal["discrete", "continuous"] = "discrete",
    temperature: float = 1.0,
    mask: Optional[torch.Tensor] = None,
    use_exp_weight: bool = True,
) -> torch.Tensor:
    """
    Shapes (time series版):
      policy_out:
        discrete: logits [B,T,N,A]
        cont:     mu/log_std [B,T,N,A]
      actions_teacher:
        discrete: [B,T,N]
        cont:     [B,T,N,A]
      advantages: [B,T,N] or broadcastable
      mask: [B,T] or [B,T,N] or broadcastable
    """
    if use_exp_weight:
        w = torch.exp(torch.clamp(advantages / max(temperature, 1e-6), -20.0, 20.0))
    else:
        w = torch.clamp(advantages, min=0.0)

    if action_type == "discrete":
        logits = policy_out["logits"]
        A = logits.size(-1)
        ce = F.cross_entropy(
            logits.reshape(-1, A),
            actions_teacher.reshape(-1).long(),
            reduction="none",
        ).reshape_as(actions_teacher)  # [B,T,N] or [B,N]
        per = ce
    else:
        mu = policy_out["mu"]
        log_std = policy_out["log_std"]
        std = torch.exp(log_std)
        a = actions_teacher
        nll = 0.5 * (((a - mu) / (std + 1e-8)) ** 2 + 2.0 * log_std + math.log(2.0 * math.pi))
        per = nll.sum(dim=-1)  # [...,]

    # apply weights + mask
    while w.dim() < per.dim():
        w = w.unsqueeze(-1)
    loss = per * w
    return masked_mean(loss, mask)


# ---------------------------
# Combined decentralized agent model
# ---------------------------
class DecentralizedAgentModel(nn.Module):
    """
    - local encoder: o_i -> z_i_pred
    - policy: (o_i, z_i_pred) -> action distribution
    """
    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        act_dim: int,
        action_type: Literal["discrete", "continuous"] = "discrete",
        enc_hidden: int = 256,
        pi_hidden: int = 256,
    ):
        super().__init__()
        self.encoder = AgentObsEncoder(obs_dim=obs_dim, latent_dim=latent_dim, hidden_dim=enc_hidden)
        self.policy = AgentPolicy(obs_dim=obs_dim, latent_dim=latent_dim, act_dim=act_dim,
                                  action_type=action_type, hidden_dim=pi_hidden)
        self.action_type = action_type

    def forward(self, obs: torch.Tensor):
        """
        obs: [B,T,N,obs_dim] or [B,N,obs_dim]
        returns:
          z_pred: same leading dims + latent
          policy_out: dict of logits or (mu, log_std) with same leading dims
        """
        z_pred = self.encoder(obs)

        if obs.dim() == 3:
            out = self.policy(obs, z_pred)
            return z_pred, out

        # obs.dim()==4: policy expects [B*,N,*] に潰して戻す
        B, T, N, D = obs.shape
        z_bt = z_pred.reshape(B * T, N, -1)
        obs_bt = obs.reshape(B * T, N, D)
        out = self.policy(obs_bt, z_bt)

        if "logits" in out:
            out["logits"] = out["logits"].reshape(B, T, N, -1)
        else:
            out["mu"] = out["mu"].reshape(B, T, N, -1)
            out["log_std"] = out["log_std"].reshape(B, T, N, -1)

        return z_pred, out


@dataclass
class DistillBatch:
    """
    典型形:
      obs_local:        [B,T,N,obs_dim]
      vae_latent:       [B,T,latent_dim]  (全体からVAEで出したz)
      actions_teacher:  discrete [B,T,N] or continuous [B,T,N,act_dim]
      advantages:       [B,T,N]  (adv付模倣用)
      mask:             [B,T] or [B,T,N] (padding等)
    """
    obs_local: torch.Tensor
    vae_latent: torch.Tensor
    actions_teacher: torch.Tensor
    advantages: torch.Tensor
    mask: Optional[torch.Tensor] = None


def compute_total_loss(
    model: DecentralizedAgentModel,
    batch: DistillBatch,
    lambda_latent: float = 1.0,
    lambda_policy: float = 1.0,
    latent_loss_type: Literal["mse", "cosine"] = "mse",
    aw_temperature: float = 1.0,
    use_exp_weight: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    VAEの潜在 z_vae は [B,T,latent] を想定し、各agentへ複製してターゲットにする。
    """
    z_pred, policy_out = model(batch.obs_local)  # z_pred: [B,T,N,latent]

    # [B,T,latent] -> [B,T,N,latent]
    z_t = batch.vae_latent.unsqueeze(2).expand_as(z_pred)

    L_lat = latent_mimic_loss(z_pred, z_t, mask=batch.mask, loss_type=latent_loss_type)

    L_pi = advantage_weighted_imitation_loss(
        policy_out=policy_out,
        actions_teacher=batch.actions_teacher,
        advantages=batch.advantages,
        action_type=model.action_type,
        temperature=aw_temperature,
        mask=batch.mask,
        use_exp_weight=use_exp_weight,
    )

    total = lambda_latent * L_lat + lambda_policy * L_pi
    return {"total": total, "latent": L_lat, "policy": L_pi}
