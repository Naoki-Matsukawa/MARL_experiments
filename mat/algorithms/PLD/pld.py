import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from mat.utils.util import get_gard_norm, huber_loss, mse_loss
from mat.utils.valuenorm import ValueNorm
from mat.algorithms.utils.util import check


import argparse
import os
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


from VAE import MultiAgentVAE


class GlobalObsDataset(Dataset):
    """
    1サンプル = ある時刻の全エージェント観測 [N, obs_dim]
    """
    def __init__(self, x: torch.Tensor):
        # x: [M, N, obs_dim]
        assert x.dim() == 3
        self.x = x

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.x[idx]


def _load_npz(path: str) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _to_tensor(x: Any, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


def load_global_obs(
    data_path: str,
    device: torch.device,
    key: Optional[str] = None,
) -> torch.Tensor:
    """
    .npz or .pt から全体観測を読み込んで [M, N, obs_dim] に整形して返す。
    想定キー:
      - "global_obs" or "obs" or "joint_obs"
    形状の想定:
      - [M, N, obs_dim]
      - [E, T, N, obs_dim] (→ [E*T, N, obs_dim] に潰す)
    """
    ext = os.path.splitext(data_path)[1].lower()

    if ext == ".npz":
        raw = _load_npz(data_path)
    elif ext in (".pt", ".pth"):
        raw = torch.load(data_path, map_location="cpu")
        if not isinstance(raw, dict):
            raise ValueError("Expected a dict in .pt/.pth")
    else:
        raise ValueError(f"Unsupported extension: {ext}")

    cand_keys = [key] if key else ["global_obs", "joint_obs", "obs"]
    x = None
    for k in cand_keys:
        if k is None:
            continue
        if k in raw:
            x = raw[k]
            break
    if x is None:
        raise KeyError(f"Could not find any of keys: {cand_keys} in {data_path}")

    x = _to_tensor(x, device=device, dtype=torch.float32)

    if x.dim() == 4:
        # [E, T, N, obs_dim] -> [E*T, N, obs_dim]
        E, T, N, D = x.shape
        x = x.view(E * T, N, D)
    elif x.dim() == 3:
        pass
    else:
        raise ValueError(f"Unexpected obs tensor shape: {tuple(x.shape)}")

    return x


def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor,
             recon_loss: str = "mse") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    recon_x, x: [B, N, obs_dim]
    mu, logvar: [B, latent_dim]
    """
    if recon_loss == "mse":
        rec = F.mse_loss(recon_x, x, reduction="mean")
    elif recon_loss == "l1":
        rec = F.l1_loss(recon_x, x, reduction="mean")
    else:
        raise ValueError(recon_loss)

    # KL(q(z|x) || N(0,I)) : 標準的なVAE
    kld = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    total = rec + kld
    return total, rec, kld


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to rollout data (.npz or .pt)")
    parser.add_argument("--key", type=str, default=None, help="Key name for global obs array (optional)")
    parser.add_argument("--out", type=str, default="global_vae.pt", help="Output checkpoint path")

    parser.add_argument("--num_agents", type=int, required=True)
    parser.add_argument("--obs_dim", type=int, required=True)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--agent_feature_dim", type=int, default=64)

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--recon_loss", type=str, default="mse", choices=["mse", "l1"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    x = load_global_obs(args.data, device=device, key=args.key)
    # shape check
    if x.size(1) != args.num_agents or x.size(2) != args.obs_dim:
        raise ValueError(f"obs shape mismatch: got {tuple(x.shape)}, "
                         f"expected [M,{args.num_agents},{args.obs_dim}]")

    ds = GlobalObsDataset(x)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    vae = MultiAgentVAE(
        num_agents=args.num_agents,
        obs_dim=args.obs_dim,
        latent_dim=args.latent_dim,
        agent_feature_dim=args.agent_feature_dim,
    ).to(device)

    optim = torch.optim.Adam(vae.parameters(), lr=args.lr)

    vae.train()
    for ep in range(1, args.epochs + 1):
        total_sum = 0.0
        rec_sum = 0.0
        kld_sum = 0.0
        n = 0

        for batch in dl:
            # batch: [B, N, obs_dim]
            optim.zero_grad()
            recon_x, mu, logvar = vae(batch)
            loss, rec, kld = vae_loss(recon_x, batch, mu, logvar, recon_loss=args.recon_loss)
            loss.backward()
            optim.step()

            bs = batch.size(0)
            total_sum += loss.item() * bs
            rec_sum += rec.item() * bs
            kld_sum += kld.item() * bs
            n += bs

        print(f"[ep {ep:03d}] total={total_sum/n:.6f} rec={rec_sum/n:.6f} kld={kld_sum/n:.6f}")

    ckpt = {
        "state_dict": vae.state_dict(),
        "config": {
            "num_agents": args.num_agents,
            "obs_dim": args.obs_dim,
            "latent_dim": args.latent_dim,
            "agent_feature_dim": args.agent_feature_dim,
        },
    }
    torch.save(ckpt, args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
