import os
import glob
import numpy as np
import torch

from mat.algorithms.PLD.VAE import MultiAgentVAE
from mat.algorithms.PLD.decentralized_policy import latent_mimic_loss, advantage_weighted_imitation_loss
from mat.algorithms.utils.util import check


class PLDTrainer:
    """
    Trainer for PLD that uses rollout buffer data.
    """
    def __init__(self, args, policy, num_agents, device=torch.device("cpu")):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.num_agents = num_agents

        self.latent_dim = args.pld_latent_dim
        self.vae = MultiAgentVAE(
            num_agents=num_agents,
            obs_dim=policy.share_obs_dim,
            latent_dim=args.pld_latent_dim,
            agent_feature_dim=args.pld_agent_feature_dim,
        ).to(self.device)

        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=args.pld_lr)

        self.batch_size = args.pld_batch_size
        self.epochs = args.pld_epochs
        self.recon_loss = args.pld_recon_loss
        self.vae_coef = args.pld_vae_coef
        self.latent_coef = args.pld_latent_coef
        self.policy_coef = args.pld_policy_coef
        self.use_exp_weight = args.pld_use_exp_weight
        self.temperature = args.pld_temperature
        self.dataset_dir = args.pld_dataset_dir
        self.freeze_vae = args.pld_freeze_vae
        self.vae_path = args.pld_vae_path
        self.disable_obs_encoder = args.pld_disable_obs_encoder

        self.value_normalizer = None
        self._dataset = self._load_dataset(self.dataset_dir)
        self._maybe_init_vae()

    def _maybe_init_vae(self):
        if self.vae_path is None:
            return
        if os.path.exists(self.vae_path):
            ckpt = torch.load(self.vae_path, map_location="cpu")
            state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            self.vae.load_state_dict(state_dict)
            self.freeze_vae = True

    def prep_rollout(self):
        self.policy.eval()
        self.vae.eval()

    def prep_training(self):
        self.policy.train()
        self.vae.train()

    def _vae_loss(self, recon_x, x, mu, logvar):
        if self.recon_loss == "mse":
            rec = torch.mean((recon_x - x) ** 2)
        else:
            rec = torch.mean(torch.abs(recon_x - x))
        kld = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        total = rec + kld
        return total, rec, kld

    def _iter_minibatches(self, batch_size, total_size):
        indices = np.random.permutation(total_size)
        for start in range(0, total_size, batch_size):
            yield indices[start:start + batch_size]

    def _load_dataset(self, dataset_dir):
        if dataset_dir is None:
            raise ValueError("pld_dataset_dir is required for offline PLD training.")
        paths = sorted(glob.glob(os.path.join(dataset_dir, "*.npz")))
        if not paths:
            raise FileNotFoundError(f"No .npz files found in {dataset_dir}")

        obs_list = []
        share_obs_list = []
        actions_list = []
        advantages_list = []
        active_masks_list = []
        available_actions_list = []

        for path in paths:
            data = np.load(path, allow_pickle=False)
            if "obs" not in data or "actions" not in data or "advantages" not in data:
                raise KeyError(f"Missing required keys in {path}. Need obs/actions/advantages.")
            obs = data["obs"]
            share_obs = data["share_obs"] if "share_obs" in data else obs
            actions = data["actions"]
            advantages = data["advantages"]
            active_masks = data["active_masks"] if "active_masks" in data else None
            available_actions = data["available_actions"] if "available_actions" in data else None

            # Allow [T, R, N, D] or [C, T, R, N, D] or [M, N, D]
            if obs.ndim == 5:
                # [chunk, T, rollout, agents, dim] -> [chunk*T*rollout, agents, dim]
                obs = obs.reshape(-1, obs.shape[3], obs.shape[4])
                share_obs = share_obs.reshape(-1, share_obs.shape[3], share_obs.shape[4])
                actions = actions.reshape(-1, actions.shape[3], actions.shape[4])
                advantages = advantages.reshape(-1, advantages.shape[3], advantages.shape[4])
                if active_masks is not None:
                    active_masks = active_masks.reshape(-1, active_masks.shape[3], active_masks.shape[4])
                if available_actions is not None:
                    available_actions = available_actions.reshape(-1, available_actions.shape[3], available_actions.shape[4])
            elif obs.ndim == 4:
                # [T, rollout, agents, dim] -> [T*rollout, agents, dim]
                obs = obs.reshape(-1, obs.shape[2], obs.shape[3])
                share_obs = share_obs.reshape(-1, share_obs.shape[2], share_obs.shape[3])
                actions = actions.reshape(-1, actions.shape[2], actions.shape[3])
                advantages = advantages.reshape(-1, advantages.shape[2], advantages.shape[3])
                if active_masks is not None:
                    active_masks = active_masks.reshape(-1, active_masks.shape[2], active_masks.shape[3])
                if available_actions is not None:
                    available_actions = available_actions.reshape(-1, available_actions.shape[2], available_actions.shape[3])
            elif obs.ndim != 3:
                raise ValueError(f"Unexpected obs shape in {path}: {obs.shape}")

            obs_list.append(obs)
            share_obs_list.append(share_obs)
            actions_list.append(actions)
            advantages_list.append(advantages)
            if active_masks is not None:
                active_masks_list.append(active_masks)
            if available_actions is not None:
                available_actions_list.append(available_actions)

        dataset = {
            "obs": np.concatenate(obs_list, axis=0),
            "share_obs": np.concatenate(share_obs_list, axis=0),
            "actions": np.concatenate(actions_list, axis=0),
            "advantages": np.concatenate(advantages_list, axis=0),
        }
        if active_masks_list:
            dataset["active_masks"] = np.concatenate(active_masks_list, axis=0)
        if available_actions_list:
            dataset["available_actions"] = np.concatenate(available_actions_list, axis=0)
        return dataset

    def train(self, buffer):
        self.prep_training()

        obs = self._dataset["obs"]
        share_obs = self._dataset["share_obs"]
        actions = self._dataset["actions"]
        advantages = self._dataset["advantages"]
        active_masks = self._dataset.get("active_masks", None)
        available_actions = self._dataset.get("available_actions", None)

        total_size = obs.shape[0]

        if self.vae_path is None or not os.path.exists(self.vae_path):
            vae_infos = self._train_vae_only(obs, share_obs, total_size)
            if self.vae_path is not None:
                torch.save({"state_dict": self.vae.state_dict()}, self.vae_path)
            self.freeze_vae = True

        total_loss_sum = 0.0
        rec_sum = 0.0
        kld_sum = 0.0
        latent_sum = 0.0
        policy_sum = 0.0
        count = 0

        for _ in range(self.epochs):
            for idx in self._iter_minibatches(self.batch_size, total_size):
                obs_b = check(obs[idx]).to(**self.tpdv)
                share_obs_b = check(share_obs[idx]).to(**self.tpdv)
                actions_b = check(actions[idx]).to(**self.tpdv)
                advantages_b = check(advantages[idx]).to(**self.tpdv).squeeze(-1)
                if active_masks is not None:
                    active_masks_b = check(active_masks[idx]).to(**self.tpdv).squeeze(-1)
                else:
                    active_masks_b = torch.ones_like(advantages_b)
                if available_actions is not None:
                    avail_b = check(available_actions[idx]).to(**self.tpdv)
                else:
                    avail_b = None

                self.vae_optimizer.zero_grad()
                self.policy.optimizer.zero_grad()

                with torch.no_grad():
                    recon_x, mu, logvar = self.vae(share_obs_b)
                vae_loss = torch.zeros(1, device=self.device)
                rec = torch.zeros(1, device=self.device)
                kld = torch.zeros(1, device=self.device)

                z_target = mu.detach().unsqueeze(1).repeat(1, self.num_agents, 1)
                if self.disable_obs_encoder:
                    z_pred = torch.zeros((obs_b.size(0), self.num_agents, self.latent_dim), device=self.device)
                else:
                    z_pred = self.policy.encoder(obs_b)
                latent_loss = latent_mimic_loss(z_pred, z_target, mask=active_masks_b)

                policy_out = self.policy.policy(obs_b, z_pred)
                if self.policy.action_type == "Discrete":
                    logits = policy_out["logits"]
                    if avail_b is not None:
                        logits = logits.masked_fill(avail_b == 0, float("-inf"))
                    policy_out = {"logits": logits}
                    actions_teacher = actions_b.squeeze(-1).long()
                else:
                    actions_teacher = actions_b

                policy_loss = advantage_weighted_imitation_loss(
                    policy_out,
                    actions_teacher,
                    advantages_b,
                    action_type="discrete" if self.policy.action_type == "Discrete" else "continuous",
                    temperature=self.temperature,
                    mask=active_masks_b,
                    use_exp_weight=self.use_exp_weight,
                )

                total_loss = (
                    self.latent_coef * latent_loss +
                    self.policy_coef * policy_loss
                )

                total_loss.backward()
                self.policy.optimizer.step()

                bs = obs_b.size(0)
                total_loss_sum += total_loss.item() * bs
                rec_sum += rec.item() * bs
                kld_sum += kld.item() * bs
                latent_sum += latent_loss.item() * bs
                policy_sum += policy_loss.item() * bs
                count += bs

        if count == 0:
            count = 1

        train_infos = {
            "pld_total_loss": total_loss_sum / count,
            "pld_latent_loss": latent_sum / count,
            "pld_policy_loss": policy_sum / count,
        }
        if self.vae_path is None or not os.path.exists(self.vae_path):
            train_infos.update(vae_infos)
        else:
            train_infos["pld_vae_loss"] = 0.0
            train_infos["pld_vae_recon_loss"] = 0.0
            train_infos["pld_vae_kld_loss"] = 0.0
        return train_infos

    def _train_vae_only(self, obs, share_obs, total_size):
        self.vae.train()
        total_loss_sum = 0.0
        rec_sum = 0.0
        kld_sum = 0.0
        count = 0

        for _ in range(self.epochs):
            for idx in self._iter_minibatches(self.batch_size, total_size):
                share_obs_b = check(share_obs[idx]).to(**self.tpdv)
                self.vae_optimizer.zero_grad()
                recon_x, mu, logvar = self.vae(share_obs_b)
                vae_loss, rec, kld = self._vae_loss(recon_x, share_obs_b, mu, logvar)
                vae_loss.backward()
                self.vae_optimizer.step()

                bs = share_obs_b.size(0)
                total_loss_sum += vae_loss.item() * bs
                rec_sum += rec.item() * bs
                kld_sum += kld.item() * bs
                count += bs

        if count == 0:
            count = 1

        return {
            "pld_vae_loss": total_loss_sum / count,
            "pld_vae_recon_loss": rec_sum / count,
            "pld_vae_kld_loss": kld_sum / count,
        }
