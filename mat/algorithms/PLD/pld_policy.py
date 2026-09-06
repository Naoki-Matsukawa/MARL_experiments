import numpy as np
import torch
from torch.distributions import Categorical, Normal

from mat.algorithms.utils.util import check
from mat.algorithms.PLD.decentralized_policy import AgentObsEncoder, AgentPolicy
from mat.utils.util import get_shape_from_obs_space, get_shape_from_act_space


class PLDPolicy:
    """
    Policy wrapper for PLD that fits the runner interface.
    """
    def __init__(self, args, obs_space, cent_obs_space, act_space, num_agents, device=torch.device("cpu")):
        self.device = device
        self.num_agents = num_agents
        self.action_type = "Continuous" if act_space.__class__.__name__ == "Box" else "Discrete"
        self.act_space = act_space

        self.obs_dim = get_shape_from_obs_space(obs_space)[0]
        self.share_obs_dim = get_shape_from_obs_space(cent_obs_space)[0]
        if self.action_type == "Discrete":
            self.act_dim = act_space.n
            self.act_num = 1
        else:
            self.act_dim = act_space.shape[0]
            self.act_num = self.act_dim

        self.latent_dim = args.pld_latent_dim
        self.encoder = AgentObsEncoder(
            obs_dim=self.obs_dim,
            latent_dim=self.latent_dim,
            hidden_dim=args.pld_hidden_dim,
            num_layers=args.pld_num_layers,
            use_layernorm=args.pld_use_layernorm,
        ).to(self.device)
        self.disable_obs_encoder = args.pld_disable_obs_encoder
        self.policy = AgentPolicy(
            obs_dim=self.obs_dim,
            latent_dim=self.latent_dim,
            act_dim=self.act_dim,
            action_type="discrete" if self.action_type == "Discrete" else "continuous",
            hidden_dim=args.pld_hidden_dim,
            num_layers=args.pld_num_layers,
            use_layernorm=args.pld_use_layernorm,
            log_std_bounds=(args.pld_log_std_min, args.pld_log_std_max),
        ).to(self.device)

        params = list(self.encoder.parameters()) + list(self.policy.parameters())
        self.optimizer = torch.optim.Adam(params, lr=args.pld_lr)
        self.tpdv = dict(dtype=torch.float32, device=self.device)

    def lr_decay(self, episode, episodes):
        # PLD uses a fixed LR unless the caller updates externally.
        return

    def _mask_logits(self, logits, available_actions):
        if available_actions is None:
            return logits
        avail = check(available_actions).to(**self.tpdv)
        return logits.masked_fill(avail == 0, float("-inf"))

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks,
                    available_actions=None, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        obs = obs.reshape(-1, self.num_agents, self.obs_dim)
        if available_actions is not None:
            available_actions = available_actions.reshape(-1, self.num_agents, self.act_dim)

        if self.disable_obs_encoder:
            z_pred = torch.zeros((obs.shape[0], self.num_agents, self.latent_dim), device=self.device)
        else:
            z_pred = self.encoder(obs)
        out = self.policy(obs, z_pred)

        if self.action_type == "Discrete":
            logits = self._mask_logits(out["logits"], available_actions)
            dist = Categorical(logits=logits)
            if deterministic:
                actions = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                actions = dist.sample().unsqueeze(-1)
            action_log_probs = dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
        else:
            mu = out["mu"]
            log_std = out["log_std"]
            std = torch.exp(log_std)
            dist = Normal(mu, std)
            if deterministic:
                actions = mu
            else:
                actions = dist.rsample()
            action_log_probs = dist.log_prob(actions)

        values = torch.zeros((obs.shape[0] * self.num_agents, 1), device=self.device, dtype=torch.float32)

        # unused, just for compatibility
        rnn_states_actor = check(rnn_states_actor).to(**self.tpdv)
        rnn_states_critic = check(rnn_states_critic).to(**self.tpdv)
        return values, actions.view(-1, self.act_num), action_log_probs.view(-1, self.act_num), rnn_states_actor, rnn_states_critic

    def act(self, cent_obs, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        # wrapper to match runner eval interface
        rnn_states_critic = np.zeros_like(rnn_states_actor)
        _, actions, _, rnn_states_actor, _ = self.get_actions(
            cent_obs,
            obs,
            rnn_states_actor,
            rnn_states_critic,
            masks,
            available_actions,
            deterministic,
        )
        return actions, rnn_states_actor

    def get_values(self, cent_obs, obs, rnn_states_critic, masks, available_actions=None):
        batch = obs.shape[0]
        values = torch.zeros((batch, 1), device=self.device, dtype=torch.float32)
        return values

    def train(self):
        self.encoder.train()
        self.policy.train()

    def eval(self):
        self.encoder.eval()
        self.policy.eval()

    def save(self, save_dir, episode):
        payload = {
            "encoder": self.encoder.state_dict(),
            "policy": self.policy.state_dict(),
        }
        torch.save(payload, str(save_dir) + "/pld_policy_" + str(episode) + ".pt")

    def restore(self, model_dir):
        payload = torch.load(model_dir, map_location="cpu")
        if isinstance(payload, dict) and "encoder" in payload and "policy" in payload:
            self.encoder.load_state_dict(payload["encoder"])
            self.policy.load_state_dict(payload["policy"])
        else:
            raise ValueError("Invalid PLD checkpoint format.")
