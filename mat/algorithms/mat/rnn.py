import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


class BaseRecurrentPolicy(nn.Module):
    def __init__(self, obs_dim, hidden_dim=128, use_mlp=False):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
        )
        self.hidden_dim = hidden_dim
        self.use_mlp = use_mlp
        if use_mlp:
            # Per-timestep MLP so that no temporal information leaks through hidden state.
            self.actor_mlp = nn.Linear(128, hidden_dim)
            self.critic_mlp = nn.Linear(128, hidden_dim)
        else:
            self.actor_rnn = nn.GRU(input_size=128, hidden_size=hidden_dim, batch_first=True)
            self.critic_rnn = nn.GRU(input_size=128, hidden_size=hidden_dim, batch_first=True)

    def _forward_core(self, obs_seq, hidden=None):
        x = self.obs_encoder(obs_seq)
        if self.use_mlp:
            b, t, _ = x.shape
            out = self.actor_mlp(x.view(-1, x.size(-1))).view(b, t, -1)
            dummy_hidden = torch.zeros(1, b, self.hidden_dim, device=x.device, dtype=x.dtype)
            return out, dummy_hidden
        else:
            out, new_hidden = self.actor_rnn(x, hidden)
            return out, new_hidden

    def _forward_value_core(self, obs_seq, hidden=None):
        x = self.obs_encoder(obs_seq)
        if self.use_mlp:
            b, t, _ = x.shape
            out = self.critic_mlp(x.view(-1, x.size(-1))).view(b, t, -1)
            dummy_hidden = torch.zeros(1, b, self.hidden_dim, device=x.device, dtype=x.dtype)
            return out, dummy_hidden
        else:
            out, new_hidden = self.critic_rnn(x, hidden)
            return out, new_hidden
    
    def update(self, loss):
        raise NotImplementedError
    
    def act(self, obs, hidden=None, deterministic=False):
        raise NotImplementedError


class DiscreteRecurrentPolicy(BaseRecurrentPolicy):
    def __init__(self, obs_dim, action_dim, hidden_dim=128, lr=1e-3, value_dim=None, use_mlp=False):
        super().__init__(obs_dim, hidden_dim, use_mlp)
        self.action_dim = action_dim
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, value_dim or hidden_dim)
        self.aux_value = nn.Linear(hidden_dim, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, obs_seq, hidden=None):
        features, new_hidden = self._forward_core(obs_seq, hidden)
        logits = self.policy_head(features)
        probs = F.softmax(logits, dim=-1)
        aux_value = self.aux_value(features)
        return probs, aux_value, new_hidden

    def forward_value(self, obs_seq, hidden=None):
        value_features, new_hidden = self._forward_value_core(obs_seq, hidden)
        values = self.value_head(value_features)
        return values, new_hidden

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    @torch.no_grad()
    def act(self, obs, hidden=None, deterministic=False):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0).unsqueeze(1)
        elif obs.dim() == 2:
            obs = obs.unsqueeze(1)

        probs, aux_value, new_hidden = self.forward(obs, hidden)
        last_probs = probs[:, -1, :]

        if deterministic:
            action = last_probs.argmax(dim=-1)
        else:
            dist = Categorical(probs=last_probs)
            action = dist.sample()
            last_probs = dist.probs

        log_prob = torch.log(torch.gather(last_probs, -1, action.unsqueeze(-1))).squeeze(-1)
        return action, log_prob, new_hidden

    def evaluate_actions(self, obs, actions, hidden=None):
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        if actions.dim() > 1:
            actions = actions.squeeze(-1)

        probs, aux_value, new_hidden = self.forward(obs, hidden)
        probs = probs.squeeze(1)
        actions = actions.long()

        gathered_probs = torch.gather(probs, -1, actions.unsqueeze(-1)).squeeze(-1)
        log_probs = torch.log(gathered_probs + 1e-8)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(-1)

        return log_probs, entropy, new_hidden


class ContinuousRecurrentPolicy(BaseRecurrentPolicy):
    def __init__(self, obs_dim, action_dim, hidden_dim=128, lr=1e-3, log_std_init=0.0, value_dim=None, use_mlp=False):
        super().__init__(obs_dim, hidden_dim, use_mlp)
        self.action_dim = action_dim
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
        self.value_head = nn.Linear(hidden_dim, value_dim or hidden_dim)
        self.aux_value = nn.Linear(hidden_dim, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, obs_seq, hidden=None):
        features, new_hidden = self._forward_core(obs_seq, hidden)
        means = self.mean_head(features)
        log_std = self.log_std.view(1, 1, -1).expand_as(means)
        aux_value = self.aux_value(features)
        return means, log_std, aux_value, new_hidden

    def forward_value(self, obs_seq, hidden=None):
        value_features, new_hidden = self._forward_value_core(obs_seq, hidden)
        values = self.value_head(value_features)
        return values, new_hidden

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    @torch.no_grad()
    def act(self, obs, hidden=None, deterministic=False):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0).unsqueeze(1)
        elif obs.dim() == 2:
            obs = obs.unsqueeze(1)

        means, log_std, aux_value, new_hidden = self.forward(obs, hidden)
        last_mean = means[:, -1, :]
        last_log_std = log_std[:, -1, :]
        dist = Normal(last_mean, torch.exp(last_log_std))

        if deterministic:
            action = last_mean
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, new_hidden

    def evaluate_actions(self, obs, actions, hidden=None):
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        means, log_std, aux_value, new_hidden = self.forward(obs, hidden)
        means = means.squeeze(1)
        log_std = log_std.squeeze(1)

        dist = Normal(means, torch.exp(log_std))
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)

        return log_probs, entropy, new_hidden
