import os
import warnings

import torch
import torch.nn as nn

from mat.algorithms.mat.algorithm.ma_transformer import MultiAgentTransformer
from mat.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from mat.algorithms.utils.util import check, init
from mat.utils.util import get_shape_from_act_space, get_shape_from_obs_space, update_linear_schedule


class CDBDActor(R_Actor):
    """Recurrent student actor with a projection head on its hidden features."""

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super().__init__(args, obs_space, action_space, device)
        latent_dim = getattr(args, "cdbd_latent_dim", None)
        latent_dim = getattr(args, "n_embd", self.hidden_size) if latent_dim is None else int(latent_dim)
        use_orthogonal = getattr(args, "use_orthogonal", True)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.projection = init_(nn.Linear(self.hidden_size, latent_dim)).to(device)

    def forward_features(self, obs, rnn_states, masks):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        actor_features = self.base(obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        return actor_features, rnn_states

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features, rnn_states = self.forward_features(obs, rnn_states, masks)
        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
        student_latent = self.projection(actor_features)
        return actions, action_log_probs, rnn_states, student_latent

    def evaluate_actions_with_latent(self, obs, rnn_states, action, masks,
                                     available_actions=None, active_masks=None):
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
        action = check(action).to(**self.tpdv)

        actor_features, _ = self.forward_features(obs, rnn_states, masks)
        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features,
            action,
            available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None,
        )
        student_latent = self.projection(actor_features)
        return action_log_probs, dist_entropy, student_latent


class CDBDMATTeacher(nn.Module):
    """Frozen MAT teacher that returns value predictions and encoder belief features."""

    def __init__(self, args, obs_dim, share_obs_dim, act_dim, num_agents, action_type, device):
        super().__init__()
        self.device = device
        self.obs_dim = obs_dim
        self.share_obs_dim = share_obs_dim
        self.act_dim = act_dim
        self.num_agents = num_agents
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.transformer = MultiAgentTransformer(
            share_obs_dim,
            obs_dim,
            act_dim,
            num_agents,
            n_block=args.n_block,
            n_embd=args.n_embd,
            n_head=args.n_head,
            encode_state=args.encode_state,
            device=device,
            action_type=action_type,
            dec_actor=args.dec_actor,
            share_actor=args.share_actor,
            use_agent_id=getattr(args, "use_agent_id", False),
        )

    def load_checkpoint(self, path):
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            for key in ("transformer", "teacher", "model", "state_dict"):
                if key in payload:
                    payload = payload[key]
                    break
        self.transformer.load_state_dict(payload, strict=False)

    def forward(self, cent_obs, obs, available_actions=None):
        del cent_obs, available_actions
        obs = check(obs).to(**self.tpdv).reshape(-1, self.num_agents, self.obs_dim)
        state = torch.zeros(
            obs.shape[0],
            self.num_agents,
            37,
            dtype=obs.dtype,
            device=obs.device,
        )
        values, belief = self.transformer.encoder(state, obs)
        return values.reshape(-1, 1), belief.reshape(-1, belief.shape[-1])


class CDBDPolicy:
    """
    Online critic-latent belief distillation policy.

    The public methods intentionally match the existing runner policy interface.
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, num_agents=None,
                 device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space
        self.num_agents = num_agents
        self.action_type = "Continuous" if act_space.__class__.__name__ == "Box" else "Discrete"
        self.obs_dim = get_shape_from_obs_space(obs_space)[0]
        self.share_obs_dim = get_shape_from_obs_space(cent_obs_space)[0]
        self.act_dim = act_space.n if self.action_type == "Discrete" else get_shape_from_act_space(act_space)

        self.actor = CDBDActor(args, self.obs_space, self.act_space, self.device)
        self.teacher = CDBDMATTeacher(
            args,
            self.obs_dim,
            self.share_obs_dim,
            self.act_dim,
            num_agents,
            self.action_type,
            self.device,
        )
        self._load_teacher(args)
        self.freeze_teacher()

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def freeze_teacher(self):
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def _load_teacher(self, args):
        model_dir = getattr(args, "cdbd_teacher_model_dir", None)
        if model_dir is None:
            warnings.warn("CDBD is using a randomly initialized frozen MAT teacher.")
            return

        candidates = []
        if os.path.isdir(model_dir):
            candidates.extend([
                os.path.join(model_dir, "transformer.pt"),
                os.path.join(model_dir, "teacher.pt"),
                os.path.join(model_dir, "cdbd_policy.pt"),
            ])
            candidates.extend(sorted(
                [os.path.join(model_dir, name) for name in os.listdir(model_dir) if name.startswith("transformer_")],
                key=os.path.getmtime,
                reverse=True,
            ))
        else:
            candidates.append(model_dir)

        for path in candidates:
            if not os.path.exists(path):
                continue
            self.teacher.load_checkpoint(path)
            return

        raise FileNotFoundError("No MAT teacher checkpoint found under {}".format(model_dir))

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks,
                    available_actions=None, deterministic=False):
        rnn_states_critic = check(rnn_states_critic).to(**self.actor.tpdv)
        actions, action_log_probs, rnn_states_actor, _ = self.actor(
            obs,
            rnn_states_actor,
            masks,
            available_actions,
            deterministic,
        )
        with torch.no_grad():
            values, _ = self.teacher(cent_obs, obs, available_actions)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, obs=None, rnn_states_critic=None, masks=None, available_actions=None):
        with torch.no_grad():
            values, _ = self.teacher(cent_obs, obs, available_actions)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        action_log_probs, dist_entropy, student_latent = self.actor.evaluate_actions_with_latent(
            obs,
            rnn_states_actor,
            action,
            masks,
            available_actions,
            active_masks,
        )
        with torch.no_grad():
            values, teacher_latent = self.teacher(cent_obs, obs, available_actions)
        return values, action_log_probs, dist_entropy, student_latent, teacher_latent

    def act(self, cent_obs, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        actions, _, rnn_states_actor, _ = self.actor(
            obs,
            rnn_states_actor,
            masks,
            available_actions,
            deterministic,
        )
        return actions, rnn_states_actor

    def save(self, save_dir, episode):
        payload = {
            "actor": self.actor.state_dict(),
            "teacher": self.teacher.state_dict(),
        }
        torch.save(payload, os.path.join(str(save_dir), "cdbd_policy_{}.pt".format(episode)))
        torch.save(self.actor.state_dict(), os.path.join(str(save_dir), "actor_{}.pt".format(episode)))

    def restore(self, model_dir):
        candidates = []
        if os.path.isdir(model_dir):
            candidates.extend([
                os.path.join(model_dir, "cdbd_policy.pt"),
                os.path.join(model_dir, "actor.pt"),
            ])
            candidates.extend(sorted(
                [os.path.join(model_dir, name) for name in os.listdir(model_dir) if name.startswith("cdbd_policy_")],
                key=os.path.getmtime,
                reverse=True,
            ))
            candidates.extend(sorted(
                [os.path.join(model_dir, name) for name in os.listdir(model_dir) if name.startswith("actor_")],
                key=os.path.getmtime,
                reverse=True,
            ))
        else:
            candidates.append(model_dir)

        path = next((candidate for candidate in candidates if os.path.exists(candidate)), None)
        if path is None:
            raise FileNotFoundError("No CDBD checkpoint found under {}".format(model_dir))

        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict) and "actor" in payload:
            self.actor.load_state_dict(payload["actor"], strict=False)
            if "teacher" in payload:
                self.teacher.load_state_dict(payload["teacher"], strict=False)
                self.freeze_teacher()
            return
        self.actor.load_state_dict(payload, strict=False)
