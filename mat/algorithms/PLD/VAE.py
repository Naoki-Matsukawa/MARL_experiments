import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from mat.utils.util import get_gard_norm, huber_loss, mse_loss
from mat.utils.valuenorm import ValueNorm

class MultiAgentVAE(nn.Module):
    def __init__(self, num_agents, obs_dim, latent_dim=32, agent_feature_dim=64):
        super(MultiAgentVAE, self).__init__()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.agent_feature_dim = agent_feature_dim

        # --- 1. Agent-wise Encoder (共通の重み) ---
        # 入力: [Batch, Agents, Obs]
        # この層は「各エージェントごと」に独立して適用されます（隣を見ない）
        self.agent_encoder = nn.Sequential(
            nn.Linear(obs_dim, agent_feature_dim),
            nn.ReLU(),
            nn.Linear(agent_feature_dim, agent_feature_dim),
            nn.ReLU(),
        )
        
        # --- 2. Global Encoder (全体をまとめる) ---
        # ここで初めて全エージェントの情報を集約します
        # [Batch, Agents, AgentFeature] -> [Batch, Agents * AgentFeature]
        self.flatten_dim = num_agents * agent_feature_dim
        
        self.mu_layer = nn.Linear(self.flatten_dim, latent_dim)
        self.logvar_layer = nn.Linear(self.flatten_dim, latent_dim)

        # --- 3. Decoder ---
        # 潜在変数 z から、まず「全エージェントの特徴量」に復元
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)
        
        # 特徴量から観測値へ戻す（ここもAgent-wiseで共通の重み）
        self.agent_decoder = nn.Sequential(
            nn.Linear(agent_feature_dim, agent_feature_dim),
            nn.ReLU(),
            nn.Linear(agent_feature_dim, obs_dim),
            # データが正規化されているかどうかで活性化関数を変える
            # nn.Tanh() 
        )

    def encode(self, x):
        # x: [Batch, Agents, ObsDim]
        
        # Step 1: 各エージェントの特徴を個別に抽出
        # 出力: [Batch, Agents, AgentFeature]
        h_agents = self.agent_encoder(x) 
        
        # Step 2: フラット化して全体の特徴へ
        # 出力: [Batch, Agents * AgentFeature]
        h_flat = h_agents.view(x.size(0), -1) 
        
        mu = self.mu_layer(h_flat)
        logvar = self.logvar_layer(h_flat)
        return mu, logvar

    def decode(self, z):
        # Step 1: 潜在変数から全エージェントの特徴量一式を展開
        h_flat = self.decoder_input(z)
        
        # Step 2: 行列の形に戻す
        # [Batch, Agents * AgentFeature] -> [Batch, Agents, AgentFeature]
        h_agents = h_flat.view(-1, self.num_agents, self.agent_feature_dim)
        
        # Step 3: 各エージェントごとに観測値を復元
        # 出力: [Batch, Agents, ObsDim]
        recon_x = self.agent_decoder(h_agents)
        return recon_x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) # reparameterizeは前回と同じ
        recon_x = self.decode(z)
        return recon_x, mu, logvar
