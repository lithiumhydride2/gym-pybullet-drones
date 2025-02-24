from functools import partial
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from gym_pybullet_drones.envs.IPPArguments import IPPArg
from .attention_net import AttentionNet, SingleHeadAttention
from torch import nn
import numpy as np
from torch.cuda.amp.autocast_mode import autocast
import torch


class IPPFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space, features_dim=3):
        super().__init__(observation_space, features_dim)
        self.attention_net = AttentionNet(IPPArg.EMBEDDING_DIM)  # Args todo

    def forward(self, observation):
        '''
        Args:
            observation:
        Return:
            return in shape (batch_size, features_dim )
        '''
        ## 这里从 pedded 的 node_inputs 中取出实际有用的部分
        if IPPArg.NUM_DRONE < IPPArg.MAX_NUM_DRONE:
            node_input_feat_dim = 2 + (IPPArg.NUM_DRONE -
                                       1) * IPPArg.BELIEF_FEATURE_DIM
            node_inputs = observation["node_inputs"][
                ..., :node_input_feat_dim]  # 仅从最后一个维度提取
        else:
            raise ValueError
        return self.attention_net(node_inputs=node_inputs,
                                  dt_pool_inputs=observation["dt_pool_inputs"],
                                  current_index=observation["curr_index"],
                                  dist_inputs=observation["dist_inputs"],
                                  edge_inputs=observation["edge_inputs"])


class IPPMlpExtractor(nn.Module):

    def __init__(self, feature_dim: int, last_layer_dim_pi: int,
                 last_layer_dim_vf: int):
        super().__init__()

        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        # policy network
        # self.policy_net = nn.Linear(feature_dim, last_layer_dim_pi)
        self.value_net = nn.Linear(feature_dim, last_layer_dim_vf).float()
        self.policy_net = SingleHeadAttention(IPPArg.EMBEDDING_DIM).float()

    def forward_actor(self, features):
        logp = self.policy_net(features[:, :1, :], features[:, 1:, :])
        logp = logp.squeeze(dim=1)
        return logp

    def forward_critic(self, features):
        val = self.value_net(features[:, :1, :])
        val = val.squeeze(dim=1)
        return val

    def forward(self, features):
        '''
        Args:
            features: shape (batch_size, curr_edge + k_size, feature_dim
        '''
        features = features.float()
        return self.forward_actor(features), self.forward_critic(features)


class IPPActorCriticPolicy(ActorCriticPolicy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args,
                         **kwargs,
                         features_extractor_class=IPPFeaturesExtractor)
        self.ortho_init = False

    def _build_mlp_extractor(self):
        self.mlp_extractor = IPPMlpExtractor(IPPArg.EMBEDDING_DIM,
                                             last_layer_dim_pi=IPPArg.k_size,
                                             last_layer_dim_vf=1)
