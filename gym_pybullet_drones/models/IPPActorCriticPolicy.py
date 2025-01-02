from functools import partial
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from gym_pybullet_drones.envs.IPPArguments import IPPArg
from .attention_net import AttentionNet, SampleNet
from torch import nn
import numpy as np
import torch


class IPPFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space, features_dim=IPPArg.sample_num):
        super().__init__(observation_space, features_dim)
        #TODO: 需要看看 attnetion_net 中实现了哪些内容， 对于我减少观测数量与类型的 obs 应当设计怎样的 attention_net
        # self.attention_net = AttentionNet(IPPArg.EMBEDDING_DIM)  # Args todo
        self.sample_net = SampleNet(IPPArg.EMBEDDING_DIM)

    def forward(self, observation):
        '''
        Args:
            observation:
        Return:
            return in shape (batch_size, features_dim )
        '''
        curr_index = observation["curr_index"]
        curr_index = curr_index.unsqueeze(-1).repeat(1, 5, 1, 12).long()
        input = torch.gather(observation["node_inputs"],
                             dim=2,
                             index=curr_index)  #(bach,feature)
        input = input[:, -1, :, :].reshape(-1, 12)
        return self.sample_net(input)
        # stable_baselines3 自动转换observation并添加 batch dim

        return self.attention_net(node_inputs=observation["node_inputs"],
                                  dt_pool_inputs=observation["dt_pool_inputs"],
                                  current_index=observation["curr_index"],
                                  dist_inputs=observation["dist_inputs"])


class IPPActorCriticPolicy(ActorCriticPolicy):

    def __init__(self,
                 observation_space,
                 action_space,
                 lr_schedule,
                 net_arch=None,
                 activation_fn=nn.Tanh,
                 ortho_init=True,
                 use_sde=False,
                 log_std_init=0,
                 full_std=True,
                 use_expln=False,
                 squash_output=False,
                 features_extractor_class=IPPFeaturesExtractor,
                 features_extractor_kwargs=None,
                 share_features_extractor=True,
                 normalize_images=True,
                 optimizer_class=torch.optim.Adam,
                 optimizer_kwargs=None):
        # note: net_arch 是 None, 默认的 mlp_extractor 是 nn.Identity()
        super().__init__(observation_space, action_space, lr_schedule,
                         net_arch, activation_fn, ortho_init, use_sde,
                         log_std_init, full_std, use_expln, squash_output,
                         features_extractor_class, features_extractor_kwargs,
                         share_features_extractor, normalize_images,
                         optimizer_class, optimizer_kwargs)

    def _build(self, lr_schedule):
        # 由于使用 自定义 features_extractor, 强制将 value_net, action_net 的 net_arch 设为空 list
        self.net_arch = []
        super()._build(lr_schedule)
