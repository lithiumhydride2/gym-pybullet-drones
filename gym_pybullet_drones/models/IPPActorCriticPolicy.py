from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from .attention_net import AttentionNet
from torch import nn
import torch as th


class IPPFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space, features_dim=0):
        super().__init__(observation_space, features_dim)
        self.attention_net = AttentionNet()  # Args todo

    def forward(self, observation):
        return self.attention_net(observation)


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
                 features_extractor_class=...,
                 features_extractor_kwargs=None,
                 share_features_extractor=True,
                 normalize_images=True,
                 optimizer_class=th.optim.Adam,
                 optimizer_kwargs=None):
        super().__init__(observation_space, action_space, lr_schedule,
                         net_arch, activation_fn, ortho_init, use_sde,
                         log_std_init, full_std, use_expln, squash_output,
                         features_extractor_class, features_extractor_kwargs,
                         share_features_extractor, normalize_images,
                         optimizer_class, optimizer_kwargs)
