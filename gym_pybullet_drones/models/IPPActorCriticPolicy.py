from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from gym_pybullet_drones.envs.IPPArguments import IPPArg
from .attention_net import AttentionNet
from torch import nn
import torch as th


class IPPFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space, features_dim=IPPArg.INPUT_DIM):
        super().__init__(observation_space, features_dim)
        #TODO: 需要看看 attnetion_net 中实现了哪些内容， 对于我减少观测数量与类型的 obs 应当设计怎样的 attention_net
        self.attention_net = AttentionNet(IPPArg.INPUT_DIM,
                                          IPPArg.EMBEDDING_DIM)  # Args todo

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
                 features_extractor_class=IPPFeaturesExtractor,
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
