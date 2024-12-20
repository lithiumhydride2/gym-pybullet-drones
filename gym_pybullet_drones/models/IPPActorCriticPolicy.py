from functools import partial
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from gym_pybullet_drones.envs.IPPArguments import IPPArg
from .attention_net import AttentionNet
from torch import nn
import numpy as np
import torch


class IPPFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space, features_dim=IPPArg.FEATURE_DIM):
        super().__init__(observation_space, features_dim)
        #TODO: 需要看看 attnetion_net 中实现了哪些内容， 对于我减少观测数量与类型的 obs 应当设计怎样的 attention_net
        self.attention_net = AttentionNet(IPPArg.EMBEDDING_DIM)  # Args todo

    def forward(self, observation):
        '''
        Args:
            observation:
        Return:
            return in shape (batch_size, features_dim )
        '''
        # stable_baselines3 自动转换observation并添加 batch dim
        return self.attention_net(
            node_inputs=observation["node_inputs"],
            edge_inputs=observation["edge_inputs"],
            current_index=observation["curr_index"],
            pos_encoding=observation["graph_pos_encoding"],
        )


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

        super().__init__(observation_space, action_space, lr_schedule,
                         net_arch, activation_fn, ortho_init, use_sde,
                         log_std_init, full_std, use_expln, squash_output,
                         features_extractor_class, features_extractor_kwargs,
                         share_features_extractor, normalize_images,
                         optimizer_class, optimizer_kwargs)

    def _build(self, lr_schedule):
        # self.mlp_extractor = nn.Identity()
        # self.action_net = nn.Identity()
        # self.value_net = nn.Identity()
        if self.ortho_init:
            module_gains = {self.features_extractor: np.sqrt(2)}
        for module, gain in module_gains.items():
            module.apply(partial(self.init_weights, gain=gain))

        self.optimizer = self.optimizer_class(self.parameters(),
                                              lr=lr_schedule(1),
                                              **self.optimizer_kwargs)

    def forward(self, obs, deterministic=False):
        '''
        这里 features extractor 其实已经包含了 action_net
        '''
        logp_list, value = self.features_extractor(obs)

        if deterministic:
            action_index = torch.argmax(logp_list, dim=1).long()
        else:
            action_index = torch.multinomial(logp_list.exp(),
                                             num_samples=1).long().squeeze(1)
        # log_p of the next action
        log_p = torch.gather(logp_list, dim=1, index=action_index.unsqueeze(0))
        edge_inputs = obs["edge_inputs"]
        curr_index = obs["curr_index"]
        next_node_index = edge_inputs[:,
                                      curr_index.item(),
                                      action_index.item()]
        return next_node_index, value, log_p
