import numpy as np


class IPPArguments:

    def __init__(self):

        self.N_ENVS = 16
        #### graph
        self.k_size = 10  # knn
        self.sample_num = 36
        self.gen_range = np.deg2rad([0, 180])  # 限制采样的范围
        #### terminated
        self.MAX_EPISODE_LEN = 180  # max length of an episode / s
        self.TERMINATE_MIN_DIS = 1.0
        self.TERMINATE_MAX_DIS = 5.5

        # 选取前 32 大的特征值，这里抛去第一大特征值，因此最大为 sample_num - 1
        self.num_eigen_value = min(32, self.sample_num - 1)
        ### parameter of attention_net
        self.FEATURE_DIM = 10086  # 用于初始化 custom policy, 似乎没有作用
        self.EMBEDDING_DIM = 128
        self.N_HEAD = 4  # head num of decoder
        self.N_LAYER = 1
        self.ADAPTIVE_TH = 0.4  # what's this
        self.NODE_COORD_DIM = 2
        self.BELIEF_FEATURE_DIM = 2  # mean and std


IPPArg = IPPArguments()
