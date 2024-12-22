import numpy as np


class IPPArguments:

    def __init__(self):
        #### graph
        self.k_size = 5  # knn
        self.sample_num = 40
        self.gen_range = np.deg2rad([0, 180])  # 限制采样的范围

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
