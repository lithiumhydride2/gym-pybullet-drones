import numpy as np


class IPPArguments:

    def __init__(self):

        self.N_ENVS = 1
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
        self.history_size = 50  # avgpool 历史 50 个历史时刻
        ### parameter of attention_net
        self.FEATURE_DIM = 10086  # 用于初始化 custom policy, 似乎没有作用
        self.EMBEDDING_DIM = 128
        self.N_HEAD = 4  # head num of decoder
        self.N_LAYER = 1
        self.NODE_COORD_DIM = 2
        self.BELIEF_FEATURE_DIM = 4  # mean, std, predict_mean, predict_std
        self.PREDICT_FEATURE_TIME = 2.0  # [s] 使用 GP 预测 2s 后的特征
        self.history_stride = 5  # set 1 to disable pooling
        self.dt_normlization = 1.993 * 3  # UNC WITH 1% uncertainty , 时间维度的归一化参数


IPPArg = IPPArguments()
