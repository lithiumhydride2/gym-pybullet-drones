import numpy as np


class IPPArguments:

    def __init__(self):
        if __debug__:
            self.N_ENVS = 1
            self.DEFAULT_GUI = True
            self.DEFAULT_USER_DEBUG_GUI = True
        else:
            self.N_ENVS = 16
            self.DEFAULT_GUI = False
            self.DEFAULT_USER_DEBUG_GUI = False
        self.CONTROL_BY_RL_MASK = None  # "random" 为随机生成,其余为固定
        self.RANDOM_POINT = False  # 是否随机生成目标点
        self.NUM_DRONE = 4
        #### graph
        self.k_size = 10  # knn
        self.sample_num = 36
        self.gen_range = np.deg2rad([0, 180])  # 限制采样的范围
        #### terminated
        self.MAX_EPISODE_LEN = 180  # max length of an episode / s
        self.TERMINATE_MIN_DIS = 1.0
        self.TERMINATE_MAX_DIS = 5.0

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
        self.DECISION_FREQ = 2
        self.FLOCKIN_FREQ = 5
        self.EXIST_THRESHOLD = np.exp(-0.5)  # 约 0.6
        self.LOSE_BELIEF_THERSHOLD = 0.1  # 信念值低于 0.1 时认为丢失目标


IPPArg = IPPArguments()
