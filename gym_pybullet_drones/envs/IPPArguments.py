import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from gym_pybullet_drones.utils.enums import FOVType
import os


class IPPArguments:

    def __init__(self):
        if __debug__:
            self.N_ENVS = 1
            if os.getenv("DISPLAY") is None:
                self.DEFAULT_GUI = False
                self.DEFAULT_USER_DEBUG_GUI = False
            else:
                self.DEFAULT_GUI = True
                self.DEFAULT_USER_DEBUG_GUI = True
            self.VEC_ENV_CLS = DummyVecEnv
        else:
            self.N_ENVS = 16
            self.DEFAULT_GUI = False
            self.DEFAULT_USER_DEBUG_GUI = False
            self.VEC_ENV_CLS = DummyVecEnv

        self.CONTROL_BY_RL_MASK = "all"  # "random" 为随机生成,其余为固定, "all" 为所有无人机受 RL 控制
        self.RANDOM_POINT = False  # 是否随机生成目标点，当前参数为 random_50
        self.WAYPOINT_FILE_NAME = "circle_7"  # or "circle_7"
        self.NUM_DRONE = 6
        self.FOV_CONFIG = FOVType.DOUBLE
        self.MAX_NUM_DRONE = 20  # 在 observation space 中使用的最大无人机数量
        #### graph
        self.k_size = 3
        self.sample_num = 24  # 应当为一个偶数
        self.gen_range = np.deg2rad([0, 180])  # 限制采样的范围
        #### terminated
        self.MAX_EPISODE_LEN = 600  # max length of an episode / s
        self.TERMINATE_MIN_DIS = 1.0
        self.TERMINATE_MAX_DIS = 5.0

        # 选取前 32 大的特征值，这里抛去第一大特征值，因此最大为 sample_num - 1
        self.num_eigen_value = min(32, self.sample_num - 1)
        self.history_size = 10  # avgpool 10 个历史时刻， 在 decision_freq 为 2hz 情况下，约使用 5s 的历史数据
        self.history_stride = 1  # set 1 to disable pooling
        ### parameter of attention_net
        self.EMBEDDING_DIM = 128
        self.N_HEAD = 4  # head num of decoder
        self.N_LAYER = 1
        self.NODE_COORD_DIM = 2
        self.BELIEF_FEATURE_DIM = 3  # yaw_heading, belief
        self.PREDICT_FEATURE_TIME = 2.0  # [s] 使用 GP 预测 2s 后的特征
        self.dt_normlization = 1.993 * 3  # UNC WITH 1% uncertainty , 时间维度的归一化参数
        self.DECISION_FREQ = 5
        self.FLOCKIN_FREQ = 5
        self.EXIST_THRESHOLD = np.exp(-0.5)  # 约 0.6
        self.LOSE_BELIEF_THERSHOLD = 0.1  # 信念值低于 0.1 时认为丢失目标


IPPArg = IPPArguments()
