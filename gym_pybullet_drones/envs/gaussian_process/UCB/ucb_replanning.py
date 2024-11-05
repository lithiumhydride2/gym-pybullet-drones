import numpy as np
from .motion_primitive import MotionPrimitive
from ..gaussian_process import GaussianProcessWrapper
from .uav_detection_sim import UavDetectionSim
from copy import deepcopy


class UCBReplanning:

    def __init__(self) -> None:
        self.mp = MotionPrimitive()  # motion primitive as mp
        self.init_GP_Wrapper = None
        self.GP_forward_sim = GPForwardSim()

    def step(self, GPWrapper: GaussianProcessWrapper, **kwargs):
        """
        # description :
        UCB-Replanning
         --------------- 
        # param :
         - item: 
         --------------- 
        # returns :
        运动基元的索引、该运动基元
         --------------- 
        """
        self.init_GP_Wrapper = GPWrapper

        # NO-regret replanning
        rewards = [0.0] * self.mp.num_primitive
        for k in range(self.mp.num_primitive):
            # reset: GP 仿真
            # 为当前 基元 初始化 GP
            self.GP_forward_sim.reset(deepcopy(self.init_GP_Wrapper))
            # 计算轨迹的均值序列和标准差序列
            mean_s = [0.0] * self.mp.num_segment
            std_s = [0.0] * self.mp.num_segment
            for j in range(self.mp.num_segment):

                forward_t = self.mp.time_range[j]
                action = self.mp.motion_primitive[k][j]
                #forward sim
                mean, std = self.GP_forward_sim.step(action=action,
                                                     forward_t=forward_t)
                mean_s[j] = mean.squeeze()
                std_s[j] = std.squeeze()
            # to np.ndarray
            mean_s = np.array(mean_s)
            std_s = np.array(std_s)
            rewards[k] = self.get_rewards(mean_s, std_s)

        rewards = np.array(rewards)
        return np.argmax(rewards), self.mp.motion_primitive[np.argmax(rewards)]

    def get_rewards(self, mean_s: np.ndarray, std_s: np.ndarray):
        """
        # description :
         --------------- 
        # param :
         - item: 
         --------------- 
        # returns : 
            根据 均值 与 标准差 返回 reward
         --------------- 
        """
        # TODO: 这里用整个 MAP 计算 reward 的代价太大了，应当使用在 heat_map 上抽象出来的图
        return .0


def add_t(X, t: float):
    return np.concatenate((X, np.zeros((X.shape[0], 1)) + t), axis=1)


def change_t(X: 'list[np.ndarray]', new_t: float):
    '将时间替换为 new_t'
    for date in X:
        date[:, 2] = new_t
    return X


class GPForwardSim:

    def __init__(self, init_state: GaussianProcessWrapper = None) -> None:
        self.state = init_state
        self.init_state_time = 0.0
        self.uav_detection_sim = UavDetectionSim()

    def reset(self, init_state: GaussianProcessWrapper = None):
        self.state = init_state
        self.init_state_time = self.state.curr_t  # 在 GPwrapper.update_grids()中更新
        # maybe deepcopy self.state.GPs
        self.uav_detection_sim.reset(self.state.GPs)

    def step(self, action: np.ndarray = None, forward_t=.0):
        """
        ## description :
         --------------- 
        ## param :
         - action: 一条运动基元
         - forward_t: 该基元已经前向的时间,非time_gap
         ---------------
        ## return:
            返回更新后的 gaussian std and mean
        """

        observe_points, observe_vals = self.uav_detection_sim.step(
            action=action)
        self.state.add_observed_points(
            change_t(observe_points, forward_t + self.init_state_time),
            observe_vals)
        self.state.update_GPs()
        sim_mean, sim_std, _ = self.state.update_grids(forward_t +
                                                       self.init_state_time)
        return sim_mean, sim_std
