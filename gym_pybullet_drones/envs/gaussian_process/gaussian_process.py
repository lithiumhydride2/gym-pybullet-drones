import copy
import enum
import numpy as np
import math
from itertools import product
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from gym_pybullet_drones.envs.gaussian_process.UCB.gp_torch import GaussianProcessTorch
from sklearn.gaussian_process.kernels import RBF
from collections import deque
from sympy import N


def add_t(X, t: float):
    return np.concatenate((X, np.zeros((X.shape[0], 1)) + t), axis=1)


class GaussianProcessGroundTruth:
    """
    用作每个无人机对 Gaussian Process 真值的处理
    """

    def __init__(self, target_num=2, init_other_pose: list = None) -> None:
        """
        ### description : 由相对位置驱动
         ---------------
        ### param :
         - target_num: 其他无人机的数量
         - init_other_pose: [target_0.x target_0.y target_1.x target_1.y] 可能为 None
         ---------------
        ### returns :
         ---------------
        """
        self.target_num = target_num
        self._pose_max_val = 6.0  # 供归一化使用
        assert init_other_pose is not None
        self.mean = (np.array(init_other_pose).reshape(-1, 2) /
                     self._pose_max_val)  # in shape (target_num,2)
        self.sigma = np.array([0.1] * self.target_num)  # 此 sigma 为给定超参
        self.max_value = 1 / (2 * np.pi * self.sigma**2)  # 用于高斯函数值的归一化

        self.grid_size = 40
        self.grid = np.array(
            list(
                product(
                    np.linspace(-1, 1, self.grid_size),
                    np.linspace(-1, 1, self.grid_size),
                )))  # in shape (self.grid_size **2 , 2)

    def step(self, other_pose: np.ndarray):
        """
        由无人机真实相对位置，更新 mean of gaussian_process
        ## param:
        other_pose: np.ndarray or list contains: [t0.x t0.y t1.x t1.y ...]
        """
        self.mean = np.array(other_pose).reshape(-1, 2) / self._pose_max_val

    def get_high_info_indx(self,
                           ground_truth,
                           high_info_threshold=math.exp(-0.5)):
        '''
        ground_truth 由 GaussianProcessGroundTruth.fn()给出
        '''
        high_info_idx = []
        for i in range(self.target_num):
            idx = np.argwhere(ground_truth[:, i] > high_info_threshold)
            high_info_idx += [idx.squeeze(1)]
        return high_info_idx

    def fn(self):
        """
        # description:
            根据 self.mean 计算真实高斯分布
         ---------------
        # param :
         - X : in shape (grid_size^2,2)
         ---------------
        # returns :
        - y : 表示每个坐标对应的每个高斯函数的函数值
         ---------------
        """
        X = self.grid
        y = np.zeros(shape=(
            X.shape[0],
            self.target_num))  # in shape (grid_size**2, self.target_num)
        row_mat, col_mat = X[:, 0], X[:, 1]

        for target_id in range(self.target_num):
            gaussian_mean = self.mean[target_id]
            # suppose covariance is zero
            sigma_x1 = sigma_x2 = self.sigma[target_id]
            covariance = 0
            # 相关系数
            r = covariance / (sigma_x1 * sigma_x2)
            coefficients = 1 / (2 * math.pi * sigma_x1 * sigma_x2 *
                                np.sqrt(1 - math.pow(r, 2)))
            p1 = -1 / (2 * (1 - math.pow(r, 2)))
            px = np.power((row_mat - gaussian_mean[0]) / sigma_x1, 2)
            py = np.power((col_mat - gaussian_mean[1]) / sigma_x2, 2)
            pxy = (2 * r * (row_mat - gaussian_mean[0]) *
                   (col_mat - gaussian_mean[1]) / (sigma_x1 * sigma_x2))
            distribution_matrix = coefficients * np.exp(p1 * (px - pxy + py))
            y[:, target_id] += distribution_matrix
        y /= self.max_value
        return y


class GaussianProcess:

    def __init__(self, node_coords, adaptive_kernel=False, id=0, other_id=0):
        """
        # description :
         ---------------
        # param :
         - node_coords: 一组随机初始化的数据点
         - adaptive_kernel: 是否为 adaptive kernel
         - id: 无人机的编号, 应为 1~ num_uav
         - other_id: 为一个不同于 id 的编号
         ---------------
        # returns :
         ---------------
        """
        self.id = id
        self.other_id = other_id
        self.fig_name = "UAV_{}_GP".format(self.id)
        self.color_map = plt.get_cmap("Set1").colors
        ########### guassian process for other uav
        if adaptive_kernel:
            # length_scale: 每个维度定义各自特征维度的长度尺度
            # length_scale_bounds: 参数可调整的上下限
            self.kernel = Matern(length_scale=[0.1, 0.1, 3],
                                 length_scale_bounds=[1e-2, 1e1],
                                 nu=1.5)
            self.gp = GaussianProcessRegressor(kernel=self.kernel,
                                               optimizer="fmin_l_bfgs_b",
                                               n_restarts_optimizer=1,
                                               alpha=1e-2)
        else:
            self.kernel = Matern(length_scale=[0.1, 0.1, 3], nu=1.5)
            self.gp = GaussianProcessRegressor(kernel=self.kernel,
                                               optimizer=None,
                                               n_restarts_optimizer=20)
        ############ gaussian process for FOV
        self.negitive_kernel_length_scale = [2, 2, 4]
        self.negitive_kernel = Matern(
            length_scale=self.negitive_kernel_length_scale, nu=1.5)
        self.negitive_gp = GaussianProcessRegressor(
            kernel=self.negitive_kernel, optimizer=None)
        self.neg_observed_points = deque()
        self.neg_observed_value = deque()
        self.fov_mask_queue = deque()
        self.neg_std_at_grid = None
        ############# stack for information
        self.observed_points = []
        self.observed_value = []
        self.y_pred_at_node, self.std_at_node = None, None
        self.y_pred_at_grid, self.std_at_grid = None, None
        self.y_pred_at_grid_lists = []
        self.grid_size = 40
        self.grid = np.array(
            list(
                product(
                    np.linspace(-1, 1, self.grid_size),
                    np.linspace(-1, 1, self.grid_size),
                )))  # in shape (self.grid_size **2 , 2)
        self.curr_t = -1.0  # 标记上一次 update_grid 时刻

    def add_observed_point(self, point_pos, value):
        self.observed_points.append(point_pos)
        self.observed_value.append(value)

    def update_gp(self):
        """
        # description :
        - scale_t: 高斯过程的长度尺度参数，控制核函数平滑程度，如果已经拟合过，为 self.gp.kernel_
        - dt: 用于筛选最近的观测点,1.993表示时间间隔内，观测点相关性超过 99%
        - curr_t: 最后一个观测的时间戳
        将满足 存储条件的观测点的索引 存入 mask_idx, 如果 mask_idx 不为空, 则利用所有数据进行拟合
         ---------------
        # param :
         ----------
        # returns :
         ----------
        ## Matern
        https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html
        """
        scale_t = (self.gp.kernel_.length_scale[-1] if hasattr(
            self.gp, "kernel_") else self.gp.kernel.length_scale[-1])
        # Matern1.5: 2.817: 0.1%; 1.993: 1%; 1.376: 5%; 1.093: 10%
        dt = 1.993 * scale_t
        curr_t = self.observed_points[-1][0][-1] if self.observed_points else 0
        if curr_t != 0:
            dt = 1.993 * scale_t
        mask_idx = []

        for i, ob in enumerate(self.observed_points):
            if curr_t - ob[0][-1] < dt:
                mask_idx.append(i)
        if self.observed_points:
            X = np.vstack([self.observed_points[idx]
                           for idx in mask_idx]).reshape(-1, 3)
            y = np.hstack([self.observed_value[idx]
                           for idx in mask_idx]).reshape(-1, 1)
            try:
                self.gp.fit(X, y)
            except Exception as e:
                print("X is : ", X)
                print("y is : ", y)
                print("mask_idx is : ", mask_idx)
                print("ovserved_points is :", self.observed_points)

    def update_node(self, t):
        '''
        predict using GaussianProcess with node_coords
        '''
        self.y_pred_at_node, self.std_at_node = self.gp.predict(
            add_t(self.node_coords, t), return_std=True)
        return self.y_pred_at_node, self.std_at_node

    def update_grid(self, t):
        """
        predict using tha Gaussian Process with grid, and storage it
        """
        # use cache
        if self.curr_t == t:
            return self.y_pred_at_grid, self.std_at_grid

        self.curr_t = t
        if self.y_pred_at_grid is None:
            self.y_pred_at_grid, self.std_at_grid = self.gp.predict(
                add_t(self.grid, t), return_std=True)
        # 记录历史状态
        return self.y_pred_at_grid, self.std_at_grid

    def evaulate_grid(self, t):
        y_pred_at_grid, std_at_grid = self.gp.predict(add_t(self.grid, t),
                                                      return_std=True)
        return y_pred_at_grid, std_at_grid

    def evaluate_RMSE(self, y_true, t=None):
        if t is not None:
            self.update_grid(t)
        RMSE = np.sqrt(mean_squared_error(self.y_pred_at_grid, y_true))
        return RMSE

    def evaluate_cov_trace(self, idx=None, t=None):
        if t is not None:
            self.update_grid(t)
        if idx is not None:
            X = self.std_at_grid[idx]
            return np.sum(X * X)
        else:
            return np.sum(self.std_at_grid * self.std_at_grid)

    def evaluate_unc(self, idx=None, t=None):
        '''
        在 t 时刻评估 unc , t 必须有效
        '''
        # 评估在特定点的不确定性水平
        if t is None or t != self.curr_t:
            raise ValueError
        if idx is not None:
            X = self.std_at_grid[idx]
            return np.mean(X)
        else:
            return np.mean(self.std_at_grid)


class GaussianProcessWrapper:

    def __init__(self,
                 num_uav: int,
                 other_list: list[int],
                 id=0,
                 use_gpytorch=True) -> None:
        """
        ### param :
         - num_uav: 无人机总数量
         - other_list: list of other index
         - id: 无人机自身编号 0 ~ num_uav - 1
         ---------------
        ### returns :
         ---------------
        """

        self.num_uav = num_uav
        self.id = id
        self.other_list = other_list
        if use_gpytorch:
            self.GPs: list[GaussianProcessTorch] = [
                GaussianProcessTorch(id=self.id, other_id=other)
                for other in self.other_list
            ]
        else:
            self.GPs = [
                GaussianProcess(
                    adaptive_kernel=False,
                    id=self.id,
                    other_id=other,
                ) for other in self.other_list
            ]
        self.curr_t = None  # self.curr_t 用来记录上一次调用 self.update_grids 的时刻
        self.kTargetExistBeliefThreshold = 0.4
        self.kHighInfoIdxThreshold = math.exp(-0.5)
        self.kAddNegitiveGP = True

    def GPbyOtherID(self, id):
        for gp in self.GPs:
            if id == gp.other_id:
                return gp
        # if not find this gp
        raise KeyError

    def add_init_measures(self, all_point_pos):
        for i, gp in enumerate(self.GPs):
            gp.add_observed_point(all_point_pos[i].reshape(-1, 3), 1.0)

    def add_observed_points(self, point_pos: 'list[np.ndarray]',
                            values: 'list'):  # value: (1, n)
        assert len(self.GPs) == len(point_pos)
        for i, gp in enumerate(self.GPs):
            gp.add_observed_point(point_pos[i], values[i])

    def update_GPs(self):
        for _, GP in enumerate(self.GPs):
            GP.update_gp()

    def update_node_feature(self, time, node_coords):
        '''
        update node feature at time t
        '''
        node_info = []
        for gp in self.GPs:
            node_pred, node_std = gp.update_node(time, node_coords)
            node_pred: np.ndarray
            node_std: np.ndarray
            node_info += [
                np.hstack((node_pred.reshape(-1, 1), node_std.reshape(-1, 1)))
            ]
        node_feature = np.asarray(node_info)  #(target,node,2 (features) )
        node_feature = node_feature.transpose(1, 0, 2).reshape(
            node_coords.shape[0], -1)  #(node, target * feature)
        return node_feature

    def update_grids(self, time: float = None):
        '''
        update grids at time t
        return: all_pred, all_std, preds, stds
        '''
        if time is None:
            raise ValueError

        preds = []
        stds = []
        for gp in self.GPs:
            pred_temp, std_temp = gp.update_grid(time)
            preds.append(pred_temp.squeeze())
            stds.append(std_temp.squeeze())
        # take maximum
        all_pred = np.asarray(preds).max(axis=0)
        # take minimum
        all_std = np.asarray(stds).min(axis=0)

        return all_pred, all_std, np.asarray(preds), np.asarray(stds)

    def get_high_info_idx(self,
                          source="detection",
                          kHighInfoIdxThreshold=None):
        '''
        返回 detection 的 high_info_idx , 供 eval_avg_unc 使用以增强探索行为
        ### param
        - source : dete 来源，如为 detection , 则无 ground_truth
        '''
        if kHighInfoIdxThreshold is None:
            kHighInfoIdxThreshold = self.kHighInfoIdxThreshold
        high_info_idx = []
        for index, gp in enumerate(self.GPs):
            idx = np.argwhere(
                gp.y_pred_at_grid.reshape(1600, ) > kHighInfoIdxThreshold)
            high_info_idx += [idx.squeeze(1)]
        return high_info_idx

    def eval_avg_RMSE(self, y_true, t, return_all=False):
        RMSE = []
        if t != self.curr_t:
            self.curr_t = t
            self.update_grids()
        for i, gp in enumerate(self.GPs):
            RMSE += [gp.evaluate_RMSE(y_true[:, i])]
        avg_RMSE = np.sqrt(np.mean(np.square(RMSE)))  # quadratic mean
        return (avg_RMSE, RMSE) if return_all else avg_RMSE

    def eval_avg_cov_trace(self, t, high_info_idx=None, return_all=False):
        cov_trace = []
        if t != self.curr_t:
            self.curr_t = t
            self.update_grids()
        for i, gp in enumerate(self.GPs):
            idx = None if high_info_idx is None else high_info_idx[i]
            cov_trace += [gp.evaluate_cov_trace(idx)]
        avg_cov_trace = np.mean(cov_trace)
        return (avg_cov_trace, cov_trace) if return_all else avg_cov_trace

    def eval_unc_with_grid(self,
                           high_info_idx=None,
                           std_at_grid: np.ndarray = None):
        """
        # description :
         --------------- 
        # param :
         - high_info_idx: 如果为 None, 自己根据 参数获取 high_info_idx
         --------------- 
        # returns :
        std_trace , np.mean(std_trace)
         --------------- 
        """

        std_trace = []
        if high_info_idx is None:
            high_info_idx = self.get_high_info_idx()
        if std_at_grid is None:
            std_at_grid = self.GPs[0].std_at_grid

        for i, gp in enumerate(self.GPs):
            idx = high_info_idx[i]
            if idx.size == 0:
                std_trace += [np.mean(std_at_grid)]
            else:
                std_trace += [np.mean(std_at_grid[idx])]
        return std_trace, np.mean(std_trace)

    def eval_avg_unc(self, t, high_info_idx=None, return_all=False):
        '''
        eval UNC at timestamp t
        '''
        std_trace = []
        if t != self.curr_t:
            self.curr_t = t
            self.update_grids(t)

        for i, gp in enumerate(self.GPs):
            idx = None if high_info_idx is None else high_info_idx[i]
            std_trace += [gp.evaluate_unc(idx, t)]
        avg_std_trace = np.mean(std_trace)
        return (avg_std_trace, std_trace) if return_all else avg_std_trace

    def eval_avg_unc_sum(self, unc, high_info_idx=None, return_all=False):
        std_sum = []
        num_high = list(map(len, high_info_idx))
        for i in range(len(self.GPs)):
            std_sum += [unc[i] * num_high[i]]
        avg_std_sum = np.mean(std_sum)
        return (avg_std_sum, std_sum) if return_all else avg_std_sum


if __name__ == "__main__":
    pass
