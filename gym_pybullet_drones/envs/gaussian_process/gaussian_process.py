import copy
import enum
import numpy as np
import math
from itertools import product
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
import warnings
from sklearn.exceptions import ConvergenceWarning
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
        # description : 由相对位置驱动
         ---------------
        # param :
         - target_num: 无人机的数量,默认为2
         - init_other_pose: [target_0.x target_0.y target_1.x target_1.y] 可能为 None
         ---------------
        # returns :
         ---------------
        """
        self.target_num = target_num
        # TODO：此处需要考虑位置的归一化问题，当前仅进行简单的归一化
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

        self.trajectories = [self.mean.copy()]
        self.y_true_lists = [self.fn(self.grid)]
        pass

    def step(self, other_pose: np.ndarray):
        """
        ## param:
        other_pose: np.ndarray or list contains: [t0.x t0.y t1.x t1.y ...]
        """
        self.mean = np.array(other_pose).reshape(-1, 2) / self._pose_max_val
        self.trajectories += [self.mean.copy()]
        self.y_true_lists += [self.fn(self.grid)]

    def fn(self, X: np.ndarray):
        """
        # description : 计算一个二维数组 X 中每个点的函数值 y
         ---------------
        # param :
         - X : in shape (grid_size^2,2)
         ---------------
        # returns :
        - y : 表示每个坐标对应的每个高斯函数的函数值
         ---------------
        """
        pass
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
        # kernel
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
        self.negitive_kernel_length_scale = [2, 2, 4]
        self.negitive_kernel = Matern(
            length_scale=self.negitive_kernel_length_scale, nu=1.5)
        self.negitive_gp = GaussianProcessRegressor(
            kernel=self.negitive_kernel, optimizer=None)
        self.neg_observed_points = deque()
        self.neg_observed_value = deque()
        self.fov_mask_queue = deque()
        self.neg_std_at_grid = None
        self.observed_points = []
        self.observed_value = []
        self.node_coords = node_coords
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

    def update_negitive_gp_data(self, X=None, Y=None, mask=None):
        # 存储数据点
        self.neg_observed_points.append(X)
        self.neg_observed_value.append(Y)
        self.fov_mask_queue.append(mask)
        # 删除旧数据
        scale_t = self.negitive_kernel_length_scale[-1]
        # Matern1.5: 2.817: 0.1%; 1.993: 1%; 1.376: 5%; 1.093: 10%
        dt = 1.993 * scale_t
        curr_t = X.squeeze()[-1]  # 时间为最后一项
        while curr_t - self.neg_observed_points[0].squeeze()[-1] >= dt:
            self.neg_observed_points.popleft()
            self.neg_observed_value.popleft()
            self.fov_mask_queue.popleft()

    def fit_negitive_gp_at_time(self, curr_t=None):
        std_neg_list = []
        for index in range(len(self.neg_observed_points)):
            self.negitive_gp.fit(
                self.neg_observed_points[index].reshape(-1, 3),
                np.array(self.neg_observed_value[index]).reshape(-1, 1))
            fov_mask = self.fov_mask_queue[index]
            std_neg_at_index = self.update_negitive_grid(curr_t, fov_mask)
            std_neg_list.append(std_neg_at_index +
                                np.ones_like(std_neg_at_index) *
                                (~fov_mask.astype(bool)).astype(int))
        self.neg_std_at_grid = np.min(np.asarray(std_neg_list), axis=0)
        return self.neg_std_at_grid

    def update_negitive_gp(self, X=None, Y=None, mask=None):
        self.update_negitive_gp_data(X=X, Y=Y, mask=mask)
        return self.fit_negitive_gp_at_time(curr_t=X.squeeze()[-1])

    def update_negitive_grid(self, t, fov_mask=None):
        '''
        '''
        _, std_neg_at_grid = self.negitive_gp.predict(add_t(self.grid, t),
                                                      return_std=True)
        std_neg_at_grid = std_neg_at_grid * fov_mask
        return std_neg_at_grid

    def update_node(self, t):
        '''
        predict using GaussianProcess with node_coords
        '''
        self.y_pred_at_node, self.std_at_node = self.gp.predict(
            add_t(self.node_coords, t), return_std=True)
        return self.y_pred_at_node, self.std_at_node

    def update_grid(self, t):
        """
        predict using tha Gaussian Process with grid
        """
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

    def evaluate_F1score(self, y_true, t):
        score = self.gp.score(add_t(self.grid, t), y_true)
        return score

    def evaluate_cov_trace(self, idx=None, t=None):
        if t is not None:
            self.update_grid(t)
        if idx is not None:
            X = self.std_at_grid[idx]
            return np.sum(X * X)
        else:
            return np.sum(self.std_at_grid * self.std_at_grid)

    def evaluate_unc(self, idx=None, t=None):
        # 评估在特定点的不确定性水平
        if t is not None:
            self.update_grid(t)
        if idx is not None:
            X = self.std_at_grid[idx]
            return np.mean(X)
        else:
            return np.mean(self.std_at_grid)

    def evaluate_mutual_info(self, t):
        n_sample = self.grid.shape[0]
        _, cov = self.gp.predict(add_t(self.grid, t), return_cov=True)
        # 评估时间t时，模型预测不确定性大小
        mi = (1 / 2) * np.log(
            np.linalg.det(0.01 * cov.reshape(n_sample, n_sample) +
                          np.identity(n_sample)))
        return mi

    def evaluate_KL_div(self, y_true, t=None, norm=True, base=None):
        if t is not None:
            self.update_grid(t)
        y_pred = copy.deepcopy(self.y_pred_at_grid)
        y_pred[y_pred < 0] = 0
        P = np.array(y_true) + 1e-8
        Q = np.array(y_pred).reshape(-1) + 1e-8
        if norm:
            P /= np.sum(P, axis=0, keepdims=True)
            Q /= np.sum(Q, axis=0, keepdims=True)
        vec = P * np.log(P / Q)
        S = np.sum(vec, axis=0)
        if base is not None:
            S /= np.log(base)
        return S

    def evaluate_JS_div(self, y_true, t=None, norm=True):
        if t is not None:
            self.update_grid(t)
        y_pred = copy.deepcopy(self.y_pred_at_grid)
        y_pred[y_pred < 0] = 0
        P = np.array(y_true) + 1e-8
        Q = np.array(y_pred).reshape(-1) + 1e-8
        if norm:
            P /= np.sum(P, axis=0, keepdims=True)
            Q /= np.sum(Q, axis=0, keepdims=True)
        M = 0.5 * (P + Q)
        vec_PM = P * np.log(P / M)
        vec_QM = Q * np.log(Q / M)
        KL_PM = np.sum(vec_PM, axis=0)
        KL_QM = np.sum(vec_QM, axis=0)
        JS = 0.5 * (KL_PM + KL_QM)
        return JS


class GaussianProcessWrapper:

    def __init__(self, num_uav: int, node_coords: np.ndarray, id=0) -> None:
        """
        # description :
         ---------------
        # param :
         - num_uav: 无人机总数量
         - node_coords: 随机初始化的一组点(初始观测),in shape(-1,2)
         - id: 无人机自身编号 1 ~ num_uav
         ---------------
        # returns :
         ---------------
        """

        self.num_uav = num_uav
        assert id > 0
        self.id = id
        self.other_list = list(range(1, self.id)) + list(
            range(self.id + 1, num_uav + 1))
        self.node_coords = node_coords
        self.GPs: list[GaussianProcess] = [
            GaussianProcess(
                node_coords=node_coords,
                adaptive_kernel=False,
                id=self.id,
                other_id=other,
            ) for other in self.other_list
        ]
        self.curr_t = None
        self.kTargetExistBeliefThreshold = 0.4
        self.kHighInfoIdxThreshold = math.exp(-0.5)
        self.kAddNegitiveGP = True

    def add_init_measures(self, all_point_pos):
        for i, gp in enumerate(self.GPs):
            gp.add_observed_point(all_point_pos[i].reshape(-1, 3), 1.0)

    def add_observed_points(self, point_pos: 'list[np.ndarray]',
                            values: 'list'):  # value: (1, n)
        assert len(self.GPs) == len(point_pos)
        for i, gp in enumerate(self.GPs):
            gp.add_observed_point(point_pos[i], values[i])

    def update_GPs(self):
        for id, GP in enumerate(self.GPs):
            GP.update_gp()

    def update_negititve_gps(self, X=None, Y=None, time=None, mask=None):
        '''
        ### param:
            mask: fake_fov mask
        ### return:
            返回 neg_std
        当前仅将 negitive_gp 附加在 self.GPs[0] 之上
        '''
        if time is not None:
            self.curr_t = time

        return self.GPs[0].update_negitive_gp(add_t(X, self.curr_t), Y, mask)

    def update_node_feature(self, t):
        """
        t: time of now

        未后续提供 feature
        """
        if t != self.curr_t:
            self.curr_t = t
            self.update_grids()

        node_info, node_info_future = [], []  # (target, node, 2)
        for gp in self.GPs:
            node_pred, node_std = gp.update_node(t)
            node_pred_future, node_std_future = gp.update_node(t + 2)
            node_info += [
                np.hstack((node_pred.reshape(-1, 1), node_std.reshape(-1, 1)))
            ]
            node_info_future += [
                np.hstack((node_pred_future.reshape(-1, 1),
                           node_std_future.reshape(-1, 1)))
            ]
        node_feature = np.concatenate(
            (np.asarray(node_info), np.asarray(node_info_future)),
            axis=-1)  # (target, node, features(4))
        node_feature = node_feature.transpose(
            (1, 0, 2)).reshape(self.node_coords.shape[0],
                               -1)  # (node, (targetxfeature))
        # contiguous at feature level
        return node_feature

    def update_grids(self, time: float = None):
        '''
        if time is None, use self.curr_t as time

        return: all_pred, all_std, preds
        '''
        if time is None and self.curr_t is not None:
            time = self.curr_t
        if time is not None:
            self.curr_t = time
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
        # 叠加 negitive_gp

        return all_pred, all_std, preds

    def get_observed_points(self, kTargetExistBeliefThreshold=None, time=None):
        '''
        返回当前高斯过程中，置信度最大的点，作为 decision 或其他环节输入
        ---
        如果输入时间，则是对未来或当下的估计
        '''
        if kTargetExistBeliefThreshold is None:
            kTargetExistBeliefThreshold = self.kTargetExistBeliefThreshold
        observed_points = []
        grid_size = self.GPs[0].grid_size
        for index, gp in enumerate(self.GPs):
            if time is None:
                y_pred_grid = gp.y_pred_at_grid
            else:
                y_pred_grid, _ = gp.evaulate_grid(time)
            if y_pred_grid is None:
                continue

            y_pred_grid = y_pred_grid.reshape(grid_size, grid_size)
            max_row, max_col = np.unravel_index(y_pred_grid.argmax(),
                                                y_pred_grid.shape)
            # 原点为 self.grid_size / 2, from row,col to [x,y]
            point_vector = np.array([
                max_row - grid_size / 2, max_col - grid_size / 2
            ]) / (grid_size / 2)
            if np.max(y_pred_grid) > kTargetExistBeliefThreshold:
                observed_points.append(point_vector)
        return observed_points

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
        std_trace = []
        if t != self.curr_t:
            self.curr_t = t
            self.update_grids()
        for i, gp in enumerate(self.GPs):
            idx = None if high_info_idx is None else high_info_idx[i]
            std_trace += [gp.evaluate_unc(idx)]
        avg_std_trace = np.mean(std_trace)
        return (avg_std_trace, std_trace) if return_all else avg_std_trace

    def eval_avg_unc_sum(self, unc, high_info_idx=None, return_all=False):
        std_sum = []
        num_high = list(map(len, high_info_idx))
        for i in range(len(self.GPs)):
            std_sum += [unc[i] * num_high[i]]
        avg_std_sum = np.mean(std_sum)
        return (avg_std_sum, std_sum) if return_all else avg_std_sum

    def eval_avg_KL(self, y_true, t, return_all=False):
        KL = []
        if t != self.curr_t:
            self.curr_t = t
            self.update_grids()
        for i, gp in enumerate(self.GPs):
            KL += [gp.evaluate_KL_div(y_true[:, i])]
        avg_KL = np.mean(KL)
        return (avg_KL, KL) if return_all else avg_KL

    def eval_avg_JS(self, y_true, t, return_all=False):
        JS = []
        if t != self.curr_t:
            self.curr_t = t
            self.update_grids()
        for gp in self.GPs:
            JS += [
                np.amax([
                    gp.evaluate_JS_div(y_true[:, i])
                    for i in range(self.num_uav - 1)
                ])
            ]
        avg_JS = np.mean(JS)
        return (avg_JS, JS) if return_all else avg_JS

    def eval_all_js(self, y_true_all, y_pred_all, norm=True):
        P = np.array(y_true_all) + 1e-8
        Q = np.array(y_pred_all).reshape(-1) + 1e-8
        if norm:
            P /= np.sum(P, axis=0, keepdims=True)
            Q /= np.sum(Q, axis=0, keepdims=True)
        M = 0.5 * (P + Q)
        vec_PM = P * np.log(P / M)
        vec_QM = Q * np.log(Q / M)
        KL_PM = np.sum(vec_PM, axis=0)
        KL_QM = np.sum(vec_QM, axis=0)
        JS = 0.5 * (KL_PM + KL_QM)
        return JS

    def eval_avg_F1(self, y_true, t, return_all=False):
        F1 = []
        if t != self.curr_t:
            self.curr_t = t
            self.update_grids()
        for i, gp in enumerate(self.GPs):
            F1 += [gp.evaluate_F1score(y_true[:, i], self.curr_t)]
        avg_F1 = np.mean(F1)
        return (avg_F1, F1) if return_all else avg_F1

    def eval_avg_MI(self, t, return_all=False):
        MI = []
        if t != self.curr_t:
            self.curr_t = t
            self.update_grids()
        for gp in self.GPs:
            MI += [gp.evaluate_mutual_info(self.curr_t)]
        avg_MI = np.mean(MI)
        return (avg_MI, MI) if return_all else avg_MI


if __name__ == "__main__":
    pass
