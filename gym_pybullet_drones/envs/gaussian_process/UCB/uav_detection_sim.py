from tkinter import SE
import numpy as np
from copy import deepcopy
from ..gaussian_process import GaussianProcessWrapper, add_t


class UavDetectionSim:

    def __init__(self, fake_fov_range, num_uav) -> None:
        self.kTargetExistBeliefThreshold = 0.8
        self.fake_fov_range = fake_fov_range
        self.ego_heading = 0.0
        self.curr_t = 0.0
        self.num_uav = num_uav
        self.num_latent_target = num_uav - 1

    def reset(self, ego_heading, gp_wrapper: 'GaussianProcessWrapper' = None):
        """
        # description :
        固定 reset 调用时的 坐标系进行仿真推演
        """
        self.ego_heading = ego_heading
        self.latest_observed_point = gp_wrapper.get_observed_points(
            kTargetExistBeliefThreshold=self.kTargetExistBeliefThreshold)
        self.curr_t = gp_wrapper.curr_t if gp_wrapper.curr_t is not None else 0.0

        # 更新 gp_wrapper
        if not hasattr(self, "gp_wrapper"):
            self.gp_wrapper = deepcopy(gp_wrapper)
        else:
            self.gp_wrapper.GPs = deepcopy(gp_wrapper.GPs)
            # self.gp_wrapper = deepcopy(gp_wrapper)

    def step_primitive(self, primitive, time_span):
        '''
        利用运动基元进行仿真推演
        '''
        primitive_reward = 1
        # 如果造成丢失，直接返回,此处无需评估时间，time_span 设置为0
        self.__take_action(primitive[-1, -1], 0)
        for observe_point in self.latest_observed_point:
            if not self.__in_fov(observe_point):
                self.ego_heading -= primitive[-1][-1]
                return primitive_reward
        self.ego_heading -= primitive[-1][-1]

        # 仅评估轨迹最后一条
        for id, action in enumerate(primitive[-1:]):
            heading_action = action[-1]
            self.__take_action(heading_action, time_span)
            self.gp_wrapper.GPs[0].update_negitive_gp_data(
                X=add_t(np.zeros((1, 2)), self.curr_t),
                Y=0,
                mask=self.__get_fov_mask())
            # 仅在最后一个 segment , 评估 unc reward
            if id == len(primitive) - 1:
                neg_std = self.gp_wrapper.GPs[0].fit_negitive_gp_at_time(
                    self.curr_t)
                _, unc_reward = self.gp_wrapper.eval_unc_with_grid(
                    high_info_idx=None, std_at_grid=neg_std)
                primitive_reward = unc_reward

        return primitive_reward

    def step(self, action, time_span):
        """
        # description :
         --------------- 
        # param :
         - action: 下一个决策值
         - time_span: action 的时间跨度
         - gp_wrapper: gaussian process
         --------------- 
        # returns :
         --------------- 
        """
        unc_reward = 1.0
        # 仅评估是否造成目标丢失
        self.__take_action(action, time_span)
        # 如果造成丢失
        # 在数量较大时，考虑从一直探索。
        for observe_point in self.latest_observed_point:
            if self.num_latent_target <= 5 and not self.__in_fov(
                    observe_point):
                return unc_reward

        # 评估更新航向角之后的 gp
        self.gp_wrapper.GPs[0].update_negitive_gp_data(
            X=add_t(np.zeros((1, 2)), self.curr_t),
            Y=0,
            mask=self.__get_fov_mask())

        neg_std = self.gp_wrapper.GPs[0].fit_negitive_gp_at_time(self.curr_t)
        _, unc_reward = self.gp_wrapper.eval_unc_with_grid(
            high_info_idx=np.array([[]] * self.num_latent_target),  # or None
            std_at_grid=neg_std)
        return unc_reward

    def __take_action(self, action, time_span):
        '''
        take action, update state and fov
        '''
        self.ego_heading += action
        self.fake_fov = self.ego_heading + self.fake_fov_range
        self.curr_t += time_span

    def __in_fov(self, point: np.ndarray):
        point = point.squeeze()
        # fov 与计算坐标系存在转换
        fov_angle = self.fake_fov + np.pi / 2
        fov_vector = [
            np.array([np.cos(fov), np.sin(fov)]) for fov in fov_angle
        ]
        return_val = (np.cross(fov_vector[0], fov_vector[1]) *
                      np.dot(fov_vector[0], fov_vector[1])) < 0
        if ~return_val:
            fov_vector = fov_vector[::-1]
        if np.cross(fov_vector[0], point) >= 0 and np.cross(
                point, fov_vector[1]) >= 0:
            return return_val
        return ~return_val

    def __get_fov_mask(self):
        '''
        根据当前 FOV 产生 mask
        '''
        grid = self.gp_wrapper.GPs[0].grid
        conditions = np.apply_along_axis(self.__in_fov, 1, grid)
        return conditions.astype(int)
