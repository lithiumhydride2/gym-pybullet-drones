import numpy as np

from gym_pybullet_drones.envs.gaussian_process.gaussian_process import GaussianProcessGroundTruth
from gym_pybullet_drones.envs.gaussian_process.gaussian_process import GaussianProcessWrapper
from gym_pybullet_drones.envs.gaussian_process.UCB.tsp_base_line import TSPBaseLine
from gym_pybullet_drones.utils.utils import *
# from .metrics import Metrics
# from .util.utils import *


def add_t(X, t: float):
    """
    add timestamp for measurement
    """
    return np.concatenate((X, np.zeros((X.shape[0], 1)) + t), axis=1)


class UAVGaussian():

    def __init__(
        self,
        fov_range,
        nth_drone=0,
        num_drone=3,
        planner="tsp",
        **kwargs,
    ) -> None:
        """
        ## description :
         ---------------
        ## param :
         - id: 无人机编号, 为 1 ~ num_uav
         - planner: tsp|rl
        ## kwargs:
         - enable_exploration: 是否使能探索功能
        """

        self.id = nth_drone
        self.num_drone = num_drone
        self.other_list = list(set(range(num_drone)) - set([nth_drone]))

        self.num_latent_target = len(self.other_list)  # 潜在目标数量
        self.last_detection_target_map = None

        ########### gaussian_process_part ############
        self.GP_ground_truth = GaussianProcessGroundTruth(
            target_num=self.num_latent_target,
            init_other_pose=np.zeros(
                (2 * self.num_latent_target, )))  # x,y init as 0.0

        self.GP_detection = GaussianProcessWrapper(num_uav=self.num_drone,
                                                   other_list=self.other_list,
                                                   node_coords=np.zeros(
                                                       (1, 2)),
                                                   id=self.id)

        self.cache = {}  # cache
        ######### metrics
        self.last_all_std = None
        # self.metrics_obj = Metrics(**kwargs)
        ################### planner select ##############

        if planner == "tsp":
            self.planner = TSPBaseLine(
                num_latent_target=self.num_latent_target,
                fake_fov_range=fov_range,
                **kwargs)
        else:
            raise NameError
        # fov 需要设置为参数
        self.ego_heading = 0.0  # 自身朝向状态量
        self.last_ego_heading = 0.0  # 上一个朝向状态量
        self.last_yaw_action = 0.0  # 上一个朝向输出

        # 首先关闭 FOV 对高斯过程的影响
        self.kFovEffectGP = False  # FOV信息是否影响高斯过程
        self.last_negitive_sample_time = 0.0

    def _gp_step(self,
                 detection_map: dict[int, np.ndarray] = None,
                 other_pose=None,
                 ego_heading=None,
                 fov_vector=None,
                 time=None):
        """
        ### description : 进行 gp_ground_truth 和 gp_detection 的step
         ---------------
        ### param :
         - detection_map: { 0:pos_target_0, 1:pos_target_1 ... },if no detection, the value is none
         ---------------
        ### returns :
        - all_std: 叠加了(如果打开了 kFovEffectGP 开关) negitive sample 之后的 std
         ---------------
        """
        # TODO(lih): gaussian process part
        all_std = None
        ### Ground Truth Part
        self.GP_ground_truth.step(other_pose)

        ### Detection part
        observe_value = np.ones(1, )
        for other_nth in detection_map.keys():
            observe_point = detection_map[other_nth].reshape(-1,
                                                             2) / 6.0  # 归一化
            self.GP_detection.GPbyOtherID(other_nth).add_observed_point(
                add_t(observe_point, time), observe_value)

        # 更新 GP 参数
        self.GP_detection.update_GPs()
        all_pred, all_std, _ = self.GP_detection.update_grids(time)

        ##### 触发 fov 影响的采集
        # if self.kFovEffectGP and self.negitive_gather:
        #     neg_std = self.GP_detection.update_negititve_gps(
        #         X=np.zeros((1, 2)),
        #         Y=0,
        #         time=time,
        #         mask=self.__get_fov_mask(fov_vector))
        #     all_std = np.min(np.array([neg_std, all_std]), axis=0)
        # ##### 否则叠加最新的 fov effect
        # elif self.kFovEffectGP and len(self.all_stds_list):
        #     all_std = self.all_stds_list[-1]
        self.cache["all_std"] = all_std
        self.cache["all_pred"] = all_pred
        return all_std

    @property
    def negitive_gather(self):
        if self.curr_time - self.last_negitive_sample_time >= 0.5:
            self.last_negitive_sample_time = self.curr_time
            return True
        return False

    def __get_fov_mask(self, fov_vector):
        '''
        根据当前 FOV 产生 mask
        Args:
            fov_vector: 由两向量组成的 fov_vector
        '''
        grid = self.GP_detection.GPs[0].grid

        def in_fov_along_axis(point):
            return in_fov(point, fov_vector)

        conditions = np.apply_along_axis(in_fov_along_axis, 1, grid)
        return conditions.astype(int)

    def step_flocking_metrics(self, detection_absolute_index):
        detection_data = np.array(self.detection.loc[detection_absolute_index])
        curr_time = detection_data[0]
        self.curr_index = self.pose.loc[self.pose["Time"] >= curr_time].head(
            1).index[0]
        self.__update_relative_pos()
        detection_pose = detection_data[1:].reshape(-1, 2)

        flocking_graph = dict()
        for other in self.other_list:
            flocking_graph[other] = 0
        ground_truth_pose = np.array(
            self.relative_pose.loc[self.curr_index]).reshape(-1, 2)
        # 补充自身位置的连接
        # 在自身位置处插入 zero
        ground_truth_pose = np.insert(ground_truth_pose,
                                      self.id - 1,
                                      np.zeros((1, 2)),
                                      axis=0)

        tree = cKDTree(ground_truth_pose)
        # indice 为对 detection_pose的查询
        _, indice = tree.query(detection_pose)
        estimation_err = [np.array([np.nan, np.nan])] * (self.num_uav)
        for index, sense in enumerate(indice):
            if sense >= self.num_uav:
                continue
            flocking_graph[sense + 1] = 1
            estimation_err[
                sense] = detection_pose[index] - ground_truth_pose[sense]
            # estimation_err.append(detection_pose[index] -
            #                       ground_truth_pose[sense])

        return flocking_graph, ground_truth_pose, estimation_err

    def step(self, curr_time, detection_map, ego_heading, fov_vector,
             relative_pose):
        """
        -----
        Args:
            curr_time: ros传入的当前时刻
            detection_map: key: nth_drone value: position estimation, 需要坐标系下直接计算的 相对位置
            ego_heading: 无人机当前 yaw 角， 世界坐标系下
            fov_vector: 无人机当前 fov, 由两向量组成，世界坐标系下
            relative_pose: 其他无人机的真实位置,需不包含与自身的相对位置
        ----
        ## return:
        all_std: [1600,]
        """
        self.curr_time = curr_time
        self.ego_heading = ego_heading

        # GP_step
        all_std = self._gp_step(detection_map=detection_map,
                                other_pose=relative_pose,
                                ego_heading=ego_heading,
                                fov_vector=fov_vector,
                                time=curr_time)
        return all_std

    def DecisionStep(self, obs):
        '''
        Args:
            obs: 从 gp_step 中得到的 obs, 为 heat_map 的形式
        '''
        action = np.zeros(3)
        action = self.planner.step(self.GP_detection,
                                   curr_t=self.curr_time,
                                   ego_heading=self.ego_heading,
                                   std_at_grid=obs)

        # history
        self.last_ego_heading = self.ego_heading
        self.last_yaw_action = action[-1]

        # metrices
        # if self.fake_ros:
        #     all_pred, _, _ = self.GP_detection.update_grids(self.curr_time)
        #     JS = self.GP_detection.eval_all_js(
        #         np.max(self.GP_ground_truth.y_true_lists[-1], axis=-1),
        #         all_pred)
        #     _, UNC = self.GP_detection.eval_unc_with_grid(std_at_grid=all_std)
        #     _, FOV_UNC = self.GP_detection.eval_unc_with_grid(
        #         high_info_idx=np.array([[]] * self.num_latent_target),
        #         std_at_grid=all_std)
        #     self.metrics_obj.step(jsd=JS, unc=UNC, fovunc=FOV_UNC)
        return self.last_yaw_action

    def save_animation(self, **kwargs):
        """
        保存动画的方法。
            :param dpi: 分辨率, 默认为100。
            :param file_format: 文件格式, 默认为'mp4'。
            :param save_traj: 是否保存轨迹, 默认为False。
            :param save_relative: 是否保存相对位置, 默认为False。
            :param save_gaussian: 是否保存高斯效果, 默认为False。
            :param save_step=10 step为10,则每100ms为一帧数
        """
        self.__setPlot()
        self.uav_animation.__dict__.update(self.__dict__)

        if not (self.id == 1):
            kwargs.setdefault("save_traj", False)
            kwargs["save_traj"] = False
        self.uav_animation.save_animation(**kwargs)

    def save_metrics(self, **kwargs):
        """
        保存 metrics 的方法
            :param dpi: 分辨率, 默认为100。
            :param file_format: 文件格式, 默认为'mp4'。
            :param save_traj: 是否保存轨迹, 默认为False。
            :param save_relative: 是否保存相对位置, 默认为False。
            :param save_gaussian: 是否保存高斯效果, 默认为False。
            :param save_step=10 step为10,则每100ms为一帧数
        """
        self.__setPlot()
        self.metrics_obj.__dict__.update(self.__dict__)
        self.metrics_obj.save_figure(**kwargs)


if __name__ == "__main__":
    print("this is uav gaussian")
