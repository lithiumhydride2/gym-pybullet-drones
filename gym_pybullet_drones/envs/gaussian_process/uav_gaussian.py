import numpy as np

from gym_pybullet_drones.envs.gaussian_process.gaussian_process import GaussianProcessGroundTruth
from gym_pybullet_drones.envs.gaussian_process.gaussian_process import GaussianProcessWrapper
from gym_pybullet_drones.envs.gaussian_process.UCB.tsp_base_line import TSPBaseLine
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
        nth_drone=0,
        num_drone=3,
        planner="tsp",
        **kwargs,
    ) -> None:
        """
        # description :
         ---------------
        # param :
         - id: 无人机编号, 为 1 ~ num_uav
         - planner: tsp|ucb|stamp
         ## **kwargs
         - num_uav : 执行任务无人机数量
         - camera_config: 定义 fov 使用的配置文件
         ---------------
        # returns :
         ---------------
        """

        self.id = nth_drone
        self.num_drone = num_drone
        self.other_list = list(range(1, self.id)) + list(
            range(self.id + 1, self.num_uav + 1))

        self.num_latent_target = len(self.other_list)  # 潜在目标数量
        self.last_detection_target_map = None

        ########### gaussian_process_part
        self.GP_ground_truth = GaussianProcessGroundTruth(
            target_num=self.num_latent_target,
            init_other_pose=np.zeros(
                (2 * self.num_latent_target, )))  # x,y init as 0.0
        # TODO here: 继续修改代码
        self.GP_detection = None  # 由 detection 驱动的 Gaussian Process
        self.all_means_list = []
        self.all_stds_list = []
        self.gps_means_list = []
        self.observed_points_list = []
        ######### metrics
        self.last_all_std = None
        self.metrics_obj = Metrics(**kwargs)
        self.gp_got_first_detection = False
        ################### planner select ##############

        if planner == "tsp":
            self.planner = TSPBaseLine(
                num_latent_target=self.num_latent_target,
                mode_simulation=self.planner_debug_msg,
                fake_fov_range=self.fake_fov_range,
                **kwargs)
        else:
            raise NameError
        # fov 需要设置为参数
        self.fake_fov = np.deg2rad([-20, 20])
        self.fake_fov_list = []
        self.ego_heading = 0.0  # 自身朝向状态量
        self.last_ego_heading = 0.0  # 上一个朝向状态量
        self.last_yaw_action = 0.0  # 上一个朝向输出
        self.kApproveNextActionHeadingThreshold = np.deg2rad(3)

        self.kFovEffectGP = True  # FOV信息是否影响高斯过程
        self.last_negitive_sample_time = 0.0
        # 是否仅进行 analyse
        self.mode_simulation = mode_simulation

    def __update_relative_pos(self):
        """
        根据 fake_ros_msg 提供的 curr_index 更新 self.relative_pose
        """
        if self.other_pose_local is None:
            # all pose here are in world corridnate
            ego_pos = np.array([
                self.pose.loc[self.curr_index]["pose.position.x"],
                self.pose.loc[self.curr_index]["pose.position.y"],
            ] * len(self.other_list))
            other_pos = np.array([
                self.poses[other - 1].loc[self.curr_index][key]
                for other in self.other_list
                for key in ["pose.position.x", "pose.position.y"]
            ])
            relative_pos = other_pos - ego_pos
        else:
            relative_pos = np.array([
                self.other_pose_local[index].loc[self.curr_index][key]
                for index in range(len(self.other_list))
                for key in ["point.x", "point.y"]
            ])
        self.relative_pose.loc[self.curr_index] = relative_pos

    def get_fake_detection(self):
        if not hasattr(self, "kCountGetFakeDetection"):
            self.count = 0
            self.kCountGetFakeDetection = 5
        self.count += 1
        if self.count >= self.kCountGetFakeDetection:
            self.count = 0
            return True
        return False

    def _gp_step(self, detection_map=None):
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
        if self.fake_ros:
            if self.GP_ground_truth is None:
                self.GP_ground_truth = GaussianProcessGroundTruth(
                    target_num=len(self.other_list),
                    init_other_pose=self.relative_pose.loc[
                        self.start_index].to_numpy(),
                )
            else:
                self.GP_ground_truth.step(other_pose=self.relative_pose.loc[
                    self.curr_index].to_numpy())
        ### Detection part
        if self.GP_detection is None:
            self.GP_detection = GaussianProcessWrapper(num_uav=self.num_uav,
                                                       node_coords=np.zeros(
                                                           (1, 2)),
                                                       id=self.id)
        else:
            for other_index in range(self.num_latent_target):
                # 判断当前能否获得 detection
                observe_point_time = self.curr_time
                if len(detection_map):
                    if other_index in detection_map:
                        # 获取可用 detection
                        self.gp_got_first_detection = True
                        observe_value = np.ones(1, )
                        # 6.0 用作归一化，见 GaussianProcessGroundTruth._pose_max_val
                        observe_point = detection_map[other_index].reshape(
                            -1, 2) / 6.0

                        self.GP_detection.GPs[other_index].add_observed_point(
                            point_pos=add_t(observe_point, observe_point_time),
                            value=observe_value,
                        )
            # 更新 GP 参数
            self.GP_detection.update_GPs()

            # predcitoin
            all_pred, all_std, pred_s = self.GP_detection.update_grids(
                self.curr_time)

            # 反例采集, 每 0.5s
            if self.kFovEffectGP and self.negitive_gather():
                neg_std = self.GP_detection.update_negititve_gps(
                    X=np.zeros((1, 2)),
                    Y=0,
                    time=self.curr_time,
                    mask=self.__get_fov_mask())
                all_std = np.min(np.array([neg_std, all_std]), axis=0)
            else:
                if self.fake_ros and len(self.all_stds_list):
                    all_std = self.all_stds_list[-1]

            if self.fake_ros:
                self.all_means_list.append(all_pred)
                self.all_stds_list.append(all_std)
                self.gps_means_list.append(pred_s)
        return all_std

    def negitive_gather(self):
        if self.curr_time - self.last_negitive_sample_time >= 0.5:
            self.last_negitive_sample_time = self.curr_time
            return True
        return False

    def get_fov_mask(self):
        return self.__get_fov_mask()

    def __get_fov_mask(self):
        '''
        根据当前 FOV 产生 mask
        '''
        grid = self.GP_detection.GPs[0].grid
        conditions = np.apply_along_axis(self.__in_fov, 1, grid)
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

    def command_post_process(self, command):
        """
        后退时,command可能不在FOV范围内, 将command限制在FOV范围内, 避免距离过近
        """
        # this command is in base_link_no_rotation
        if self.fake_fov_range[0] == 0:
            return command

        # self.__in_fov in no_rotation_axis
        command = np.asarray(command)
        horizon_command = command[:2]
        if not self.__in_fov(np.asarray(horizon_command)):
            # 尝试一下仅单独置0
            command[:2] = command[:2] * 0.1

        return command

    def step(self, curr_time, detection_map, ego_heading):
        """
        -----
        Args:
            curr_time: ros传入的当前时刻
            detection_map: key: nth_drone value: position estimation
            ego_heading: 无人机当前 yaw 角， 世界坐标系下
        ----
        ## return:
        action [vx,vy,yaw]
        """

        # GP_step
        all_std = None
        all_std = self._gp_step(detection_map=detection_map)

        action = np.zeros(3)
        action = self.planner.step(self.GP_detection,
                                   detection_map=detection_map,
                                   curr_t=self.curr_time,
                                   ego_heading=self.ego_heading,
                                   std_at_grid=all_std)

        # history
        self.last_ego_heading = self.ego_heading
        self.last_yaw_action = action[-1]

        # metrices
        if self.fake_ros:
            all_pred, _, _ = self.GP_detection.update_grids(self.curr_time)
            JS = self.GP_detection.eval_all_js(
                np.max(self.GP_ground_truth.y_true_lists[-1], axis=-1),
                all_pred)
            _, UNC = self.GP_detection.eval_unc_with_grid(std_at_grid=all_std)
            _, FOV_UNC = self.GP_detection.eval_unc_with_grid(
                high_info_idx=np.array([[]] * self.num_latent_target),
                std_at_grid=all_std)
            self.metrics_obj.step(jsd=JS, unc=UNC, fovunc=FOV_UNC)
        return action

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
