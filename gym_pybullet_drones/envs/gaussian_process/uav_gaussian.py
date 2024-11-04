import rospkg
from scipy.spatial import cKDTree
import numpy as np
import pandas as pd
from sympy import det
import yaml
from copy import deepcopy
import rospy
import scienceplots
import matplotlib.pyplot as plt
from Gaussian_Process.uav_animation import UAVAnimation
from Gaussian_Process.gaussian_process import GaussianProcessGroundTruth
from Gaussian_Process.gaussian_process import GaussianProcessWrapper
from Gaussian_Process.UCB.ucb_replanning import UCBReplanning
from Gaussian_Process.UCB.tsp_base_line import TSPBaseLine
from .metrics import Metrics
from .util.utils import *


def add_t(X, t: float):
    """
    add timestamp for measurement
    """
    return np.concatenate((X, np.zeros((X.shape[0], 1)) + t), axis=1)


class UAV():
    """
    作为未来可视化或分析的接口
    从ros中读取数据,并进行可视化
    """

    def __init__(
        self,
        id=0,
        fake_ros=True,
        visualization=False,
        poses=[],
        detections=None,
        other_pose_local=None,
        mode_simulation=False,
        planner="tsp",
        **kwargs,
    ) -> None:
        """
        # description :
         ---------------
        # param :
         - id: 无人机编号, 为 1 ~ num_uav
         - fake_ros: if true, use ros info from fake ros
         - planner: tsp|ucb|stamp
         ## **kwargs
         - num_uav : 执行任务无人机数量
         - camera_config: 定义 fov 使用的配置文件
         ---------------
        # returns :
         ---------------
        """

        self.id = id
        self.fake_ros = fake_ros  # if true, use ros info from fake ros

        self.__get_config(**kwargs)
        self.visualization = visualization
        self.poses = poses  # all pose data from fake gps
        self.detections = detections  # all detections from fake gps
        self.other_pose_local = other_pose_local
        self.num_uav = kwargs.get("num_uav", len(poses))  # 无人机总数量
        self.other_list = list(range(1, self.id)) + list(
            range(self.id + 1, self.num_uav + 1))

        try:
            self.pose: pd.DataFrame = self.poses[self.id - 1]
            self.detection: pd.DataFrame = self.detections[self.id - 1]
        except IndexError as e:
            print(e)

        self.start_index = None  # 记录从 fake_ros 传递的第一个索引
        self.start_time = None  # 记录从 fake_ros 传递的第一个索引
        self.curr_index = 0  # 当前 fake_ros 传递的索引
        self.index_history = []
        self.curr_time = 0.0  # 从 vision_pose 话题中更改当前时间
        if self.fake_ros:
            self.__init_relative_pose()  # 初始化 相对位置 部分
        self.__last_time = None  # 用作从 pose_time 获取 detection_time
        self.__detection_index = None  # 当前观测的 index
        self.detection_2_curr_index = {
        }  # a map from self.curr_index to detection_index
        # for detection 的目标匹配
        self.num_latent_target = len(self.other_list)  # 潜在目标数量
        self.last_detection_target_map = None
        # gaussian_process_part
        self.GP_ground_truth = None  # GP ground truth 在step()中完成初始化
        self.GP_detection = None  # 由 detection 驱动的 Gaussian Process
        self.all_means_list = []
        self.all_stds_list = []
        self.gps_means_list = []
        self.observed_points_list = []
        self.last_all_std = None
        self.metrics_obj = Metrics(**kwargs)
        self.gp_got_first_detection = False
        # planner select
        self.planner_debug_msg = True  # 此处打开 planner 的 debug msg
        if planner == "ucb":
            self.planner = UCBReplanning()
        elif planner == "tsp":
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
        # animation part
        self.uav_animation = UAVAnimation(**kwargs)
        if self.visualization:
            pass
            # TODO: ROS中实时的可视化？

        # 是否仅进行 analyse
        self.mode_simulation = mode_simulation

    def __get_config(self, **kwargs):
        '''
        set self.fake_fov_range
        '''
        camera_config = kwargs.get("camera_config", "single")
        config_file = "/home/lih/catkin_ws/src/vswarm_cp/src/decision/Gaussian_Process/config/flocking_node/{}.yaml".format(
            camera_config)
        with open(config_file) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        self.fake_fov_range = np.array(
            [config['fov'][1]['negative'], config['fov'][0]['positive']])

        print("set fake fov range: ", self.fake_fov_range)

    def __update_from_fake_ros(self, fake_ros_msg=None):
        """
        update self.curr_index and self.curr_time
        """
        if fake_ros_msg is None:
            print("error: no fake ros msg")
        if self.start_index is None:
            self.start_index = fake_ros_msg
            self.start_time = self.pose.loc[self.start_index, "Time"]
        self.curr_index = fake_ros_msg
        self.index_history.append(self.curr_index)
        self.curr_time = self.pose.loc[self.curr_index, "Time"]

        if not self.mode_simulation:
            self.ego_heading = yaw_from_quaternion(
                x=self.pose.loc[self.curr_index, "pose.orientation.x"],
                y=self.pose.loc[self.curr_index, "pose.orientation.y"],
                z=self.pose.loc[self.curr_index, "pose.orientation.z"],
                w=self.pose.loc[self.curr_index, "pose.orientation.w"])
            # 此处由于 vision pose 是从 gazebo 中获取，与 world 坐标系存在偏转
            self.ego_heading = normalize_radians(self.ego_heading - np.pi / 2)

    def __update_from_real_ros(self, **kwargs):
        if self.start_index is None:
            self.start_index = 0
            self.start_time = kwargs.get("curr_time", None)

        self.curr_index += 1
        self.curr_time = kwargs.get("curr_time", None)

    def __init_relative_pose(self):
        """
        使用 self.poses 其中包含所有无人机的 pose, 计算 无人机 相对位置
        """
        columns = [
            "uav{}_to_ego_pos.{}".format(id, item) for id in self.other_list
            for item in ["x", "y"]
        ]
        self.relative_pose = pd.DataFrame(data=None, columns=columns)

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

    def __update_detection(self, time: float = None, **kwargs):
        """
        # description : 在 step()中调用，更新 __detection_index
         ---------------
        # param :
         - time: 当前时间

        ## kwargs:
        - detections
         ---------------
        # returns : detection_index,detection_target_map
         ---------------
        """
        if self.fake_ros and not self.mode_simulation:
            detection_index = self.__getDetectionFromPoseTime(time)
            # 存储该 map
            self.detection_2_curr_index[self.curr_index] = detection_index
            observe_points = []
            # 获取 pandas 数据
            if detection_index is not None:
                for other_idx in range(self.num_latent_target):
                    keys = [
                        f"bbox{other_idx}.center.position.x",
                        f"bbox{other_idx}.center.position.y",
                    ]
                    observe_pd = self.detection.loc[detection_index, keys]
                    if all(pd.isna(observe_pd)):
                        pass
                    else:
                        observe_points.append(observe_pd.to_numpy())
        elif self.fake_ros and self.mode_simulation and self.get_fake_detection(
        ):
            observe_points = []
            detection_index = None
            for other_id in self.other_list:
                observe_points.append(np.array([
                    self.relative_pose.loc[self.curr_index]["uav{}_to_ego_pos.x".format(other_id)], \
                    self.relative_pose.loc[self.curr_index]["uav{}_to_ego_pos.y".format(other_id)]
                ]))
        else:
            detection_index = None
            detections = kwargs.get("detections", [])
            observe_points = []
            for detection in detections:
                if np.any(np.isnan(detection)):
                    pass
                else:
                    observe_points.append(detection[:2])
            observe_points = [detection[:2] for detection in detections]

        # last_detection_target_map init
        if len(observe_points) and self.last_detection_target_map is None:
            self.last_detection_target_map = {}
            for key in range(self.num_latent_target):
                self.last_detection_target_map[key] = np.full((1, 2), np.inf)
            # 根据第一个可用的 observe_points 的顺序初始化 self.last_detection_target_map
            for index, ob_point in enumerate(observe_points):
                self.last_detection_target_map[index] = ob_point

        # 使用 detection_target_map() 维护一个由 target_id 到 detection 的映射
        detection_target_map = {}

        def getTargetKeyForDetction(ob_point: np.ndarray):
            dists = [np.inf] * self.num_latent_target
            for key, value in self.last_detection_target_map.items():
                dists[key] = np.linalg.norm(
                    ob_point.reshape(-1, 2) - value.reshape(-1, 2))
            return np.array(dists).argmin()

        def updateLastDetectionMap(detection_target_map):
            # 不减少key
            if self.last_detection_target_map is None:
                return
            for key, _ in self.last_detection_target_map.items():
                if key in detection_target_map:
                    self.last_detection_target_map[key] = detection_target_map[
                        key]

        for ob_point in observe_points:
            key = getTargetKeyForDetction(ob_point)
            # 避免重复
            if key in detection_target_map:
                key += 1
            detection_target_map[key] = ob_point
        # last_map update
        updateLastDetectionMap(detection_target_map)
        return detection_index, detection_target_map

    def get_fake_detection(self):
        if not hasattr(self, "kCountGetFakeDetection"):
            self.count = 0
            self.kCountGetFakeDetection = 5
        self.count += 1
        if self.count >= self.kCountGetFakeDetection:
            self.count = 0
            return True
        return False

    def __getDetectionFromPoseTime(self, pose_time: float, reset=False):
        """
        给定 pose_time 返回一个 detection_time 大于等于 pose_time 的 detection , 且前一个 pose_time 小于 detection_time
        如无 detection, 返回 None

        Note: 由于该函数需持续更改类的对象, 因此在再次循环调用该函数之前, 需调用该函数 reset 方法
        """
        index = self.detection.loc[lambda df: df["Time"] <= pose_time].index
        if len(index) == 0:
            return 0
        return index[-1]
        if self.__last_time is None:
            self.__last_time = 0.0
            self.__detection_index = (
                self.detection.loc[lambda df: df["Time"] >= pose_time].index[0]
                - 1)
            if self.__detection_index < 0:
                self.__detection_index = 0
        if self.__detection_index >= len(self.detection):
            return

        # 当前 index 是否可以发送
        curr_detection_time = self.detection.loc[self.__detection_index,
                                                 "Time"]
        return_index = None
        if self.__last_time < curr_detection_time and pose_time >= curr_detection_time:
            return_index = self.__detection_index
            self.__detection_index = self.detection.loc[
                lambda df: df["Time"] >= pose_time].index[0]
        self.__last_time = pose_time
        return return_index

    def _gp_step(self, detection_map=None):
        """
        # description : 进行 gp_ground_truth 和 gp_detection 的step
         ---------------
        # param :
         - detection_map: { 0:pos_target_0, 1:pos_target_1 ... },if no detection, the value is none
         ---------------
        # returns :
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

    def __fake_fov_restrict(self, detection_map: dict = None):
        """
        根据当前 fake_fov 的限制,更新 detection_map
        """
        # if exist detections
        if len(detection_map):
            for other_index in range(self.num_latent_target):
                if other_index in detection_map:
                    observe_point = detection_map[other_index] / 6.0
                    if not self.__in_fov(observe_point):
                        # delete 改 detection
                        del detection_map[other_index]

    def get_fov_mask(self):
        return self.__get_fov_mask()

    def __get_fov_mask(self):
        '''
        根据当前 FOV 产生 mask
        '''
        grid = self.GP_detection.GPs[0].grid
        conditions = np.apply_along_axis(self.__in_fov, 1, grid)
        return conditions.astype(int)

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

    def __fake_ros_apply_action(self, action: np.ndarray):
        """
        将 planner 产生的运动指令 应用于 fake_fov
        """
        if action is None:
            return

        if self.fake_ros and self.mode_simulation:
            # action 为增量式
            self.ego_heading = normalize_radians(self.ego_heading + action[-1])
        if self.fake_ros:
            self.__update_fake_fov()

    def __update_fake_fov(self):
        self.fake_fov = self.fake_fov_range + self.ego_heading
        self.fake_fov = np.array(
            [normalize_radians(heading) for heading in self.fake_fov])
        self.fake_fov_list.append(self.fake_fov)

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
            # # 求方向向量在两个方向上的投影
            # fov_angle = self.fake_fov + np.pi / 2
            # fov_vector = [
            #     np.array([np.cos(fov), np.sin(fov)]) for fov in fov_angle
            # ]
            # # 判断投影的长度关系
            # projection = lambda a, b: np.dot(a, b) / np.dot(b, b)  # a 在 b上的投影
            # if projection(horizon_command,
            #               fov_vector[0]) > projection(horizon_command,
            #                                           fov_vector[1]):
            #     projected_command = projection(horizon_command,
            #                                    fov_vector[0]) * horizon_command
            # else:
            #     projected_command = projection(horizon_command,
            #                                    fov_vector[1]) * horizon_command
            # command[:2] = projected_command
            # rospy.loginfo(f"UAV {self.id} comnad not in fov")

        return command

    def step(self, fake_ros_msg=None, **kwargs):
        """
        从 fake_ros_msg 或真实 ros 中接受一条信息
        并对 ros 的信息进行处理
        -----
        ## kwargs:
        curr_time: ros传入的当前时刻
        detection: 处理完成后的 detection
        ego_heading: 无人机当前 yaw 角
        ----
        ## return:
        action [vx,vy,yaw]
        """
        self.warning_error_msg = ""
        if self.fake_ros:
            self.__update_from_fake_ros(fake_ros_msg)
            self.__update_relative_pos()  # 仅在 fake_ros 运行，供绘图使用
        else:
            self.__update_from_real_ros(**kwargs)
            self.ego_heading = kwargs.get("ego_heading", 0.0)

        _, detection_map = self.__update_detection(self.curr_time, **kwargs)
        # if have fake fov restrict
        if self.fake_ros and self.mode_simulation:
            self.__fake_fov_restrict(detection_map)

        # GP_step
        all_std = None
        all_std = self._gp_step(detection_map=detection_map)

        action = np.zeros(3)
        # 当上一个控制指令被执行后，才执行下一个控制指令
        if not self.fake_ros or self.mode_simulation:
            # if abs(self.ego_heading - self.last_ego_heading
            #        ) < self.kApproveNextActionHeadingThreshold:
            action = self.planner.step(self.GP_detection,
                                       detection_map=detection_map,
                                       curr_t=self.curr_time,
                                       ego_heading=self.ego_heading,
                                       std_at_grid=all_std)
            self.warning_error_msg += self.planner.warning_error_msg
        self.__fake_ros_apply_action(action)

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

    def set_plot(self):
        self.__setPlot()

    def __setPlot(self, grid=False):
        if grid:
            plt.style.use(['science', 'ieee', 'no-latex', 'grid'])
        else:
            plt.style.use(['science', 'ieee', 'no-latex'])
        plt.rcParams["axes.labelweight"] = "bold"
        plt.rcParams["axes.titleweight"] = "bold"
        plt.rcParams["figure.titleweight"] = "bold"
        return
        plt.style.use("default")
        # plt.rcParams["figure.figsize"] = [10, 10]
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["mathtext.fontset"] = "dejavuserif"
        plt.rcParams["font.size"] = 12.0
        plt.rcParams["figure.facecolor"] = "#ffffff"
        # plt.rcParams[ 'font.family'] = 'Roboto'
        # plt.rcParams['font.weight'] = 'bold'
        plt.rcParams["xtick.color"] = "#01071f"
        plt.rcParams["xtick.minor.visible"] = True
        plt.rcParams["ytick.minor.visible"] = True
        plt.rcParams["xtick.labelsize"] = 10
        plt.rcParams["ytick.labelsize"] = 10
        plt.rcParams["ytick.color"] = "#01071f"
        plt.rcParams["axes.labelcolor"] = "#000000"
        plt.rcParams["text.color"] = "#000000"
        plt.rcParams["axes.labelcolor"] = "#000000"
        plt.rcParams["grid.color"] = "#f0f1f5"
        plt.rcParams["axes.labelsize"] = 10
        plt.rcParams["axes.titlesize"] = 10
        plt.rcParams["axes.labelweight"] = "bold"
        plt.rcParams["axes.titleweight"] = "bold"
        plt.rcParams["figure.titlesize"] = 24.0
        plt.rcParams["figure.titleweight"] = "bold"
        plt.rcParams["legend.markerscale"] = 1.0
        plt.rcParams["legend.fontsize"] = 8.0
        plt.rcParams["legend.framealpha"] = 0.5

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
    uav = UAV(1, visualization=True)
