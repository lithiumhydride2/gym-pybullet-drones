import numpy as np
import torch
from gym_pybullet_drones.envs.gaussian_process.gaussian_process import GaussianProcessGroundTruth
from gym_pybullet_drones.envs.gaussian_process.gaussian_process import GaussianProcessWrapper
from gym_pybullet_drones.envs.gaussian_process.UCB.tsp_base_line import TSPBaseLine
from gym_pybullet_drones.utils.utils import *
from scipy.spatial.transform import Rotation as R
from gym_pybullet_drones.envs.IPPArguments import IPPArg


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
        planner=None,
        node_coords=None,
    ) -> None:
        """
        ## description :
         ---------------
        Args:
            fov_range: 以两个角度表示的 fov_range
            nth_drone: uav id
            num_drone: 集群中无人机的数量
            planner: 规划器种类，可选 tsp, 若无需基于规则规划期，置为 None
            node_coords: 采样得到的节点坐标
        """

        self.id = nth_drone
        self.num_drone = num_drone
        self.other_list = list(set(range(num_drone)) - set([nth_drone]))

        self.num_latent_target = len(self.other_list)  # 潜在目标数量

        ########### gaussian_process_part ############
        self.GP_ground_truth = GaussianProcessGroundTruth(
            target_num=self.num_latent_target,
            init_other_pose=np.zeros(
                (2 * self.num_latent_target, )))  # x,y init as 0.0

        self.GP_detection = GaussianProcessWrapper(num_uav=self.num_drone,
                                                   other_list=self.other_list,
                                                   id=self.id)
        self.node_coords = node_coords
        self.fov_range = fov_range
        self.fov_masks = self._get_FOV_masks_of_node()
        self.cache: dict[str, np.ndarray] = {}  # cache
        ################### planner select ##############

        if planner == "tsp":
            self.planner = TSPBaseLine(
                num_latent_target=self.num_latent_target,
                fake_fov_range=fov_range)

        self.ego_heading = 0.0  # 自身朝向状态量
        self.last_ego_heading = 0.0  # 上一个朝向状态量
        self.last_yaw_action = 0.0  # 上一个朝向输出

        # avgpool
        self.avgpool = torch.nn.AvgPool1d(kernel_size=IPPArg.history_stride,
                                          stride=IPPArg.history_stride,
                                          ceil_mode=True)
        self.last_time = 0.0

    def reset(self, nth_drone):
        '''
        node_coords 无需改变
        '''
        self.id = nth_drone
        self.other_list = list(set(range(self.num_drone)) - set([nth_drone]))
        self.num_latent_target = len(self.other_list)  # 潜在目标数量

        self.cache = {}
        self.last_time = 0.0
        self.ego_heading = 0.0  # 自身朝向状态量
        self.last_ego_heading = 0.0  # 上一个朝向状态量
        self.last_yaw_action = 0.0  # 上一个朝向输出

        self.GP_ground_truth.reset(
            init_other_pose=np.zeros((2 * self.num_latent_target, )))
        self.GP_detection.reset(id=nth_drone, other_list=self.other_list)

    def _get_FOV_masks_of_node(self):
        # node_coords 在世界坐标系下,对于每个 node 生成一个 FOV_mask
        FOV_masks = []
        for node in self.node_coords:
            node = np.hstack((node, [0]))
            FOV_vector = np.asarray([
                np.dot(
                    R.from_euler('z', angle,
                                 degrees=False).as_matrix().reshape(3, 3),
                    node) for angle in self.fov_range
            ])
            FOV_masks.append(self.__get_fov_mask(FOV_vector))

        return np.asarray(FOV_masks)

    def _gp_step(self,
                 detection_map: dict[int, np.ndarray] = None,
                 other_pose=None,
                 ego_heading=None,
                 time=None):
        """
        ### description : 进行 gp_ground_truth 和 gp_detection 的step
         ---------------
        ### param :
         - detection_map: { 0:pos_target_0, 1:pos_target_1 ... },if no detection, the value is none
         - node_coords: yaw角形式是
         ---------------
        ### returns :
        - all_std
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
        # update grid
        all_pred, all_std, self.cache["preds"], self.cache[
            "stds"] = self.GP_detection.update_grids(time)
        # update node feature
        node_feature = self.update_node_feature()
        # node_feature = self.GP_detection.update_node_feature(time, node_coords)
        self.cache["all_std"] = all_std
        self.cache["all_pred"] = all_pred
        return all_std, node_feature

    def get_yaw_feature(self, gp_mean: np.ndarray):
        '''
        对于每个目标，提取其在所有 node_coords 内的 feature
        '''
        yaw_feature = np.zeros((self.node_coords.shape[0], 2))
        grid_size = self.GP_detection.GPs[0].grid_size

        for id, mask in enumerate(self.fov_masks):
            if np.max(gp_mean[mask.astype(bool)]) > IPPArg.EXIST_THRESHOLD:
                max_row, max_col = np.unravel_index(gp_mean.argmax(),
                                                    (grid_size, grid_size))
                yaw = np.asarray(
                    [max_row - grid_size / 2, max_col - grid_size / 2])
                yaw_feature[id] = yaw / np.linalg.norm(yaw)

        return yaw_feature

    def update_node_feature(self):
        '''
        通过 FOV mask 的形式提取 node_feature

        依赖 self.cache["preds"], self.cache["stds"]
        '''
        # 获取预测的特征
        _, _, predict_preds, predict_stds = self.GP_detection.update_grids(
            self.curr_time + IPPArg.PREDICT_FEATURE_TIME, predict_future=True)
        node_feature = []
        pred: np.ndarray

        def get_feature(grid, compare=np.min):
            feature = []
            for mask in self.fov_masks:
                feature.append(compare(grid[mask.astype(bool)]))
            return np.asarray(feature)

        for pred, std, predict_pred, predict_std in zip(
                self.cache["preds"], self.cache["stds"], predict_preds,
                predict_stds):
            node_feature += [
                np.hstack((get_feature(pred, np.max).reshape(-1, 1),
                           get_feature(std, np.min).reshape(-1, 1),
                           get_feature(predict_pred, np.max).reshape(-1, 1),
                           get_feature(predict_std, np.min).reshape(-1, 1),
                           self.get_yaw_feature(pred)))
            ]
        node_feature = np.asarray(node_feature)  #(target,node,feature(6))
        node_feature = node_feature.transpose(1, 0, 2).reshape(
            self.node_coords.shape[0], -1)  #(node,target * feature)
        return node_feature

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

    def step(self, curr_time, detection_map, ego_heading, relative_pose):
        """
        Args:
            curr_time: ros传入的当前时刻
            detection_map: key: nth_drone value: position estimation, 需要坐标系下直接计算的 相对位置
            ego_heading: 无人机当前 yaw 角， 世界坐标系下
            fov_vector: 无人机当前 fov, 由两向量组成，世界坐标系下
            relative_pose: 其他无人机的真实位置,需不包含与自身的相对位置
            node_coords: 节点坐标，用于评估节点特征
        
        Return:
            node_feature
        """
        self.curr_time = curr_time
        self.ego_heading = ego_heading

        # GP_step
        all_std, node_feature = self._gp_step(detection_map=detection_map,
                                              other_pose=relative_pose,
                                              ego_heading=ego_heading,
                                              time=curr_time)
        # node_inputs 为 node_coords 与 node_feature 的结合
        node_inputs = np.concatenate((self.node_coords, node_feature), axis=1)
        history_pool_inputs, dt_pool_inputs = self.avg_pool_node_inputs(
            node_inputs)

        self.last_time = curr_time
        return {
            "node_inputs": history_pool_inputs,
            "dt_pool_inputs": dt_pool_inputs
        }

    def avg_pool_node_inputs(self, node_inputs):
        node_inputs = torch.Tensor(node_inputs).unsqueeze(0)
        # env.reset()时， 会重新 init UAVGaussian, 因此通过判断是否有 node_inputs_history
        if not hasattr(self, "node_inputs_history"):
            self.node_inputs_history = node_inputs.repeat(
                IPPArg.history_size, 1,
                1)  #(history_size, node_num, feature_dim)
            self.dt_history = torch.zeros((IPPArg.history_size, 1))
        else:
            # 添加新的历史
            self.node_inputs_history = torch.cat(
                (self.node_inputs_history[1:], node_inputs.clone()), dim=0)
            self.dt_history -= (self.curr_time -
                                self.last_time) / IPPArg.dt_normlization
            self.dt_history = torch.cat(
                (self.dt_history[1:], torch.Tensor([[.0]])), dim=0)

        ## avgpool 在最后一个维度上进行
        history_pool_node_inputs: torch.Tensor = self.avgpool(
            self.node_inputs_history.permute(1, 2, 0)).permute(
                2, 0, 1)  #(history_size, node_num, feature_dim)
        history_pool_dt: torch.Tensor = self.avgpool(
            self.dt_history.permute(1, 0)).permute(1, 0)
        return history_pool_node_inputs.numpy(), history_pool_dt.numpy()

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

        return self.last_yaw_action


if __name__ == "__main__":
    print("this is uav gaussian")
