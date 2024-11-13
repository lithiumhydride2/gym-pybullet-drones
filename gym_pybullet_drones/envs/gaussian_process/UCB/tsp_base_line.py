import time
import concurrent.futures
import numpy as np
from ..gaussian_process import GaussianProcessWrapper
from .motion_primitive import MotionPrimitive
from .uav_detection_sim import UavDetectionSim
from python_tsp.exact import solve_tsp_dynamic_programming as tsp_solver
from gym_pybullet_drones.utils.utils import *


def multi_thread_func(detection_sims: "list[UavDetectionSim]", ego_heading,
                      motion_primitive: MotionPrimitive,
                      gp_wrapper: GaussianProcessWrapper, index):
    detection_sim: UavDetectionSim = detection_sims[index]
    detection_sim.reset(ego_heading, gp_wrapper)

    curr_reward = detection_sim.step(motion_primitive.headings[index],
                                     motion_primitive.time_span)
    return curr_reward


class TSPBaseLine:

    def __init__(self,
                 num_latent_target=1,
                 fake_fov_range=[0, 0],
                 **kwargs) -> None:
        '''
        kwargs:
            fake_fov_range: fov的边界，以vector形式展现
        '''
        self.distance_matrix = None
        self.observed_target = {}  # key: 潜在目标索引, val: tuple(置信度，固定坐标系中角度)
        self.last_solve_tsp_observed_target = {}
        self.num_latent_target = num_latent_target
        self.ego_heading = 0.0  # 无人机 在与无人机 heading 解耦 坐标系中的朝向，是不是可以说是在 全局坐标系下对自身 heading 的估计
        self.ego_key = self.num_latent_target  # 自身键值为 最大 target_index + 1
        self.motion_primitive = MotionPrimitive()
        self.last_solve_tsp_t = 0.0
        self.curr_t = 0.0
        self.action_none = np.zeros(3, )
        self.last_action = np.zeros(3, )
        self.kUseDetectionMap = False
        self.fake_fov_range = fake_fov_range
        ### 常量
        self.kTargetBeliefThreshold = 0.5  # 若对某目标置信度低于该阈值，加入待访问节点
        self.kTargetExistBeliefThreshold = 0.1  # 若置信度低于此值，则认为环境中目标不存在
        self.kEnableExploartion = kwargs.get("enable_exploration", False)

        self.selected_motion_primitive = None
        self.last_motion_primitive = None
        self.motion_index = None
        self.warning_error_msg = ""

        # detection sim
        self.detection_sims = [
            UavDetectionSim(fake_fov_range=self.fake_fov_range,
                            num_uav=self.num_latent_target + 1)
            for _ in range(len(self.motion_primitive.headings))
        ]

    def need_replanning(self, gp_wrapper: 'GaussianProcessWrapper'):
        curr_belif = gp_wrapper.get_observed_points(
            kTargetExistBeliefThreshold=0.8)
        last_solve_belief = gp_wrapper.get_observed_points(
            kTargetExistBeliefThreshold=0.5, time=self.last_solve_tsp_t)
        # TODO: 第二个条件考虑到 replanning 的及时性
        # 第一个条件当前考虑到 确保获取 detection 之后再执行重规划
        if len(curr_belif) >= self.num_latent_target and len(curr_belif) > len(
                last_solve_belief):
            # 清空motion_index
            self.motion_index = 0
            return True
        return False

    def step(self,
             gp_wrapper: 'GaussianProcessWrapper' = None,
             curr_t=0.0,
             ego_heading=0.0,
             std_at_grid=None):
        """
        ## description :
        - 使用 TSP 方法处理 gp_wrapper
         --------------- 
        ## param :
         - gp_wrapper:  GaussianProcessWrapper
         - curr_t: time_stamp of decision
         - ego_heading: ego heading in world axis
         - std_at_grid: obs of as "all_std"
         --------------- 
        ## returns :
        - 运动基元的索引、该运动基元
        """

        detection_map = {}  # TSP 中起到作用的是 self.observed_target

        self.curr_t = curr_t
        self.ego_heading = ego_heading
        self.std_at_grid = std_at_grid

        # check if need to resolve tsp
        keys_to_visit = 0
        # 更新 self.observed_target
        if self.need_replanning(gp_wrapper) or self.motion_index is None \
            or self.motion_index == self.motion_primitive.num_segment:
            self.__update_detection(gp_wrapper)
            self.last_solve_tsp_t = self.curr_t
            self.last_solve_tsp_observed_target = self.observed_target
            keys_to_visit = self.__update_tsp_problem()
            heading = self.__tsp_post_process(keys_to_visit,
                                              gp_wrapper=gp_wrapper)

        return self.__get_action_no_wait()

    def __update_detection(self, gp_wrapper: 'GaussianProcessWrapper'):
        '''
        根据当前观测, 更新 self.ovserved_target
        提取 belief 中阈值高于 kTargetExistBeliefThreshold 的对象
        '''
        # reset
        self.observed_target = {}

        if not hasattr(self, "grid_size"):
            self.grid_size = gp_wrapper.GPs[0].grid_size

        for index, gp in enumerate(gp_wrapper.GPs):
            if gp.y_pred_at_grid is None:
                continue
            else:
                y_pred_grid = gp.y_pred_at_grid.reshape(
                    self.grid_size, self.grid_size)
                max_row, max_col = np.unravel_index(y_pred_grid.argmax(),
                                                    y_pred_grid.shape)
                # 原点为 self.grid_size / 2, from row,col to [x,y]
                point_vector = np.array([
                    max_row - self.grid_size / 2, max_col - self.grid_size / 2
                ])
                if np.max(y_pred_grid) > self.kTargetExistBeliefThreshold:
                    self.observed_target[index] = (np.max(y_pred_grid),
                                                   point_heading(point_vector))

    def __update_tsp_problem(self):
        '''
        ## Description:
        1. 根据 self.__update_detection 生成待解决的图
        2. 求解 tsp 问题
        ## Return:
        - 1. 更新 self.distance_matrix
        - 2. 将置信度阈值小于一定值的目标更新加入 node_to_build_graph
        '''
        # 清零
        self.distance_matrix = np.zeros(
            (self.num_latent_target, self.num_latent_target))

        # 将置信度低于阈值 的项目 加入待访问图中
        node_to_build_graph = []  # 利用 key(潜在目标index), heading 构建 graph

        for key, val in self.observed_target.items():
            belief, heading = val
            if belief <= self.kTargetBeliefThreshold:
                node_to_build_graph.append((key, heading))

        return self.__solve_tsp(node_to_build_graph)

    def __solve_tsp(self, node_to_build_graph: 'list[tuple]'):
        '''
        由于当前 tsp 问题的特殊性, 需要把自身当前 heading 作为一个计算距离的项目
        '''
        ego_node = (self.ego_key, self.ego_heading)  # 此处 ego_key 为最大 key
        node_to_build_graph.append(ego_node)
        n = len(node_to_build_graph)
        # 由于角度非对称性，因此两次遍历计算
        self.distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in set(range(n)) - set([i]):
                self.distance_matrix[i][j] = np.abs(
                    normalize_radians(node_to_build_graph[i][1] -
                                      node_to_build_graph[j][1]))
        permutation, _ = tsp_solver(self.distance_matrix)
        keys_to_visit = [
            node_to_build_graph[index][0] for index in permutation
        ]
        return keys_to_visit

    def __tsp_post_process(self, keys_to_visit: list, gp_wrapper=None):
        '''
        对 tsp 求解结果进行后处理 更新self.selected_motion_primitive,返回新的偏转角度
        ---
        keys_to_visit : tsp 求解将要访问的节点
        gp_wrapper : 高斯过程总类
        '''

        # 更新 self.selected_motion_primitive

        assert keys_to_visit is not None
        if len(keys_to_visit) == 1 and keys_to_visit[0] == self.ego_key:
            # 当前 fov 可获得所有观测，或无法获得观测
            # TODO(lih): 在此处加入探索
            if len(self.observed_target) == self.num_latent_target:
                primitive_idx = 0  # 无需偏转
            else:
                primitive_idx = self.__exploration(gp_wrapper)
            # elif len(self.observed_target) and len(self.observed_target) < min(
            #         self.num_latent_target, 4):
            #     # exploration here\
            #     primitive_idx = self.__exploration(gp_wrapper)
            # else:
            #     primitive_idx = self.motion_primitive.num_primitive - 1  # 最大偏转
        else:
            # 在已经对所有潜在目标进行观测时，选择最合适的运动基元
            for key in keys_to_visit:
                if key != self.ego_key:
                    heading_err = normalize_radians(
                        self.observed_target[key][1] - self.ego_heading)
                    primitive_idx = np.argmin(
                        np.abs(self.motion_primitive.headings - heading_err))

        self.selected_motion_primitive = self.motion_primitive.motion_primitive[
            primitive_idx]
        return np.rad2deg(self.motion_primitive.headings[primitive_idx])

    def __exploration(self, gp_wrapper: 'GaussianProcessWrapper'):
        '''
        评估每个运动基元对 std 差值

        1. reward 定义为当前 unc 与应用 运动基元后 unc 之差
        2. 对于每个运动基元，估计应用该基元后的 unc
        '''
        if not self.kEnableExploartion:
            return np.random.randint(self.motion_primitive.num_primitive)
        try:
            curr_neg_unc_list, curr_neg_unc = gp_wrapper.eval_unc_with_grid(
                std_at_grid=self.std_at_grid)  # 返回一个 list
        except:
            curr_neg_unc = 1.1

        reward = []
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(multi_thread_func, self.detection_sims,
                                self.ego_heading, self.motion_primitive,
                                gp_wrapper, index)
                for index in range(len(self.motion_primitive.headings))
            ]

            for future in concurrent.futures.as_completed(futures):
                reward.append(future.result())

        reward = np.array(reward)
        end_time = time.time()
        print("Time cost : {} and reward {}".format(end_time - start_time,
                                                    reward))

        # 探索无法获取收益
        if np.min(reward) > curr_neg_unc:
            return 0
        else:
            return np.argmin(reward)

    def __get_action_no_wait(self):
        '''
        以 yaw (rad) 的形式返回 action
        '''
        if self.motion_index is None:
            self.motion_index = 0
        elif self.motion_index == self.motion_primitive.num_segment:
            self.motion_index = 0

        action = self.action_none
        tim_diff = self.curr_t - self.last_solve_tsp_t
        if tim_diff >= self.motion_primitive.time_range[self.motion_index]:
            self.motion_index += 1
            action = self.selected_motion_primitive[min(
                self.motion_index, self.motion_primitive.num_segment - 1)]
        return action
