from .FlockingAviary import *
from .IPPArguments import IPPArg
from ..utils.graph_controller import GraphController
from ..utils.utils import circle_angle_diff
from gymnasium.spaces import Box, Dict, Discrete


class FlockingAviaryIPP(FlockingAviary):
    '''
    为 Flocking Aviary 添加 IPP 建模相关内容
    '''

    def __init__(self,
                 drone_model=DroneModel.CF2X,
                 num_drones=1,
                 control_by_RL_mask=None,
                 neighbourhood_radius=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics=Physics.PYB,
                 pyb_freq=240,
                 flocking_freq_hz=10,
                 decision_freq_hz=5,
                 ctrl_freq=240,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 use_reynolds=True,
                 default_flight_height=1,
                 output_folder='results',
                 fov_config=FOVType.SINGLE,
                 obs=ObservationType.GAUSSIAN,
                 act=ActionType.YAW,
                 random_point=True):
        assert act == ActionType.IPP_YAW
        super().__init__(drone_model, num_drones, control_by_RL_mask,
                         neighbourhood_radius, initial_xyzs, initial_rpys,
                         physics, pyb_freq, flocking_freq_hz, decision_freq_hz,
                         ctrl_freq, gui, record, obstacles, user_debug_gui,
                         use_reynolds, default_flight_height, output_folder,
                         fov_config, obs, act, random_point)
        # IPP 属性
        self.IPPEnvs: dict[int, IPPenv] = {}
        for nth in self.control_by_RL_ID:
            self.IPPEnvs[nth] = IPPenv(yaw_start=self._computeHeading(nth)[:2],
                                       act_type=act)

    def plot_online(self):
        super().plot_online()
        # 绘制图的采样
        if self.USER_DEBUG:
            for nth in self.control_by_RL_ID:
                node_coords = self.IPPEnvs[nth].node_coords
                self.plot_online_stuff[f"gp_pred_{nth}"][1].scatter(
                    node_coords[:, 0], node_coords[:, 1], c='orchid')
                plt.pause(1e-10)

    def reset(self, seed=None, options=None):
        '''
        reset 的最终作用为获取 initial_obs, initial_info
        '''

        #### 重新初始化 control_by_RL_MASK
        if hasattr(self, "RANDOM_RL_MASK") and self.RANDOM_RL_MASK:
            mask = np.zeros((self.NUM_DRONES, ))
            mask[np.random.randint(0, self.NUM_DRONES)] = 1
            self.control_by_RL_mask = mask.astype(bool)

            self.control_by_RL_ID = np.array(
                list(range(0, self.NUM_DRONES)),
                dtype=np.int8)[self.control_by_RL_mask]

        # 重新初始化 control_by_RL_ID
        self.decisions = {}
        self.IPPEnvs = {}
        for nth in self.control_by_RL_ID:
            self.IPPEnvs[nth] = IPPenv(yaw_start=self._computeHeading(nth)[:2],
                                       act_type=self.ACT_TYPE)
            # 使用 IPPEnvs 的采样初始化 self.decision
            self.decisions[nth] = decision(
                fov_range=self.fov_range,
                nth_drone=nth,
                num_drone=self.NUM_DRONES,
                node_coords=self.IPPEnvs[nth].node_coords)
        return super().reset(seed, options)

    def _actionSpace(self):
        if self.ACT_TYPE == ActionType.IPP_YAW:
            return Discrete(IPPArg.sample_num)

    def _observationSpace(self):
        if self.OBS_TYPE == ObservationType.IPP:
            return Dict({
                "node_inputs":
                Box(
                    low=0.,
                    high=1.,
                    shape=(IPPArg.sample_num, 2 + (self.NUM_DRONES - 1) *
                           2),  # node_coord and feature (target * (mean,std))
                    dtype=np.float32),
                "edge_inputs":
                Box(low=0,
                    high=IPPArg.sample_num - 1,
                    shape=(IPPArg.sample_num, IPPArg.k_size),
                    dtype=np.int32),
                "curr_index":
                Box(low=0,
                    high=IPPArg.sample_num - 1,
                    shape=(1, 1),
                    dtype=np.int64),
                "graph_pos_encoding":
                Box(low=0.,
                    high=1.,
                    shape=(IPPArg.sample_num, IPPArg.num_eigen_value),
                    dtype=np.float32)
            })

    def _computeObs(self):
        '''
        Return the current observation of the environment.
        '''
        # 获得增广图形式的观测
        assert self.step_counter % self.DECISION_PER_PYB == 0
        if self.OBS_TYPE == ObservationType.IPP:
            # 按照固定的时间频率，获得包含 node_feature 的观测
            obs = {}
            adjacency_Mat = self._computeAdjacencyMatFOV()
            relative_position = self._relative_position
            for nth in self.control_by_RL_ID:
                #TODO 获得观测时，需要更新 IPP_env

                # mask 用于获取真实相对位置
                other_pose_mask = np.ones((self.NUM_DRONES, )).astype(bool)
                other_pose_mask[nth] = False

                gaussian_obs = self.decisions[nth].step(
                    curr_time=self.CurrTime,
                    detection_map=self._computePositionEstimation(
                        adjacency_Mat, nth),
                    ego_heading=circle_to_yaw(self._computeHeading(nth)[:2]),
                    fov_vector=self._computeFovVector(nth),
                    relative_pose=relative_position[nth][other_pose_mask],
                    node_coords=self.IPPEnvs[nth].node_coords)
                # 合并两个 obs
                obs[nth] = gaussian_obs | self.IPPEnvs[nth].Obs

        self.plot_online()
        return obs[self.control_by_RL_ID[0]]

    def _preprocessAction(self, action):
        """
        使用 PID 控制将 action 转化为 RPM, yaw_action 后续也应该从此处产生 

        Pre-processes the action passed to `.step()` into motors' RPMs.
        Descriptions:
            此处嵌套了 reynolds 用来计算高层速度控制指令

        Parameters
        ----------
        action : ndarray
            The desired target_yaw $$[cos(\theta), sin(\theta)]$$, to be translated into RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        if self.step_counter % self.FLOCKING_PER_PYB == 0:
            #### 更新 flocking 控制指令
            # migration mask 为 control mask 取反
            flocking_command = self._get_command_migration(
                migration_mask=self.control_by_RL_mask
            ) + self._get_command_reynolds()
            command_norm = np.linalg.norm(flocking_command,
                                          axis=1,
                                          keepdims=True)
            command_norm_safe = np.where(command_norm < 1e-10, 1, command_norm)
            flocking_command = flocking_command / command_norm_safe
            # 避免除0
            self.target_vs = np.hstack(
                (flocking_command,
                 np.min((np.ones(
                     command_norm.shape), command_norm / self.SPEED_LIMIT),
                        axis=0)))  # 将最大速度限制在 speed_limit

        if self.ACT_TYPE == ActionType.YAW:
            # 将对于 control_by_RL_mask 决策的 action 嵌入 action_all
            target_yaws_circle = np.zeros((self.NUM_DRONES, 2),
                                          dtype=np.float32)
            target_yaws_circle[self.control_by_RL_mask] = action
        elif self.ACT_TYPE == ActionType.IPP_YAW:
            target_yaws_circle = np.zeros((self.NUM_DRONES, 2),
                                          dtype=np.float32)
            for id in self.control_by_RL_ID:
                target_yaws_circle[id] = self.IPPEnvs[id].node_coords[action]
        target_yaws = circle_to_yaw(target_yaws_circle)
        return self._computeRpmFromCommand(self.target_vs,
                                           target_yaws=target_yaws)

    def _computeReward(self):

        return super()._computeReward()

    def _computeTerminated(self):

        return super()._computeTerminated()


class IPPenv:

    def __init__(self, yaw_start, act_type):

        self.graph_control = GraphController(start=yaw_start,
                                             k_size=IPPArg.k_size,
                                             act_type=act_type)
        #生成图
        self.node_coords, self.graph = self.graph_control.gen_graph(
            curr_coord=yaw_start,
            samp_num=IPPArg.sample_num,
            gen_range=IPPArg.gen_range)

        self.curr_node_index = 0  # 当前 node index
        self.yaw_start = yaw_start
        self.route_coord = [self.yaw_start]

    @property
    def Obs(self):
        '''
        返回 IPPenv 获得的 obs
        '''
        # 以 dict 形式获取观测

        ### edge_inputs, 表示采样 node_coords 中节点的 knn 连接关系
        edge_inputs = []
        # 遍历 values， 不遍历 keys
        for node in self.graph.values():
            node_edges = list(map(int, node))
            edge_inputs.append(node_edges)
        edge_inputs = np.asarray(edge_inputs)
        # 计算 graph_pos_encoding
        graph_pos_encoding = self.graph_pos_encoding(edge_inputs)
        ### curr_index
        curr_index = np.asarray(self.curr_node_index).reshape(-1, 1)
        return {
            "edge_inputs": edge_inputs,
            "curr_index": curr_index,
            "graph_pos_encoding": graph_pos_encoding
        }

    def graph_pos_encoding(self, edge_inputs):
        '''
        通过图的 laplace矩阵 的 特征向量 对节点进行编码，得到每个节点的低维位置表示。
        '''
        graph_size = IPPArg.sample_num
        A_matrix = np.zeros((graph_size, graph_size))
        D_matrix = np.zeros((graph_size, graph_size))
        for i in range(graph_size):
            for j in range(graph_size):
                if j in edge_inputs[i] and i != j:
                    A_matrix[i][j] = 1.0
        for i in range(graph_size):
            D_matrix[i][i] = 1 / np.sqrt(len(edge_inputs[i]) - 1)
        L = np.eye(graph_size) - np.matmul(D_matrix, A_matrix, D_matrix)
        eigen_values, eigen_vector = np.linalg.eig(L)
        idx = eigen_values.argsort()
        eigen_values, eigen_vector = eigen_values[idx], np.real(
            eigen_vector[:, idx])
        eigen_vector = eigen_vector[:, 1:IPPArg.num_eigen_value + 1]
        return np.asarray(eigen_vector)  #(graph_size, num_eigen_value)

    def reset(self, yaw_start):
        '''
        重新进行采样
        
        如何处理运行到一半的 action 呢？
        '''

        self.curr_node_index = 0
        self.yaw_start = yaw_start
        self.route_coord = [yaw_start]

        self.node_coords, self.graph = self.graph_control.gen_graph(
            curr_coord=yaw_start,
            samp_num=IPPArg.sample_num,
            gen_range=IPPArg.gen_range)


if __name__ == "__main__":
    pass
