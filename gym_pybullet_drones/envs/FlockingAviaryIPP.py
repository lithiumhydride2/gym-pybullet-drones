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
            self.decisions[nth] = decision(
                fov_range=self.fov_range,
                nth_drone=nth,
                num_drone=self.NUM_DRONES,
                planner=None,
                node_coords=self.IPPEnvs[nth].node_coords)

    def plot_online(self):
        super().plot_online()
        # 绘制图的采样
        if self.USER_DEBUG:
            for nth in self.control_by_RL_ID:
                node_coords = self.IPPEnvs[nth].node_coords
                curr_index = self.IPPEnvs[nth].curr_node_index
                self.plot_online_stuff[f"gp_pred_{nth}"][1].scatter(
                    node_coords[:, 0], node_coords[:, 1], c='orchid')
                self.plot_online_stuff[f"gp_pred_{nth}"][1].scatter(
                    node_coords[curr_index, 0],
                    node_coords[curr_index, 1],
                    c="red")
                plt.pause(1e-10)

    def step(self, action):
        # 使用当前 obs 后处理 action
        self.IPPEnvs[self.control_by_RL_ID[0]].step(action)

        # step 中重新计算 obs 与 action

        def finish_current_action(action):
            if circle_angle_diff(
                    self.IPPEnvs[self.control_by_RL_ID[0]].node_coords[action],
                    self._computeHeading(
                        self.control_by_RL_ID[0])[:2]) < np.deg2rad(5):
                return True
            return False

        for _ in range(self.DECISION_PER_CTRL - 1):
            # subclass step is in frequency of CTRL
            # repeat, flocking update in _preprocessAction
            super().step(action, need_return=False)

        while not finish_current_action(action):
            super().step(action, need_return=False)
        # last times
        return super().step(action, need_return=True)

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

        for nth in self.control_by_RL_ID:
            # 这里的 yaw_start 由于物理引擎后更新，使用 INIT_RYPS 初始化
            self.IPPEnvs[nth].reset(
                yaw_start=yaw_to_circle(self.INIT_RPYS[nth][-1])[:2])
            # 使用 IPPEnvs 的采样初始化 self.decision
            self.decisions[nth].reset(nth_drone=nth)
        return super().reset(seed, options)

    def _actionSpace(self):
        '''
        IPP_YAW 模式下，选取动作方式为从当前 node_coords 的邻居中选取下一个节点
        '''
        if self.ACT_TYPE == ActionType.IPP_YAW:
            return Discrete(IPPArg.sample_num)

    def _observationSpace(self):
        if self.OBS_TYPE == ObservationType.IPP:
            return Dict({
                "node_inputs":
                Box(
                    low=0.,
                    high=1.,
                    shape=(IPPArg.history_size // IPPArg.history_stride,
                           IPPArg.sample_num,
                           self.NUM_DRONES * 3),  # 3: (yaw_coord, belief)
                    dtype=np.float32),
                "dt_pool_inputs":
                Box(low=-np.inf,
                    high=0.,
                    shape=(IPPArg.history_size // IPPArg.history_stride, 1),
                    dtype=np.float32),
                "curr_index":
                Box(low=0,
                    high=IPPArg.sample_num - 1,
                    shape=(1, 1),
                    dtype=np.int64),
                "dist_inputs":
                Box(low=0.,
                    high=1.,
                    shape=(IPPArg.sample_num, 1),
                    dtype=np.float32)
            })

    def _computeObs(self):
        '''
        Return the current observation of the environment.
        '''
        # 获得增广图形式的观测
        ### 这里取消 step 与 decision 的严格对其
        # assert self.step_counter % self.DECISION_PER_PYB == 0
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
                    curr_time=self.curr_time,
                    detection_map=self._computePositionEstimation(
                        adjacency_Mat, nth),
                    ego_heading=circle_to_yaw(
                        self._computeHeading(nth)[:2].reshape(-1, 2)),
                    relative_pose=relative_position[nth][other_pose_mask])
                # 合并两个 obs
                obs[nth] = gaussian_obs | self.IPPEnvs[nth].Obs
        # cache for action subprocess
        self.cache["obs"] = obs
        self.plot_online()
        return obs[self.control_by_RL_ID[0]]

    def _computeReward(self):
        reward = super()._computeReward()
        for nth in self.control_by_RL_ID:
            smooth_reward = circle_angle_diff(
                self.IPPEnvs[nth].route_coord[-1],
                self.IPPEnvs[nth].route_coord[-2]) * 0.1
            reward -= smooth_reward
        return float(reward)

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


class IPPenv:

    def __init__(self, yaw_start, act_type):

        self.graph_control = GraphController(start=yaw_start,
                                             act_type=act_type,
                                             random_sample=False)
        #生成图
        self.node_coords, self.distance_matrix = self.graph_control.gen_graph(
            curr_coord=yaw_start,
            samp_num=IPPArg.sample_num,
            gen_range=IPPArg.gen_range)

        self.curr_node_index = self.graph_control.findNodeIndex(
            yaw_start)  # 当前 node index
        self.yaw_start = yaw_start
        self.route_coord = [self.yaw_start, self.yaw_start]

    @property
    def Obs(self):
        '''
        返回 IPPenv 获得的 obs
        '''
        # 计算 graph_pos_encoding
        ### curr_index
        curr_index = np.asarray(self.curr_node_index).reshape(-1, 1)
        dist_inputs = self.calc_distance_of_nodes(curr_index)
        return {"curr_index": curr_index, "dist_inputs": dist_inputs}

    def calc_distance_of_nodes(self, current_index):
        '''
        仅计算当前节点与相连节点的距离
        
        使用 np.pi 进行归一化， 不相连节点距离设置为1
        '''
        all_dist = self.distance_matrix[current_index.item()].reshape(
            -1, 1) / np.pi
        return all_dist

    def reset(self, yaw_start):
        '''
        如何处理运行到一半的 action 呢？
        '''
        self.curr_node_index = self.graph_control.findNodeIndex(yaw_start)
        self.yaw_start = yaw_start
        self.route_coord = [yaw_start, yaw_start]

    def step(self, action):
        '''
        通过 action 更新当前节点
        '''
        self.curr_node_index = action
        self.route_coord.append(self.node_coords[action])


if __name__ == "__main__":
    pass
