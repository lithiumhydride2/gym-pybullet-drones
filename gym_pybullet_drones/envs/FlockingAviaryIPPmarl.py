import functools
from .FlockingAviary import *
from pettingzoo.utils.env import ParallelEnv
from .IPPArguments import IPPArg
from ..utils.graph_controller import GraphController
from ..utils.utils import circle_angle_diff
from gymnasium.spaces import Box, Dict, Discrete


class FlockingAviaryIPPmarl(FlockingAviary, ParallelEnv):
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

        super().__init__(drone_model, num_drones, control_by_RL_mask,
                         neighbourhood_radius, initial_xyzs, initial_rpys,
                         physics, pyb_freq, flocking_freq_hz, decision_freq_hz,
                         ctrl_freq, gui, record, obstacles, user_debug_gui,
                         use_reynolds, default_flight_height, output_folder,
                         fov_config, obs, act, random_point)
        # for petting zoo

        self.possible_agents = [i for i in range(self.NUM_DRONES)]
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
            for nth in self.control_by_RL_ID[:1]:
                node_coords = self.IPPEnvs[nth].node_coords
                curr_index = self.IPPEnvs[nth].curr_node_index
                self.plot_online_stuff[f"gp_pred_{nth}"][1].scatter(
                    node_coords[:, 0], node_coords[:, 1], c='orchid')
                self.plot_online_stuff[f"gp_pred_{nth}"][1].scatter(
                    node_coords[curr_index, 0],
                    node_coords[curr_index, 1],
                    c="red")
                plt.pause(1e-10)

    def step(self, actions):
        """Receives a dictionary of actions keyed by the agent name.

        Returns the observation dictionary, reward dictionary, terminated dictionary, truncated dictionary
        and info dictionary, where each dictionary is keyed by the agent.

        Args:
            actions : dict[AgentID, ActionType]
        """
        # override petting_zoo 的 step
        # action = actions
        assert self.ACT_TYPE in [ActionType.IPP_YAW, ActionType.YAW_DIFF]
        reprocess_action = np.zeros((self.NUM_DRONES, ))
        ### yaw_diff 模式下，action 为相对于当前节点的偏移
        if self.ACT_TYPE == ActionType.YAW_DIFF:
            for agent in self.control_by_RL_ID:
                action = self.IPPEnvs[agent].curr_node_index + actions[
                    agent] - 1
                action = IPPArg.sample_num - 1 if action == -1 else action
                action = 0 if action == IPPArg.sample_num else action
                reprocess_action[agent] = action
                self.IPPEnvs[agent].step(action)

        elif self.ACT_TYPE == ActionType.IPP_YAW:
            for agent in self.control_by_RL_ID:
                knn_edge_inputs = self.IPPEnvs[agent].knn_edge_inputs
                curr_index = self.IPPEnvs[agent].curr_node_index
                action = knn_edge_inputs[curr_index.item()][actions[agent]]
                reprocess_action[agent] = action
                self.IPPEnvs[agent].step(action)

        # step 中重新计算 obs 与 action

        def finish_current_action(action):
            for agent in self.control_by_RL_ID:
                if circle_angle_diff(
                        self.IPPEnvs[self.control_by_RL_ID[agent]].node_coords[
                            action[agent]],
                        self._computeHeading(
                            self.control_by_RL_ID[agent])[:2]) < np.deg2rad(5):
                    return True
            return False

        for _ in range(self.DECISION_PER_CTRL - 1):
            # subclass step is in frequency of CTRL
            # repeat, flocking update in _preprocessAction
            super().step(reprocess_action, need_return=False)

        # while not finish_current_action(reprocess_action):
        #     super().step(reprocess_action, need_return=False)
        # last times
        observations, rewards, terminateds, truncateds, infos = super().step(
            reprocess_action, need_return=True)

        for agent in self.agents:
            # 如果任意 agent 发生 terminated 或 truncated，则所有 agent terminated
            if terminateds.get(agent, False) or truncateds.get(agent, False):
                terminateds = {agent: True for agent in self.agents}
                self.agents = []  # for petting zoo， 需要将 agents 置空
                break

        return observations, rewards, terminateds, truncateds, infos

    def reset(self, seed=None, options=None):
        '''
        reset 的最终作用为获取 initial_obs, initial_info
        '''
        ### for petting zoo
        self.agents = self.possible_agents
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
            if self.IPPEnvs.get(nth, None) is not None:
                self.IPPEnvs[nth].reset(
                    yaw_start=yaw_to_circle(self.INIT_RPYS[nth][-1])[:2])
            else:
                self.IPPEnvs[nth] = IPPenv(yaw_start=yaw_to_circle(
                    self.INIT_RPYS[nth][-1])[:2],
                                           act_type=self.ACT_TYPE)
            # 使用 IPPEnvs 的采样初始化 self.decision
            if self.decisions.get(nth, None) is not None:
                self.decisions[nth].reset(nth_drone=nth)
            else:
                self.decisions[nth] = decision(
                    fov_range=self.fov_range,
                    nth_drone=nth,
                    num_drone=self.NUM_DRONES,
                    planner=None,
                    node_coords=self.IPPEnvs[nth].node_coords)
        return super().reset(seed, options)

    @functools.lru_cache(maxsize=IPPArg.NUM_DRONE)
    def observation_space(self, agent):
        '''
        override observation_space
        '''
        return self._observationSpace()

    @functools.lru_cache(maxsize=IPPArg.NUM_DRONE)
    def action_space(self, agent):
        '''override action_space'''
        return self._actionSpace()

    def _actionSpace(self):
        '''
        IPP_YAW 模式下，选取动作方式为从当前 node_coords 的邻居中选取下一个节点
        '''

        if self.ACT_TYPE == ActionType.YAW_DIFF:
            return Discrete(3)  # yaw 增大，保持，减小
        if self.ACT_TYPE == ActionType.IPP_YAW:
            return Discrete(IPPArg.k_size)  # 从邻居中选取一个节点

    def _observationSpace(self):
        if self.OBS_TYPE == ObservationType.SIMPLE:
            single_obs_space = Dict({
                "relative_obs":
                Box(low=-1.,
                    high=1.,
                    shape=(IPPArg.history_size // IPPArg.history_stride,
                           self.NUM_DRONES - 1, 3),
                    dtype=np.float32),  # 3: (cos,sin,belief)
                "curr_pos":
                Box(low=-1., high=1., shape=(1, 2), dtype=np.float32),
            })
        if self.OBS_TYPE == ObservationType.IPP:
            single_obs_space = Dict({
                "node_inputs":
                Box(
                    low=0.,
                    high=1.,
                    shape=(
                        IPPArg.history_size // IPPArg.history_stride,
                        IPPArg.sample_num, 2 +
                        (self.NUM_DRONES - 1) * 3),  # 3: (yaw_coord, belief)
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
                    dtype=np.float32),
                "edge_inputs":
                Box(low=0,
                    high=IPPArg.sample_num - 1,
                    shape=(IPPArg.k_size, 1),
                    dtype=np.int64)
            })
        ## 如果是 marl 的形式，返回多无人机 dict
        return single_obs_space

    def _computeObs(self):
        '''
        Return the current observation of the environment.
        OK for petting zoo
        '''
        ### 这里取消 step 与 decision 的严格对齐
        # assert self.step_counter % self.DECISION_PER_PYB == 0
        if self.OBS_TYPE == ObservationType.SIMPLE:
            obs = {}
            adjacency_Mat = self._computeAdjacencyMatFOV()
            relative_position = self._relative_position
            for nth in self.control_by_RL_ID:
                #TODO 获得观测时，需要更新 IPP_env
                # mask 用于获取真实相对位置
                other_pose_mask = np.ones((self.NUM_DRONES, )).astype(bool)
                other_pose_mask[nth] = False
                guassian_obs = self.decisions[nth].step(
                    curr_time=self.curr_time,
                    detection_map=self._computePositionEstimation(
                        adjacency_Mat, nth),
                    ego_heading=circle_to_yaw(
                        self._computeHeading(nth)[:2].reshape(-1, 2)),
                    relative_pose=relative_position[nth][other_pose_mask])
                obs[nth] = {
                    "relative_obs":
                    guassian_obs["relative_obs"],
                    "curr_pos":
                    self.IPPEnvs[nth].node_coords[
                        self.IPPEnvs[nth].curr_node_index].reshape(1, 2)
                }
            self.plot_online()
            if self.control_by_RL_mask.sum() == self.NUM_DRONES:
                ret = obs
            else:
                ret = obs[self.control_by_RL_ID[0]]

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
                graph_obs = self.IPPEnvs[nth].Obs
                obs[nth] = {
                    "node_inputs": gaussian_obs["node_inputs"],
                    "dt_pool_inputs": gaussian_obs["dt_pool_inputs"],
                    "dist_inputs": graph_obs["dist_inputs"],
                    "curr_index": graph_obs["curr_index"],
                    "edge_inputs": graph_obs["edge_inputs"]
                }
            # cache for action subprocess
            self.cache["obs"] = obs
            self.plot_online()
            if self.control_by_RL_mask.sum() == self.NUM_DRONES:
                ret = obs
            else:
                ret = obs[self.control_by_RL_ID[0]]

        return ret

    def _computeReward(self):
        '''
        转换为 petting zoo 的形式
        '''
        reward: np.ndarray = super()._computeReward()
        for nth in self.control_by_RL_ID:
            smooth_reward = circle_angle_diff(
                self.IPPEnvs[nth].route_coord[-1],
                self.IPPEnvs[nth].route_coord[-2]) * 1
            reward[nth] -= smooth_reward

        # 在 marl 的情况下， reward 为所有无人机 reward 的平均值
        if self.control_by_RL_mask.sum() == self.NUM_DRONES:
            reward_dict = {}
            for agent in self.control_by_RL_ID:
                reward_dict[agent] = reward[agent]
            return reward_dict
        else:
            return float(reward)

    def _preprocessAction(self, action):
        """
        使用 PID 控制将 action 转化为 RPM, yaw_action 后续也从此处产生， 在 BaseAviary 中被调用

        Pre-processes the action passed to `.step()` into motors' RPMs.
        Descriptions:
            此处嵌套了 reynolds 用来计算高层速度控制指令

        Parameters
        ----------
        action : ndarray
            (num_drones,)
            The desired target_yaw $$[cos(\theta), sin(\theta)]$$, to be translated into RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        if self.step_counter % self.FLOCKING_PER_PYB == 0:
            #### 更新 flocking 控制指令
            # migration_mask 为 true , 则无法获得导航迁移指令
            flocking_command = self._get_command_migration(
                migration_mask=None) + self._get_command_reynolds()
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

        if self.ACT_TYPE in [ActionType.IPP_YAW, ActionType.YAW_DIFF]:
            target_yaws_circle = np.zeros((self.NUM_DRONES, 2),
                                          dtype=np.float32)
            for id in self.control_by_RL_ID:
                target_yaws_circle[id] = self.IPPEnvs[id].node_coords[int(
                    action[id])]
        else:
            raise ValueError
        target_yaws = circle_to_yaw(target_yaws_circle)
        return self._computeRpmFromCommand(self.target_vs,
                                           target_yaws=target_yaws)


class IPPenv:

    def __init__(self, yaw_start, act_type):

        self.graph_control = GraphController(start=yaw_start,
                                             k_size=IPPArg.k_size,
                                             act_type=act_type,
                                             random_sample=False)
        #生成图
        self.node_coords, self.distance_matrix, self.knn_graph = self.graph_control.gen_graph(
            curr_coord=yaw_start,
            samp_num=IPPArg.sample_num,
            gen_range=IPPArg.gen_range)

        # 生成与当前节点相连节点的 node_coord
        self.knn_edge_inputs = []
        for node in self.knn_graph.edges.values():
            node_edges = list(map(int, node))
            self.knn_edge_inputs.append(node_edges)
        self.knn_edge_inputs = np.asarray(self.knn_edge_inputs)

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
        edge_inputs = self.knn_edge_inputs[curr_index.item()].reshape(-1, 1)
        return {
            "curr_index": curr_index,
            "dist_inputs": dist_inputs,
            "edge_inputs": edge_inputs
        }

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

    from pettingzoo.test import parallel_api_test
    env = FlockingAviaryIPPmarl()
    parallel_api_test(env)
