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
        assert act == ActionType.YAW
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

    def step(self):
        return super().step()

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
        for nth in self.control_by_RL_ID:
            self.IPPEnvs[nth] = IPPenv(yaw_start=self._computeHeading(nth)[:2],
                                       act_type=self.ACT_TYPE)

        return super().reset(seed, options)

    def _observationSpace(self):
        if self.OBS_TYPE == ObservationType.IPP:
            return Dict({
                "node_features":
                Box(
                    low=0.,
                    high=1.,
                    shape=(IPPArg.sample_num,
                           (self.NUM_DRONES - 1) * 2),  # mean and std
                    dtype=np.float32),
                # "edge_feature":
                # Box(low=0., high=1., shape=(TODO), dtype=np.float32)
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
                # 获得观测时，重新进行采样
                self.IPPEnvs[nth].resample(
                    curr_yaw=self._computeHeading(nth)[:2])

                # 用作
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
                obs[nth] = gaussian_obs

        self.plot_online()
        return obs[self.control_by_RL_ID[0]]

    def _computeReward(self):
        pass
        return .0

    def _computeTerminated(self):

        def outbudget():
            ## 判断是否将 budget 消耗完
            for nth in self.control_by_RL_ID:
                if self.IPPEnvs[nth].budget < 0:
                    return True
            return False

        return outbudget or super()._computeTerminated()


class IPPenv:

    def __init__(self, yaw_start, act_type):

        self.graph_control = GraphController(start=yaw_start,
                                             k_size=IPPArg.k_size,
                                             act_type=act_type)
        self.node_coords, self.graph = self.graph_control.gen_graph(
            curr_coord=yaw_start,
            samp_num=IPPArg.sample_num,
            gen_range=IPPArg.gen_range)
        self.curr_node_index = 0  # 当前 node index
        self.yaw_start = yaw_start
        self.route_coord = [self.yaw_start]

        self.budget_range = IPPArg.budget_range
        self.budget = np.random.uniform(low=self.budget_range[0],
                                        high=self.budget_range[1])

    @property
    def Obs(self):
        '''
        返回 IPPenv 获得的 obs
        '''
        self.node_coords, self.graph, self.budget

    def resample(self, curr_yaw):
        '''
        Description:
        
        Args:
            curr_yaw: 当前yaw 角度
        '''

        # 必须到达当前 action 才能进行离散图的 resample? 吗
        #TODO 如果在当前位置没有到达上次采样的位置，是否直接进行 resample
        self.node_coords, self.graph = self.graph_control.gen_graph(
            curr_coord=curr_yaw,
            samp_num=IPPArg.sample_num,
            gen_range=IPPArg.gen_range)

    def reset(self, yaw_start):
        '''
        重新进行采样
        
        如何处理运行到一半的 action 呢？
        '''
        # 重新采样 budget
        self.curr_node_index = 0
        self.yaw_start = yaw_start
        self.route_coord = [yaw_start]

        self.node_coords, self.graph = self.graph_control.gen_graph(
            curr_coord=yaw_start,
            samp_num=IPPArg.sample_num,
            gen_range=IPPArg.gen_range)

        self.budget = np.random.uniform(low=self.budget_range[0],
                                        high=self.budget_range[1])


if __name__ == "__main__":
    pass
