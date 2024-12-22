import os
import pdb
import matplotlib
import matplotlib.axes
import matplotlib.collections
import matplotlib.figure
import matplotlib.lines
import matplotlib.pyplot
import numpy as np
from gymnasium import spaces
from functools import lru_cache
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, FOVType, ActionType, ObservationType
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.Reynolds import Reynolds
from gym_pybullet_drones.envs.gaussian_process.uav_gaussian import UAVGaussian as decision
from scipy.spatial.transform import Rotation as R
from gym_pybullet_drones.utils.utils import *
import matplotlib.pyplot as plt
from .IPPArguments import IPPArg


class FlockingAviary(BaseRLAviary):
    """
    Multi-drone RL environment class for high-level planning.
    需要将继承的 BaseAviary 替换成为 BaseRLAviary 

    Aviary 鸟笼， 类似前面代码中的 env
    Action space: 暂定仅有 heading 控制指令
    Observation space: 高斯过程的拟合结果 or 计算的最终结果。
    """

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 1,
                 control_by_RL_mask=None,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 flocking_freq_hz: int = 10,
                 decision_freq_hz: int = 5,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 use_reynolds=True,
                 default_flight_height=1.0,
                 output_folder='results',
                 fov_config: FOVType = FOVType.SINGLE,
                 obs: ObservationType = ObservationType.GAUSSIAN,
                 act: ActionType = ActionType.YAW,
                 random_point=True):
        """Initialization of an aviary environment for or high-level planning.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
            drone model  传入方式仅为 enum, 作为 urdf 文件的索引
        num_drones : int, optional
            The desired number of drones in the aviary.
        control_by_RL_mask: ndistance_curr - position_mean
            in shape(num_drones,).astype(bool), 表示哪些无人机受到 decision 控制, 不受到 decision 控制的无人机具有超能力！
            if control_by_RL_mask is "random", 随机采样无人机 control by RL mask
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        flocking_freq_hz:
            Frequency of flocking update.
        decision_freq_hz:
            Frequency of decison update. 
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.
        use_reynolds: bool, false
            是否使用 reynolds 模型
        fov_config: FOVtype
            使用哪种 fov 配置
        obs: ObservationType
            The type of observation space
        act: ActionType
            The type of action space
        random_point:
            if use random_point in Reynolds command
        """
        #### Create integrated controllers #########################
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 和某个并行运算库相关

        if drone_model in [
                DroneModel.CF2X, DroneModel.CF2P, DroneModel.VSWARM_QUAD
        ]:
            self.ctrl = [
                DSLPIDControl(drone_model=DroneModel.CF2X
                              )  # 此处 vswarm_quad 是套壳的 cf2x ，因此使用 cf2x 的控制方法
                for _ in range(num_drones)
            ]
        # init control_by_RL_mask
        mask = np.zeros((num_drones, ))
        if isinstance(control_by_RL_mask,
                      str) and control_by_RL_mask == "random":
            self.RANDOM_RL_MASK = True
            mask[np.random.randint(0, num_drones)] = 1
        else:
            mask[0] = 1
        self.control_by_RL_mask = mask.astype(bool)
        self.control_by_RL_ID = np.array(
            list(range(0, num_drones)), dtype=np.int8)[self.control_by_RL_mask]
        ### position estimation error level #####
        self.position_noise_std = [0.11, 0.16, 0.22, 0.31, 0.42, 0.50, 0.60]

        ######### finally, super.__init__()
        if act == ActionType.YAW_RATE_DISCRETE:
            self.NUM_DISCRETE_ACTION = 9  # 最好是一个奇数，action 以 int(NUM_DISCRETE_ACTION/2) 为中点
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         output_folder=output_folder,
                         obs=obs,
                         act=act)
        #### Set a limit on the maximum target speed ###############
        self.SPEED_LIMIT = 0.6  # m/s
        if self.ACT_TYPE == ActionType.YAW_DIFF:
            MAX_YAW_RATE = np.deg2rad(10)  # max deg/s
            self.MAX_YAW_DIFF = MAX_YAW_RATE / decision_freq_hz
        if self.ACT_TYPE == ActionType.YAW_RATE or self.ACT_TYPE == ActionType.YAW_RATE_DISCRETE:
            self.MAX_YAW_RATE = np.deg2rad(35)  #

        #### reynolds #############
        self.FLOCKING_FREQ_HZ = flocking_freq_hz
        if self.PYB_FREQ % self.FLOCKING_FREQ_HZ != 0:
            raise ValueError
        self.FLOCKING_PER_PYB = int(self.PYB_FREQ / self.FLOCKING_FREQ_HZ)
        self.use_reynolds = use_reynolds
        self.default_flight_height = default_flight_height
        self.RANDOM_POINT = random_point
        if self.use_reynolds:
            self.reynolds = Reynolds(random_point=self.RANDOM_POINT)
        self.fov_range = fov_config.value
        self.FOV = None
        ### decision
        self.decisions = {}
        self.DECISION_FREQ_HZ = decision_freq_hz
        if self.PYB_FREQ % self.DECISION_FREQ_HZ != 0:
            raise ValueError
        # 每 self.DECISION_PER_PYB 次 pyb 对应一次 decision
        self.DECISION_PER_PYB = int(self.PYB_FREQ / self.DECISION_FREQ_HZ)
        if self.CTRL_FREQ % self.DECISION_FREQ_HZ != 0:
            raise ValueError
        self.DECISION_PER_CTRL = int(self.CTRL_FREQ / self.DECISION_FREQ_HZ)

        ######### for _preprocessAction
        self.target_vs = np.zeros((self.NUM_DRONES, 4))
        self.target_yaw_circle = np.zeros((self.NUM_DRONES, 2))  # 以单位圆上表达的 yaw
        # 将 yaw action 累加到 self.target_yaw 之上, 如果在 RL 模式中，为 decision 直接的输出
        self.target_yaw = np.zeros((self.NUM_DRONES, ))

        ### cache
        self.cache = {}
        self.cache['obs'] = None
        self.cache['unc'] = [1.0] * self.NUM_DRONES

        ### hyper param
        self.VISABLE_DEGREE_THERSHOLD = 5  # in degree
        self.VISABLE_FAIL_DETECT = 0.05  # 5% 的概率无法检出目标

    ################################################################################
    def _gp_debug_init(self, user_debug_gui):
        plt.close("all")
        self.plot_online_stuff = {}
        self.plot_online_stuff: dict[
            str, tuple[matplotlib.figure.Figure, matplotlib.axes.Axes,
                       matplotlib.collections.QuadMesh,
                       list[list[matplotlib.lines.Line2D]]]]

        # 获取 gird_xx 和 grid_yy 进行 plot
        grid_size = self.decisions[
            self.control_by_RL_ID[0]].GP_ground_truth.grid_size
        self.grid_xx = self.decisions[
            self.control_by_RL_ID[0]].GP_ground_truth.grid[:, 0].reshape(
                (grid_size, grid_size))
        self.grid_yy = self.decisions[
            self.control_by_RL_ID[0]].GP_ground_truth.grid[:, 1].reshape(
                grid_size, grid_size)

        def init_animation(name, figsize=[5, 5]):
            figure, ax = plt.subplots(num=name, figsize=figsize)
            ax: matplotlib.axes.Axes
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_title(name)
            quadmesh = ax.pcolormesh(self.grid_xx,
                                     self.grid_yy,
                                     np.random.random((40, 40)),
                                     shading='auto',
                                     vmin=0,
                                     vmax=1)
            fov = [ax.plot([], [], 'b-'), ax.plot([], [], 'b-')]
            return figure, ax, quadmesh, fov

        if user_debug_gui:
            for index in self.control_by_RL_ID:
                self.plot_online_stuff[f"gp_std_{index}"] = init_animation(
                    f"gp_std_{index}")
                self.plot_online_stuff[f"gp_pred_{index}"] = init_animation(
                    f"gp_pred_{index}")

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            An ndarray of shape (NUM_DRONES, 2) for the commanded yaw action vectors.
            yaw_action here is 由单位圆上的点表示
        Note
            原项目并不兼容仅部分无人机被RL控制的算法，这里兼容仅部分无人机受RL控制
        """
        #### Action vector ######### X       Y       Z   fract. of MAX_SPEED_KMH
        if self.ACT_TYPE == ActionType.YAW:
            act_lower_bound = np.array([
                -1.0 * np.ones((2, )) for mask in self.control_by_RL_mask
                if mask
            ])
            act_upper_bound = np.array(
                [np.ones((2, )) for mask in self.control_by_RL_mask if mask])
        elif self.ACT_TYPE == ActionType.YAW_DIFF or self.ACT_TYPE == ActionType.YAW_RATE:
            act_lower_bound = np.array([
                -1.0 * np.ones((1, )) for mask in self.control_by_RL_mask
                if mask
            ])
            act_upper_bound = np.array(
                [np.ones((1, )) for mask in self.control_by_RL_mask if mask])
        elif self.ACT_TYPE == ActionType.YAW_RATE_DISCRETE:
            return spaces.Discrete(n=self.NUM_DISCRETE_ACTION,
                                   seed=42,
                                   start=0)
        else:
            print("[ERROR] in FlockingAviary._actionspace()")
        return spaces.Box(low=act_lower_bound,
                          high=act_upper_bound,
                          dtype=np.float32)

    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        spaces.Box
            The observation space, i.e., and ndarray of shape (NUM_DRONES, 20).

        """
        if self.OBS_TYPE == ObservationType.POSE:
            # pose 形式的观测用什么样的观测输入呢： 初步考虑使用 distance heading
            # 对于无法观测到的无人机，使用 np.inf 作为观测输入？

            pass

        elif self.OBS_TYPE == ObservationType.GAUSSIAN:
            heat_map_sz = 40**2
            obs_lower_bound = np.array([
                -1.0 * np.ones((heat_map_sz, ))
                for mask in self.control_by_RL_mask if mask
            ])
            obs_upper_bound = np.array([
                np.ones((heat_map_sz, )) for mask in self.control_by_RL_mask
                if mask
            ])
            return spaces.Box(low=obs_lower_bound,
                              high=obs_upper_bound,
                              dtype=np.float32)

        else:
            raise NotImplementedError

    ################################################################################
    @property
    def _relative_position(self):
        '''
        其中包含了相对自身的位置 [0,0], 需处理, in shape [num_drones,num_drones,2]
        '''
        drone_poses = self.drone_states[:, 0:3]
        return np.array([[
            drone_poses[other, :2] - drone_poses[ego, :2]
            for other in range(self.NUM_DRONES)
        ] for ego in range(self.NUM_DRONES)])

    def _get_command_reynolds(self, smooth_factor=0.3):
        '''
        Args
            neighbors: 用来表示无人机间邻居/观测关系
        '''

        # observation_vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ
        drone_velocities = self.drone_states[:, 10:13]

        # 两次循环计算相对位置与相对速度，仅考虑平面上的flocking
        relative_position = self._relative_position

        relative_velocities = np.array([[
            drone_velocities[other, :2] - drone_velocities[ego, :2]
            for other in range(self.NUM_DRONES)
        ] for ego in range(self.NUM_DRONES)])

        # 计算 观测 邻接矩阵
        adjacencyMat = self._computeAdjacencyMatFOV()

        #根据相对位置与 adjacencyMat 计算 reynolds 控制指令
        reynolds_commands = []
        super_power_adj_mat = np.ones((self.NUM_DRONES, self.NUM_DRONES))
        np.fill_diagonal(super_power_adj_mat, 0)
        for i in range(self.NUM_DRONES):
            if self.control_by_RL_mask[i]:
                reynolds_commands.append(
                    self.reynolds.command(
                        relative_position[i][adjacencyMat[i].astype(bool)]))
                # 这里考虑替换 带噪声的 位置估计
            else:
                reynolds_commands.append(
                    self.reynolds.command(
                        relative_position[i][super_power_adj_mat[i].astype(
                            bool)], relative_velocities[i][
                                super_power_adj_mat[i].astype(bool)]))
                # 如果不需要超能力 relative_position[i][self.adjacencyMat[i].astype(bool)]
        reynolds_commands = np.array(reynolds_commands)

        if self.cache.get("reynolds_command", None) is None:
            self.cache["reynolds_command"] = reynolds_commands

        reynolds_commands = self.cache[
            "reynolds_command"] * smooth_factor + reynolds_commands * (
                1 - smooth_factor)
        self.cache["reynolds_command"] = reynolds_commands  # 更新历史信息

        # 添加z轴，将z轴reynolds_command 设置为0
        reynolds_commands = np.hstack(
            (reynolds_commands, np.zeros((self.NUM_DRONES, 1))))
        assert reynolds_commands.shape == (self.NUM_DRONES, 3)
        return reynolds_commands

    def _get_command_migration(self, migration_mask=None):
        '''
            将 migration_mask 为 true 的位置 migration 置 0
        '''
        drone_poses = self.drone_states[:, 0:3]
        # z 轴速度不考虑
        drone_poses[:, 2] = 0
        migration_command = self.reynolds.get_migration_command(drone_poses)
        if migration_mask is not None:
            migration_command[migration_mask] = np.zeros_like(
                migration_command[migration_mask])  # 置0

        return migration_command

    ################################################################################
    def _computePositionEstimation(self, AdjacencyMat, nth_drone):
        '''
        根据无人机邻接矩阵，计算 nth_drone 无人机对出现在视野中无人机的位置估计结果
        Args:
            AdjacencyMat:  Adjacency mat 
        Return:
            Detection_map : key(nth_drone):value(pos estimation in 2D)
        '''
        drone_poses = self.drone_states[:, 0:3]
        poses_in_fov = self.world2ego_noquad(
            nth_drone, drone_poses[AdjacencyMat[nth_drone].astype(bool)])
        pose_index_in_fov = np.array(list(range(
            self.NUM_DRONES)))[AdjacencyMat[nth_drone].astype(bool)]

        detection_map = {}
        ######### 添加测量噪声
        for pose, index in zip(poses_in_fov, pose_index_in_fov):
            noise_index = int(
                min(
                    len(self.position_noise_std) - 1,
                    np.linalg.norm(pose[:2])))
            # np.sqrt(2) 是假设 x,y 轴噪声水平相当的情况，添加噪声
            pose += np.random.normal(
                loc=0,
                scale=self.position_noise_std[noise_index],
                size=pose.shape[0]) / np.sqrt(2)
            detection_map[index] = pose[:2]
        return detection_map

    def _computeAdjacencyMatFOV(self):
        '''
        在考虑 fov 的情况下, 计算无人机间观测邻接矩阵
        '''
        # mat = np.ones((self.NUM_DRONES, self.NUM_DRONES))
        # np.fill_diagonal(mat, 0)
        # return mat
        mat = np.array([
            self._computeFovMaskOcclusion(nth_drone)
            for nth_drone in range(self.NUM_DRONES)
        ])
        return mat

    ################################################################################

    def _computeFovVector(self, nth_drone):
        '''
        return:
            fov_vector in world coordinate
        '''
        ego_heading = self._computeHeading(nth_drone)
        fov_vector = np.array([
            np.dot(
                R.from_euler('z', theta,
                             degrees=False).as_matrix().reshape(3, 3),
                ego_heading) for theta in self.fov_range
        ])
        return fov_vector

    def _computeFovMaskOcclusion(self, nth_drone):
        '''
        判断 nth_drone 的 FOV 之内有哪些无人机，且考虑遮挡关系
        '''

        def visable(start, target, obstacles):
            '''
            target 是否被 obstacles 中任意点遮挡
            '''
            line_vec = target - start
            line_length = np.linalg.norm(line_vec)
            for obs in obstacles:
                obs_vec = obs - start
                # 这里使用 arccos 计算两个向量之间的夹角
                angle = np.rad2deg(
                    np.arccos(
                        np.clip(
                            np.dot(line_vec, obs_vec) /
                            (np.linalg.norm(line_vec) *
                             np.linalg.norm(obs_vec)), -1.0, 1.0)))

                if np.abs(angle) < self.VISABLE_DEGREE_THERSHOLD:
                    proj_length = np.dot(obs_vec, line_vec) / line_length
                    if 0 < proj_length < line_length:
                        return False
            return np.random.random(
            ) > self.VISABLE_FAIL_DETECT  # 90% 的概率能够检出目标

        mask = self._computeFovMask(nth_drone)

        for index, pos in enumerate(mask.astype(bool)):
            if pos:
                mask_temp = mask
                mask_temp[index] = 0
                # 此处判断 target 是否被任意 obstacles 遮挡
                mask[index] = visable(
                    self.drone_states[nth_drone, 0:2], self.drone_states[index,
                                                                         0:2],
                    self.drone_states[mask_temp.astype(bool), 0:2])

        return mask

    def _computeFovMask(self, nth_drone):
        '''
        判断 nth_drone 的 FOV 之内有哪几个 无人机
        '''
        mask = np.zeros((self.NUM_DRONES, ))
        # 计算无人机当前在世界坐标系下的 fov vector
        fov_vector = self._computeFovVector(nth_drone)
        for other in set(range(0, self.NUM_DRONES)) - set([nth_drone]):
            mask[other] = int(
                in_fov(
                    self.world2ego_noquad(nth_drone, self.drone_states[other,
                                                                       0:3]),
                    fov_vector))

        return mask

    def computeYawActionTSP(self, obs):
        '''
        使用基本的 tsp base line 计算 yaw command
        '''
        # no action
        # return self.target_yaw_circle
        if self.step_counter % self.DECISION_PER_PYB == 0:
            yaw_action = np.zeros((self.NUM_DRONES, ))
            obs_index = 0
            for nth in self.control_by_RL_ID:
                yaw_action[nth] = self.decisions[nth].DecisionStep(
                    obs[obs_index])
                obs_index += 1
            self.target_yaw = self.target_yaw + yaw_action
            self.target_yaw_circle = yaw_to_circle(self.target_yaw)
        # if not, return last decision
        return self.target_yaw_circle

    def _computeDroneState(self):
        """
        此处的 obs 是针对于环境的 obs,计算 reynolds 是针对于 每个无人机个体的 obs
        Returns the current observation of the environment.

        For the value of the state, see the implementation of `_getDroneStateVector()`.

        Returns
        -------
        ndarray
            An ndarray of shape (NUM_DRONES, 20) with the state of each drone.

        """
        return np.array(
            [self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

    ################################################################################
    def reset(self, seed=None, options=None):
        #### 重新初始化控制器 对于所有无人机
        if self.DRONE_MODEL in [
                DroneModel.CF2X, DroneModel.CF2P, DroneModel.VSWARM_QUAD
        ]:
            self.ctrl = [
                DSLPIDControl(drone_model=DroneModel.CF2X)
                for _ in range(self.NUM_DRONES)
            ]

        ######### for _preprocessAction
        self.target_vs = np.zeros((self.NUM_DRONES, 4))
        self.target_yaw_circle = np.zeros((self.NUM_DRONES, 2))  # 以单位圆上表达的 yaw
        self.target_yaw = np.zeros((self.NUM_DRONES, ))
        self.reynolds = Reynolds(random_point=self.RANDOM_POINT)

        ### cache
        self.cache = {}
        self.cache['obs'] = None
        self.cache['unc'] = [1.0] * self.NUM_DRONES
        ### 初始化 debug gui
        if self.USER_DEBUG:
            self._gp_debug_init(self.USER_DEBUG)

        return super().reset(seed, options)

    def step(self, action):
        '''
        This step in frequency of self.DECISION_FREQ_HZ
                Parameters
        ----------
        action : ndarray | dict[..]
            The input action for one or more drones, translated into RPMs by
            the specific implementation of `_preprocessAction()` in each subclass.

        Returns
        -------
        ndarray | dict[..]
            The step's observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        float | dict[..]
            The step's reward value(s), check the specific implementation of `_computeReward()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is over, check the specific implementation of `_computeTerminated()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is truncated, check the specific implementation of `_computeTruncated()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is trunacted, always false.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.
        '''

        for _ in range(self.DECISION_PER_CTRL - 1):
            # subclass step is in frequency of CTRL
            # repeat, flocking update in _preprocessAction
            super().step(action, need_return=False)
        # last times
        return super().step(action, need_return=True)

    ################################################################################
    @property
    def drone_states(self):
        if self.cache.get("last_state_step", -1) != self.step_counter:
            self.cache["last_state_step"] = self.step_counter
            self.cache["drone_states"] = self._computeDroneState()
        # 在 step_counter 没有得到更新时， 返回cache内容
        return self.cache["drone_states"]

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

        elif self.ACT_TYPE == ActionType.YAW_RATE or self.ACT_TYPE == ActionType.YAW_RATE_DISCRETE:
            target_yaws_circle = np.zeros((self.NUM_DRONES, 2),
                                          dtype=np.float32)
            target_yaw_rates = np.zeros((self.NUM_DRONES, ), dtype=np.float32)
            for index in self.control_by_RL_ID:
                # 从 [0,10] 映射到 [-5,5]
                coeff = float(action.squeeze(
                ) - int(self.NUM_DISCRETE_ACTION / 2)) / int(
                    self.NUM_DISCRETE_ACTION / 2
                ) if self.ACT_TYPE == ActionType.YAW_RATE_DISCRETE else action.squeeze(
                )
                target_yaw_rates[index] = coeff * self.MAX_YAW_RATE
            # return in target rate mode
            return self._computeRpmFromCommand(
                self.target_vs, target_yaw_rates=target_yaw_rates)

        target_yaws = circle_to_yaw(target_yaws_circle)
        return self._computeRpmFromCommand(self.target_vs,
                                           target_yaws=target_yaws)

    def _computeRpmFromCommand(self,
                               target_vs,
                               target_yaws=None,
                               target_yaw_rates=None):
        '''
        Args:
            target_vs: in shape (num_drones,4)
            target_yaws: in shape (num_drones,1)
        '''
        if target_yaw_rates is None:
            target_yaw_rates = np.zeros((self.NUM_DRONES, ))
        rpm = np.zeros((self.NUM_DRONES, 4))
        for k in range(self.NUM_DRONES):
            #### Get the current state of the drone  ###################
            state = self._getDroneStateVector(k)
            target_v = target_vs[k]
            # in yaw_rate mode , target_yaw 应当为当前的 yaw
            target_yaw = target_yaws[k] if target_yaws is not None else state[9]
            #### Normalize the first 3 components of the target velocity
            if np.linalg.norm(target_v[0:3]) != 0:
                v_unit_vector = target_v[0:3] / np.linalg.norm(target_v[0:3])
            else:
                v_unit_vector = np.zeros(3)
            #### 如果在 reynolds 中, 需要设置默认飞行高度
            target_pos = state[0:3]
            if self.use_reynolds:
                target_pos[2] = self.default_flight_height
            temp, _, _ = self.ctrl[k].computeControl(
                control_timestep=self.CTRL_TIMESTEP,
                cur_pos=state[0:3],
                cur_quat=state[3:7],
                cur_vel=state[10:13],
                cur_ang_vel=state[13:16],
                target_pos=target_pos,  # same as the current position
                target_rpy=np.array([0, 0, target_yaw]),
                target_vel=self.SPEED_LIMIT * np.abs(target_v[3]) *
                v_unit_vector,  # target the desired velocity vector
                target_rpy_rates=np.array([0, 0, target_yaw_rates[k]]))
            rpm[k, :] = temp
        return rpm

    ################################################################################
    def _computeObs(self):
        '''
        Return the current observation of the environment.
        ## Description:
            self.drone_poses 在此处得到更新
            self.detection_map 在此处得到更新, 由于 detection 包含随机数，因此每次循环仅更新一次
        '''

        self.decisions: dict[int, decision]
        if self.OBS_TYPE == ObservationType.GAUSSIAN:
            ############ obs type in gaussian
            if self.step_counter % self.DECISION_PER_PYB == 0:
                obs = []
                adjacency_Mat = self._computeAdjacencyMatFOV()
                # 由 detection step 获得观测
                relative_position = self._relative_position
                for nth in self.control_by_RL_ID:
                    other_pose_mask = np.ones((self.NUM_DRONES, ))
                    other_pose_mask[nth] = .0
                    obs_nth = self.decisions[nth].step(
                        curr_time=self.curr_time,
                        detection_map=self._computePositionEstimation(
                            adjacency_Mat, nth),
                        ego_heading=circle_to_yaw(
                            self._computeHeading(nth)[:2].reshape(-1, 2)),
                        fov_vector=self._computeFovVector(nth),
                        relative_pose=relative_position[nth][
                            other_pose_mask.astype(bool)])
                    obs.append(obs_nth)
                # 这样做是由于 obs 在 step_counter == 0 可以初始化
                self.cache['obs'] = np.array(obs[0]).reshape(1, 1600).astype(
                    np.float32)
        self.plot_online()
        return np.asarray(self.cache['obs'])

    def plot_online(self):
        """
        更新 plot_online_stuff 的内容
        """
        if self.USER_DEBUG:
            for index in self.control_by_RL_ID:
                std_array = self.decisions[index].cache["all_std"].reshape(
                    (40, 40))
                pred_array = self.decisions[index].cache["all_pred"].reshape(
                    (40, 40))

                self.plot_online_stuff[f"gp_std_{index}"][2].set_array(
                    std_array)
                self.plot_online_stuff[f"gp_pred_{index}"][2].set_array(
                    pred_array)

                # 绘制 fov
                fov_vector = self._computeFovVector(index)
                # fov 1
                self.plot_online_stuff[f"gp_pred_{index}"][3][0][0].set_data(
                    [0, fov_vector[0][0]], [0, fov_vector[0][1]])
                # fov 2
                self.plot_online_stuff[f"gp_pred_{index}"][3][1][0].set_data(
                    [0, fov_vector[1][0]], [0, fov_vector[1][1]])

            plt.pause(1e-9)

    def _computeReward(self):
        """Computes the current reward value(s).
        - Reward 是 agent 与环境交互获得的 reward , 与 obs 无关
        
        Reward 当前为所有 reward 的累积

        Returns
        -------
        int
            Dummy value.

        """
        if self.OBS_TYPE in [ObservationType.GAUSSIAN, ObservationType.IPP]:

            def compute_reward(nth):
                ground_truth = self.decisions[nth].GP_ground_truth.fn()
                high_info_idx = self.decisions[
                    nth].GP_ground_truth.get_high_info_indx(ground_truth)
                ## Unc update reward
                _, unc_list = self.decisions[nth].GP_detection.eval_avg_unc(
                    self.curr_time, high_info_idx, return_all=True)
                unc_list = np.asarray(unc_list)
                unc_list[np.isnan(unc_list)] = 1.0  # nan值设置为1
                unc_update = self.cache['unc'][nth] - unc_list
                reward = np.sum(unc_update[unc_update > .0])
                self.cache['unc'][nth] = unc_list

                ## Unc reward, 鼓励减少不确定性
                unc_reward = 1 - unc_list
                reward += np.sum(unc_reward[unc_reward > .0])
                return reward

            reward = np.zeros((self.NUM_DRONES, ))
            for nth in self.control_by_RL_ID:
                reward[nth] = compute_reward(nth)

            # 惩罚较大 action
            return np.sum(reward).astype(float)

    ################################################################################

    def _computeTerminated(self):
        """Computes the current terminated value(s).

        这些条件都是导致

        Returns
        -------
        bool
            Dummy value.

        """
        # 这里 target_vs 的最后一项为 norm
        if np.all(np.abs(self.target_vs[:, -1]) < 1e-3):
            return True  # 不名原因速度消失

        drone_states = self.drone_states
        relative_position = self._relative_position

        def terminated(nth):
            other_mask = np.ones((self.NUM_DRONES)).astype(bool)
            other_mask[nth] = False
            relative_distance = np.linalg.norm(relative_position[nth],
                                               axis=1)[other_mask]
            # 无人机间最小距离小于 1.0 m
            if np.min(relative_distance) < 1.0:
                if self.USER_DEBUG:
                    print("Terminated min distance")
                return True
            # truncted when fly too low
            if drone_states[nth][2] < 1.5:
                if self.USER_DEBUG:
                    print("Terminated fly too low")
                return True
            # nth 无人机与其余无人机最小距离大于 x
            if np.min(relative_distance) > 3.4:
                if self.USER_DEBUG:
                    print("Terminated too close")
                return True
            if self.curr_time > IPPArg.MAX_EPISODE_LEN:
                if self.USER_DEBUG:
                    print("Terminated max episode length")
                return True
            return False

        for idx in self.control_by_RL_ID:
            if terminated(idx):
                return True
        return False

    ################################################################################

    def _computeTruncated(self):
        """Computes the current truncated value(s).

        truncted: 任务被迫中止, 由于外部限制回合被截断
        -------
        bool
            Dummy value.
        
        """
        drone_states = self.drone_states

        def truncated(nth):
            # truncate when a drone is too tilted
            if abs(drone_states[nth][7]) > .4 or abs(
                    drone_states[nth][8]) > .4:
                if self.USER_DEBUG:
                    print("Truncated too tilted")
                return True
            return False

        for idx in self.control_by_RL_ID:
            if truncated(idx):
                return True

        return False

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {
            "answer": 42
        }  #### Calculated by the Deep Thought supercomputer in 7.5M years
