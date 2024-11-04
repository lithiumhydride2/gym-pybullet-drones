import os
import numpy as np
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, FOVType
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.Reynolds import Reynolds
from scipy.spatial.transform import Rotation as R


class FlockingAviary(BaseAviary):
    """
    Multi-drone environment class for high-level planning.

    Aviary 鸟笼， 类似前面代码中的 env
    """

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 1,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 use_reynolds=False,
                 default_flight_height=1.0,
                 output_folder='results',
                 fov_config: FOVType = FOVType.SINGLE):
        """Initialization of an aviary environment for or high-level planning.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
            drone model  传入方式仅为 enum, 作为 urdf 文件的索引
        num_drones : int, optional
            The desired number of drones in the aviary.
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
        """
        #### Create integrated controllers #########################
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 和某个并行运算库相关

        if drone_model in [
                DroneModel.CF2X, DroneModel.CF2P, DroneModel.VSWARM_QUAD
        ]:
            self.ctrl = [
                DSLPIDControl(drone_model=DroneModel.CF2X
                              )  # 此处 vswarm_quad 是套壳的 cf2x ，因此使用 cf2x 的控制方法
                for i in range(num_drones)
            ]
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
                         output_folder=output_folder)
        #### Set a limit on the maximum target speed ###############
        self.SPEED_LIMIT = 0.6  # m/s

        #### reynolds #############
        self.use_reynolds = use_reynolds
        self.default_flight_height = default_flight_height
        if self.use_reynolds:
            self.reynolds = Reynolds()
        self.last_reynolds_command = None

        ### fov config ############
        self.fov_range = fov_config.value

        ### position estimation error level #####
        self.position_noise_std = [0.11, 0.16, 0.22, 0.31, 0.42, 0.50]

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            An ndarray of shape (NUM_DRONES, 4) for the commanded velocity vectors.

        """
        #### Action vector ######### X       Y       Z   fract. of MAX_SPEED_KMH
        act_lower_bound = np.array([[-1, -1, -1, 0]
                                    for i in range(self.NUM_DRONES)])
        act_upper_bound = np.array([[1, 1, 1, 1]
                                    for i in range(self.NUM_DRONES)])
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
        #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
        obs_lower_bound = np.array([[
            -np.inf, -np.inf, 0., -1., -1., -1., -1., -np.pi, -np.pi, -np.pi,
            -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0., 0., 0.,
            0.
        ] for i in range(self.NUM_DRONES)])
        obs_upper_bound = np.array([[
            np.inf, np.inf, np.inf, 1., 1., 1., 1., np.pi, np.pi, np.pi,
            np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, self.MAX_RPM,
            self.MAX_RPM, self.MAX_RPM, self.MAX_RPM
        ] for i in range(self.NUM_DRONES)])
        return spaces.Box(low=obs_lower_bound,
                          high=obs_upper_bound,
                          dtype=np.float32)

    ################################################################################
    def get_command_reynolds(self, smooth_factor=0.3):
        '''
        Args
            neighbors: 用来表示无人机间邻居/观测关系
            此处添加 对象的新属性 self.drone_states = self._computeObs()
        '''

        # observation_vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ
        # 更新状态，以供后续计算
        self.drone_states = self._computeObs()  # (num_drones * 20)

        drone_poses = self.drone_states[:, 0:3]
        drone_velocities = self.drone_states[:, 10:13]  # reynolds 可以使用速度对齐项

        # 两次循环计算相对位置与相对速度，仅考虑平面上的flocking
        relative_position = np.array([[
            drone_poses[other, :2] - drone_poses[ego, :2]
            for other in range(self.NUM_DRONES)
        ] for ego in range(self.NUM_DRONES)])

        relative_velocities = np.array([[
            drone_velocities[other, :2] - drone_velocities[ego, :2]
            for other in range(self.NUM_DRONES)
        ] for ego in range(self.NUM_DRONES)])

        # 计算 观测 邻接矩阵
        self.adjacencyMat = self._computeAdjacencyMatFOV()

        # 由邻接矩阵，选取能够观测到的对象 这里待改进
        reynolds_commands = np.array([
            self.reynolds.command(
                relative_position[i][self.adjacencyMat[i].astype(bool)],
                relative_velocities[i][self.adjacencyMat[i].astype(bool)])
            for i in range(self.NUM_DRONES)
        ])

        if self.last_reynolds_command is None:
            self.last_reynolds_command = reynolds_commands

        reynolds_commands = self.last_reynolds_command * smooth_factor + reynolds_commands * (
            1 - smooth_factor)
        self.last_reynolds_command = reynolds_commands  # 更新历史信息

        # 添加z轴，将z轴reynolds_command 设置为0
        reynolds_commands = np.hstack(
            (reynolds_commands, np.zeros((self.NUM_DRONES, 1))))
        assert reynolds_commands.shape == (self.NUM_DRONES, 3)
        return reynolds_commands

    def get_command_migration(
        self,
        neighbors: dict[set] = None,
    ):
        '''
        Args:
            neighbors: 无人机间的邻接关系
            waypoints: 从 yaml 文件中读取的所有 waypoints
        '''

        drone_poses = self.drone_states[:, 0:3]
        # z 轴速度不考虑
        drone_poses[:, 2] = 0
        return self.reynolds.get_migration_command(drone_poses)

    ################################################################################
    def _computePositionEstimation(self, AdjacencyMat, nth_drone):
        '''
        根据无人机邻接矩阵，计算 nth_drone 无人机对出现在视野中无人机的位置估计结果
        Args:
            AdjacencyMat:  Adjacency mat 
        '''
        drone_poses = self.drone_states[:, 0:3]
        poses_in_fov = drone_poses[nth_drone][AdjacencyMat[nth_drone].astype(
            bool)]  # 筛选能够观测到的无人机绝对位置

        poses_noise = []
        ######### 添加测量噪声
        for poses in poses_in_fov:
            poses += np.random.normal(loc=0,
                                      scale=self.position_noise_std[int(
                                          np.linalg.norm(poses[:2]))])
            poses_noise.append(poses)
        return poses

    def _computeAdjacencyMatFOV(self):
        '''
        在考虑 fov 的情况下, 计算无人机间观测邻接矩阵
        '''
        mat = np.array([
            self._computeFovMask(nth_drone)
            for nth_drone in range(self.NUM_DRONES)
        ])
        return mat

    ################################################################################
    def _computeFovMask(self, nth_drone):
        '''
        判断 nth_drone 的 FOV 之内有哪几个 无人机
        '''
        mask = np.zeros((self.NUM_DRONES, ))
        # 计算无人机当前在世界坐标系下的 heading 向量
        ego_heading = self._computeHeading(nth_drone)

        ########## 将 ego_heading 向指定的方向旋转指定角度，得到 fov_vector #########
        fov_vector = np.array([
            np.dot(
                R.from_euler('z', theta,
                             degrees=False).as_matrix().reshape(3, 3),
                ego_heading) for theta in self.fov_range
        ])

        ## TODO 这里的 fov_vector 在机体坐标系中，如需进行坐标变换，需进行转换

        def in_fov(point, fov_vector):
            point = point[:2]  # 仅在水平面上进行判断
            fov_vector = fov_vector[:, :2]
            # 由于 fov 可能是大于 pi 的，因此需要这个判断
            return_val = (np.cross(fov_vector[0], fov_vector[1]) *
                          np.dot(fov_vector[0], fov_vector[1])) < 0
            if ~return_val:
                fov_vector = fov_vector[::-1]
            if np.cross(fov_vector[0], point) >= 0 and np.cross(
                    point, fov_vector[1]) >= 0:
                return return_val
            return ~return_val

        for other in set(range(0, self.NUM_DRONES)) - set([nth_drone]):
            mask[other] = int(
                in_fov(
                    self.world2ego(nth_drone, self.drone_states[other, 0:3]),
                    fov_vector))

        return mask

    def _computeObs(self):
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

    def _preprocessAction(self, action):
        """
        使用 PID 控制将 action 转化为 RPM, yaw_action 后续也应该从此处产生 

        Pre-processes the action passed to `.step()` into motors' RPMs.

        Uses PID control to target a desired velocity vector.

        Parameters
        ----------
        action : ndarray
            The desired velocity input for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        rpm = np.zeros((self.NUM_DRONES, 4))
        for k in range(action.shape[0]):
            #### Get the current state of the drone  ###################
            state = self._getDroneStateVector(k)
            target_v = action[k, :]
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
                target_rpy=np.array([0, 0, state[9]]),  # keep current yaw
                target_vel=self.SPEED_LIMIT * np.abs(target_v[3]) *
                v_unit_vector  # target the desired velocity vector
            )
            rpm[k, :] = temp
        return rpm

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        int
            Dummy value.

        """
        return -1

    ################################################################################

    def _computeTerminated(self):
        """Computes the current terminated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        return False

    ################################################################################

    def _computeTruncated(self):
        """Computes the current truncated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
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
