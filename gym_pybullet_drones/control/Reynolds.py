import numpy as np
import yaml
import os


class Reynolds():
    '''
    reynolds 模型的实现， 实现 migtaration 函数 和 migration term 的生成
    '''

    def __init__(self,
                 config_file: str = "../config/reynolds/gain.yaml",
                 waypoint_name: str = "square",
                 random_range=[-10, 10]):
        '''
        Args:
            config_file: config 存放的路径
            waypoint_name: name of way_point
            random_range: 随机采样航路点的范围
        '''
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(script_dir, config_file)
        with open(file_name) as f:
            config = yaml.load(f, yaml.FullLoader)
        self.separation_gain = config.get("separation_gain", 1.0)
        self.cohesion_gain = config.get("cohesion_gain", 1.0)
        self.alignment_gain = config.get("alignment_gain", 1.0)
        self.migration_gain = config.get("migration_gain", 1.0)
        self.acceptance_radius = config.get("acceptance_radius", 3.0)
        self.perception_radius = config.get("perception_radius", 5.0)

        waypoint_file = f"../config/waypoints/{waypoint_name}.yaml"
        waypoint_file = os.path.join(script_dir, waypoint_file)
        with open(waypoint_file) as f:
            self.waypoints = yaml.load(f, yaml.FullLoader)

        self.curr_waypoint_index = 0
        self.last_waypoint_index = -1
        self.random_range = random_range

    def command(self, positions, velocities=None):
        '''
        Args:
            positions: List of relative positions to other agents.
            velocities: List of velocities of other agents (optional).
        '''
        curr_command = self.__reynolds__(positions, velocities,
                                         self.separation_gain,
                                         self.cohesion_gain,
                                         self.alignment_gain,
                                         self.perception_radius)
        return curr_command

    def get_migration_command(self, positions, random=True):
        '''
        Args:
            positions: 当前所有无人机的位置
            random: if True, 随机选取航点
        '''
        if random:
            # update curr_waypoint
            if self.curr_waypoint_index != self.last_waypoint_index:
                self.last_waypoint_index = self.curr_waypoint_index
                self.curr_waypoint = np.random.uniform(self.random_range[0],
                                                       self.random_range[1],
                                                       size=(3, ))
                self.curr_waypoint[2] = 0  # z 轴置 0
        else:
            if self.curr_waypoint_index == len(self.waypoints):
                self.curr_waypoint_index = 0  # 开始循环
            curr_waypoint = self.waypoints[self.curr_waypoint_index]
            # 将 z 轴设置为0
            self.curr_waypoint = np.array(
                [curr_waypoint.get("x", 0.0),
                 curr_waypoint.get("y", 0.0), 0])

        position_mean = np.mean(positions, axis=0)
        distance_curr = np.linalg.norm(position_mean[:2] -
                                       self.curr_waypoint[:2])
        if distance_curr <= self.acceptance_radius:
            # print(
            #     f"[INFO] Arrived waypoint {self.curr_waypoint_index}, and to next."
            # )
            self.curr_waypoint_index += 1

        num_drones = positions.shape[0]
        migration_command = np.array([
            (self.curr_waypoint - positions[i]) /
            np.linalg.norm(self.curr_waypoint - positions[i])
            for i in range(num_drones)
        ])
        return migration_command * self.migration_gain

    def __reynolds__(self,
                     positions,
                     velocities=None,
                     separation_gain=1.0,
                     cohesion_gain=1.0,
                     alignment_gain=1.0,
                     perception_radius=None,
                     max_agents=None):
        """Reynolds flocking for a single agent.

        Args:
            positions: List of relative positions to other agents.
            velocities: List of velocities of other agents (optional).
            separation_gain: Scalar gain for separation component.
            cohesion_gain: Scalar gain for cohesion component.
            alignment_gain: Scalar gain for alignment component.
            perception_radius: Scalar metric distance.
            max_agents: Maximum number of agents to include.

        Returns:
            command: Velocity command.
    """

        positions = np.array(positions)
        if velocities is None:
            velocities = np.zeros_like(positions)
        else:
            velocities = np.array(velocities)
        num_agents, dims = positions.shape

        indices = np.arange(num_agents)
        # Filter agents by metric distance
        distances = np.linalg.norm(positions, axis=1)
        if perception_radius is not None:
            indices = distances < perception_radius
            distances = distances[indices]

        # Filter agents by topological distance
        if max_agents is not None:
            indices = distances.argsort()[:max_agents]
            distances = distances[indices]

        # Return early if no agents
        if len(distances) == 0:
            return np.zeros(dims)

        # Compute Reynolds flocking only if there is an agent in range
        positions = positions[indices]
        velocities = velocities[indices]
        dist_inv = positions / distances[:, np.newaxis]**2

        # Renolds command computations
        separation = -separation_gain * dist_inv.mean(
            axis=0)  # Sum may lead to instabilities
        cohesion = cohesion_gain * positions.mean(axis=0)
        alignment = alignment_gain * velocities.mean(axis=0)

        return separation + cohesion + alignment


if __name__ == "__main__":
    # gym_pybullet_drones/config/reynolds/gain.yaml
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(script_dir, "../config/reynolds/gain.yaml")
    with open(file=file_name) as f:
        config = yaml.load(f, yaml.FullLoader)

    print(config)
