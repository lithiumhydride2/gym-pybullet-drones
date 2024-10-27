import numpy as np
import yaml
import os


class Reynolds():
    '''
    由于 smooth_factor 的存在， 必须为每个无人机对象声明独立的 reynolds
    '''

    def __init__(self, config_file: str = "../config/reynolds/gain.yaml"):
        '''
        Args:
            config_file: config 存放的路径
        '''
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(script_dir, config_file)
        with open(file_name) as f:
            config = yaml.load(file_name, yaml.FullLoader)
        self.separation_gain = config.get("separation_gain", 1.0)
        self.cohesion_gain = config.get("cohesion_gain", 1.0)
        self.alignment_gain = config.get("alignment_gain", 1.0)
        self.smoothing_factor = config.get("smoothing_factor", 0.0)
        self.last_command = np.zeros((3, ))  # 3-D 速度

    def command(self, positions, velocities=None):
        '''
        Args:
            positions: List of relative positions to other agents.
            velocities: List of velocities of other agents (optional).
        '''
        curr_command = self.__reynolds__(positions, velocities,
                                         self.separation_gain,
                                         self.cohesion_gain,
                                         self.alignment_gain)
        smooth_command = self.smoothing_factor * self.last_command + (
            1 - self.smoothing_factor) * curr_command
        self.last_command = curr_command
        return smooth_command

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
