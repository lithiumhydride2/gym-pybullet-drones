"""Script demonstrating the joint use of velocity input.

The simulation is run by a `VelocityAviary` environment.

Example
-------
In a terminal, run as:

    $ python pid_velocity.py

Notes
-----
The drones use interal PID control to track a target velocity.

"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import yaml

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary

DEFAULT_DRONE = DroneModel("vswarm_quad/vswarm_quad")
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 5
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False


def run(drone=DEFAULT_DRONE,
        gui=DEFAULT_GUI,
        num_drones=4,
        waypoints_file=None,
        record_video=DEFAULT_RECORD_VIDEO,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB):
    #### Initialize the simulation #############################
    INIT_XYZS = np.array([[.0, .0, x] for x in range(num_drones)])  # 横一字排列
    INIT_RPYS = np.zeros((num_drones, 3))  # 偏航角初始化为 0
    PHY = Physics.PYB

    #### Create the environment ################################
    env = VelocityAviary(drone_model=drone,
                         num_drones=num_drones,
                         initial_xyzs=INIT_XYZS,
                         initial_rpys=INIT_RPYS,
                         physics=Physics.PYB,
                         neighbourhood_radius=np.inf,
                         pyb_freq=simulation_freq_hz,
                         ctrl_freq=control_freq_hz,
                         gui=gui,
                         record=record_video,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         use_reynolds=True)

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()
    DRONE_IDS = env.getDroneIds()  # 6架无人机为 [1,2,3,4,5,6]

    #### Compute number of control steps in the simlation ######
    PERIOD = duration_sec
    NUM_WP = control_freq_hz * PERIOD
    wp_counters = np.array([0 for i in range(4)])

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=4,
                    output_folder=output_folder,
                    colab=colab)

    #### 读取 way_points
    with open(waypoints_file) as f:
        waypoints = yaml.load(f, yaml.FullLoader)  # list 类型

    #### Run the simulation ####################################
    action = np.zeros((4, 4))
    START = time.time()
    for i in range(0, int(duration_sec * env.CTRL_FREQ)):

        ############################################################
        # for j in range(3): env._showDroneLocalAxes(j)

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        #### Compute control for the current way point #############
        for j in range(4):
            action[j, :] = TARGET_VEL[j, wp_counters[j], :]

        #### Go to the next way point and loop #####################
        for j in range(4):
            wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP -
                                                                     1) else 0

        #### Log the simulation ####################################
        for j in range(4):
            logger.log(drone=j,
                       timestamp=i / env.CTRL_FREQ,
                       state=obs[j],
                       control=np.hstack(
                           [TARGET_VEL[j, wp_counters[j], 0:3],
                            np.zeros(9)]))

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Plot the simulation results ###########################
    logger.save_as_csv("vel")  # Optional CSV save
    if plot:
        logger.plot()


if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(
        description='Velocity control example using VelocityAviary')
    parser.add_argument('--drone',
                        default=DEFAULT_DRONE,
                        type=DroneModel,
                        help='Drone model (default: vswarm_quad/vswarm_quad)',
                        metavar='',
                        choices=DroneModel)
    parser.add_argument('--num_drones',
                        default=6,
                        type=int,
                        help="Num of drones (default: 6)")

    parser.add_argument('--gui',
                        default=DEFAULT_GUI,
                        type=str2bool,
                        help='Whether to use PyBullet GUI (default: True)',
                        metavar='')
    parser.add_argument('--record_video',
                        default=DEFAULT_RECORD_VIDEO,
                        type=str2bool,
                        help='Whether to record a video (default: False)',
                        metavar='')
    parser.add_argument(
        '--plot',
        default=DEFAULT_PLOT,
        type=str2bool,
        help='Whether to plot the simulation results (default: True)',
        metavar='')
    parser.add_argument(
        '--user_debug_gui',
        default=DEFAULT_USER_DEBUG_GUI,
        type=str2bool,
        help=
        'Whether to add debug lines and parameters to the GUI (default: False)',
        metavar='')
    parser.add_argument(
        '--obstacles',
        default=DEFAULT_OBSTACLES,
        type=str2bool,
        help='Whether to add obstacles to the environment (default: False)',
        metavar='')
    parser.add_argument('--simulation_freq_hz',
                        default=DEFAULT_SIMULATION_FREQ_HZ,
                        type=int,
                        help='Simulation frequency in Hz (default: 240)',
                        metavar='')
    parser.add_argument('--control_freq_hz',
                        default=DEFAULT_CONTROL_FREQ_HZ,
                        type=int,
                        help='Control frequency in Hz (default: 48)',
                        metavar='')
    parser.add_argument(
        '--duration_sec',
        default=DEFAULT_DURATION_SEC,
        type=int,
        help='Duration of the simulation in seconds (default: 5)',
        metavar='')
    parser.add_argument('--output_folder',
                        default=DEFAULT_OUTPUT_FOLDER,
                        type=str,
                        help='Folder where to save logs (default: "results")',
                        metavar='')
    parser.add_argument(
        '--colab',
        default=DEFAULT_COLAB,
        type=bool,
        help='Whether example is being run by a notebook (default: "False")',
        metavar='')
    # waypoint 的路径
    abs_path = os.path.dirname(os.path.abspath(__file__))
    waypoints_path = "../config/waypoints/square.yaml"
    waypoints_path = os.path.join(abs_path, waypoints_path)

    parser.add_argument('--waypoints_file',
                        default=waypoints_path,
                        type=str,
                        metavar='')

    ARGS = parser.parse_args()
    run(**vars(ARGS))
