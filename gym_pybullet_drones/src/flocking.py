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

from gym_pybullet_drones.envs.FlockingAviary import FlockingAviary

DEFAULT_DRONE = DroneModel("vswarm_quad/vswarm_quad_dae")
DEFAULT_GUI = False  # 默认不启用 gui
DEFAULT_RECORD_VIDEO = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 40
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_FLIGHT_HEIGHT = 2.0
DEFAULT_COLAB = False


def run(drone=DEFAULT_DRONE,
        gui=DEFAULT_GUI,
        num_drones=4,
        record_video=DEFAULT_RECORD_VIDEO,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        flocking_freq_hz=10,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        default_flight_height=DEFAULT_FLIGHT_HEIGHT,
        colab=DEFAULT_COLAB):
    #### Initialize the simulation #############################
    INIT_XYZS = np.array([[x * 2.5, .0, DEFAULT_FLIGHT_HEIGHT]
                          for x in range(num_drones)])  # 横一字排列
    INIT_RPYS = np.array([[0, 0, 0] for x in range(num_drones)])  # 偏航角初始化为 0
    PHY = Physics.PYB

    #### Create the environment ################################
    control_by_RL_mask = np.zeros((num_drones, ))
    control_by_RL_mask[0] = 1
    env = FlockingAviary(drone_model=drone,
                         num_drones=num_drones,
                         control_by_RL_mask=control_by_RL_mask.astype(bool),
                         initial_xyzs=INIT_XYZS,
                         initial_rpys=INIT_RPYS,
                         physics=Physics.PYB,
                         neighbourhood_radius=10,
                         pyb_freq=simulation_freq_hz,
                         ctrl_freq=control_freq_hz,
                         flocking_freq_hz=flocking_freq_hz,
                         gui=gui,
                         record=record_video,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         default_flight_height=default_flight_height,
                         use_reynolds=True)

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()
    DRONE_IDS = env.getDroneIds()  # 6架无人机为 [1,2,3,4,5,6]

    #### Compute number of control steps in the simlation ######
    PERIOD = duration_sec

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab)

    #### Run the simulation ####################################
    action = np.zeros((num_drones, 2))  # 在 flocking_freq 时刻更新 action 即可
    START = time.time()

    ##### main_loop ##########################################
    for i in range(0, int(duration_sec * env.CTRL_FREQ)):

        ############################################################
        # for j in range(3): env._showDroneLocalAxes(j)

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        #### Compute the current action#############
        action = env.computeYawActionTSP(obs)

        #### Log the simulation ####################################
        for j in range(num_drones):
            logger.log(drone=j,
                       timestamp=i / env.CTRL_FREQ,
                       state=env.drone_states[j],
                       control=np.hstack([env.target_vs[j, :3],
                                          np.zeros(9)]))

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Plot the simulation results ###########################
    # logger.save_as_csv("vel")  # Optional CSV save
    if plot:
        logger.plot()
        logger.plot_traj()


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
    parser.add_argument('--flocking_freq_hz',
                        default=10,
                        type=int,
                        help='Flocking frequency in Hz (default: 10)',
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

    ARGS = parser.parse_args()
    run(**vars(ARGS))
