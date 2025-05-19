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
import argparse
from datetime import datetime
import numpy as np
import pybullet as p
import supersuit as ss
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from pettingzoo.test import parallel_api_test
from gym_pybullet_drones.utils.utils import sync, str2bool, get_commit_id
from gym_pybullet_drones.envs.FlockingAviaryIPPmarl import FlockingAviaryIPPmarl
from gym_pybullet_drones.models.IPPActorCriticPolicy import IPPActorCriticPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, CallbackList, CheckpointCallback
from gym_pybullet_drones.utils.enums import DroneModel, ActionType, ObservationType, FOVType
from gym_pybullet_drones.envs.IPPArguments import IPPArg

DEFAULT_DRONE = DroneModel("vswarm_quad/vswarm_quad_dae")
DEFAULT_GUI = IPPArg.DEFAULT_GUI
DEFAULT_USER_DEBUG_GUI = IPPArg.DEFAULT_USER_DEBUG_GUI

DEFAULT_RECORD_VIDEO = False
DEFAULT_PLOT = True
DEFAULT_OBSTACLES = False
DEFAULT_SIMULATION_FREQ_HZ = 120
DEFAULT_CONTROL_FREQ_HZ = 60
DEFAULT_DURATION_SEC = 30
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_CONTROL_BY_RL_MASK = IPPArg.CONTROL_BY_RL_MASK
DEFAULT_FLIGHT_HEIGHT = 2.0
DEFAULT_COLAB = False
DEFAULT_NUM_DRONE = IPPArg.NUM_DRONE

DEFAULT_OBS_TYPE = ObservationType.IPP
DEFAULT_ACT_TYPE = ActionType.IPP_YAW

DEFAULT_FOV_CONFIG = IPPArg.FOV_CONFIG
DEFAULT_FLOCKING_FREQ = IPPArg.FLOCKIN_FREQ
DEFAULT_DECISION_FREQ = IPPArg.DECISION_FREQ
DEFAULT_RANDOM_POINT = IPPArg.RANDOM_POINT

vec_env_class = IPPArg.VEC_ENV_CLS


def pettingzoo_to_sb3(env):
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, base_class="stable_baselines3")
    env = VecMonitor(env)
    return env


def learn(drone=DEFAULT_DRONE,
          gui=DEFAULT_GUI,
          num_drones=DEFAULT_NUM_DRONE,
          record_video=DEFAULT_RECORD_VIDEO,
          plot=DEFAULT_PLOT,
          user_debug_gui=DEFAULT_USER_DEBUG_GUI,
          obstacles=DEFAULT_OBSTACLES,
          simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
          control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
          flocking_freq_hz=DEFAULT_FLOCKING_FREQ,
          decision_freq_hz=DEFAULT_DECISION_FREQ,
          duration_sec=DEFAULT_DURATION_SEC,
          output_folder=DEFAULT_OUTPUT_FOLDER,
          default_flight_height=DEFAULT_FLIGHT_HEIGHT,
          colab=DEFAULT_COLAB,
          fov_config=DEFAULT_FOV_CONFIG,
          obs=DEFAULT_OBS_TYPE,
          act=DEFAULT_ACT_TYPE,
          control_by_RL_mask=DEFAULT_CONTROL_BY_RL_MASK,
          random_point=DEFAULT_RANDOM_POINT,
          continue_train=None):

    filename = os.path.join(
        output_folder, 'save-' + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename + "/")

    # env config
    INIT_XYZS = np.array([[x * 2.8, .0, DEFAULT_FLIGHT_HEIGHT]
                          for x in range(num_drones)])  # 横一字排列
    INIT_RPYS = np.array([[0, 0, 0] for x in range(num_drones)])  # 偏航角初始化为 0
    INIT_RPYS[-1][-1] = np.pi
    env_kwargs = dict(drone_model=drone,
                      num_drones=num_drones,
                      control_by_RL_mask=control_by_RL_mask,
                      initial_xyzs=INIT_XYZS,
                      initial_rpys=INIT_RPYS,
                      pyb_freq=simulation_freq_hz,
                      flocking_freq_hz=flocking_freq_hz,
                      decision_freq_hz=decision_freq_hz,
                      ctrl_freq=control_freq_hz,
                      gui=gui,
                      user_debug_gui=user_debug_gui,
                      default_flight_height=default_flight_height,
                      fov_config=fov_config,
                      obs=obs,
                      act=act,
                      random_point=random_point,
                      waypoint_name=IPPArg.WAYPOINT_FILE_NAME
                      )  # 定义 action space and observation space

    train_env = FlockingAviaryIPPmarl(**env_kwargs)
    train_env = pettingzoo_to_sb3(train_env)

    # 这里 train_env 和 eval_env 的 pyplot 可能会发生冲突
    env_kwargs['user_debug_gui'] = False
    env_kwargs['gui'] = False
    eval_env = FlockingAviaryIPPmarl(**env_kwargs)
    eval_env = pettingzoo_to_sb3(eval_env)

    #### check the environment's spaces
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    ### train the model
    # use default policy
    model = PPO(policy=IPPActorCriticPolicy,
                env=train_env,
                verbose=1,
                learning_rate=3e-4,
                tensorboard_log=filename + '/tb/',
                batch_size=64,
                n_steps=2048,
                ent_coef=0.2)
    # n_steps 为交互 step 后， 更新 policy
    # ent_coef： maximum entropy regularized objective 的系数，scalable 声称能在完全分布式的环境下增强 weak cooperation 的能力
    if continue_train is not None:
        print("Continue training from ", continue_train)
        model.load(continue_train, env=train_env)
    callback_on_best = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=int(1e2), min_evals=int(1e3), verbose=1)

    eval_call_back = EvalCallback(eval_env=eval_env,
                                  callback_on_new_best=callback_on_best,
                                  verbose=1,
                                  best_model_save_path=filename + '/',
                                  log_path=filename + '/',
                                  eval_freq=int(1e3),
                                  deterministic=True,
                                  render=False)
    check_point_call_back = CheckpointCallback(save_freq=int(1e4),
                                               save_path=filename +
                                               '/checkpoints/',
                                               save_replay_buffer=False)
    callback = CallbackList([eval_call_back, check_point_call_back])

    # save commit id and config first
    # 读取 IPPArguments.py 中的参数
    with open(filename + "/config.txt", "w") as f:
        f.write(str(IPPArg.__dict__))

    commit_id_path = os.path.join(filename, "commit_id.txt")
    with open(commit_id_path, "w") as f:
        f.write(get_commit_id())
    f.close()

    model.learn(total_timesteps=int(3e6),
                callback=callback,
                log_interval=100,
                progress_bar=True)

    model.save(filename + '/final_model.zip')
    print(filename)


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
                        default=DEFAULT_NUM_DRONE,
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
    parser.add_argument('--continue_train',
                        default=None,
                        type=str,
                        metavar='',
                        help="If none, continue train from abs path")
    # 这里需要添加 args = [] 才能使用 vscode 进行 debug
    ARGS = parser.parse_args()

    learn(**vars(ARGS))
