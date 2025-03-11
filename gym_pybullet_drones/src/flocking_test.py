import numpy as np
import torch
from stable_baselines3.ppo import PPO
from stable_baselines3.common.env_checker import check_env
# from gym_pybullet_drones.envs.FlockingAviaryIPP import FlockingAviaryIPP
from gym_pybullet_drones.envs.FlockingAviaryIPPmarl import FlockingAviaryIPPmarl
import time
from gym_pybullet_drones.utils.Logger import Logger
from flocking_ipp import *


def main():
    # 现在和 ROS 环境中同一个 model
    filename = "/home/lih/fromgit/gym-pybullet-drones/gym_pybullet_drones/src/results/save-03.10.2025_22.01.18"
    model_path = filename + '/best_model.zip'
    INIT_XYZS = np.array([[x * 2.5, .0, DEFAULT_FLIGHT_HEIGHT]
                          for x in range(DEFAULT_NUM_DRONE)])  # 横一字排列

    INIT_RPYS = np.array([[0, 0, 0]
                          for x in range(DEFAULT_NUM_DRONE)])  # 偏航角初始化为 0

    env_kwargs = dict(drone_model=DEFAULT_DRONE,
                      num_drones=DEFAULT_NUM_DRONE,
                      control_by_RL_mask=DEFAULT_CONTROL_BY_RL_MASK,
                      initial_xyzs=INIT_XYZS,
                      initial_rpys=INIT_RPYS,
                      pyb_freq=DEFAULT_SIMULATION_FREQ_HZ,
                      flocking_freq_hz=DEFAULT_FLOCKING_FREQ,
                      decision_freq_hz=DEFAULT_DECISION_FREQ,
                      ctrl_freq=DEFAULT_CONTROL_FREQ_HZ,
                      user_debug_gui=DEFAULT_USER_DEBUG_GUI,
                      gui=DEFAULT_GUI,
                      default_flight_height=DEFAULT_FLIGHT_HEIGHT,
                      fov_config=DEFAULT_FOV_CONFIG,
                      obs=DEFAULT_OBS_TYPE,
                      act=DEFAULT_ACT_TYPE,
                      random_point=False)
    test_env = FlockingAviaryIPPmarl(**env_kwargs)
    test_env = pettingzoo_to_sb3(test_env)
    model = PPO.load(model_path, test_env)
    logger = Logger(logging_freq_hz=DEFAULT_DECISION_FREQ,
                    num_drones=DEFAULT_NUM_DRONE,
                    output_folder=filename + '/test/')
    obs = test_env.reset()
    start = time.time()
    TEST_DURATION = 100

    for i in range(TEST_DURATION * IPPArg.DECISION_FREQ):

        action, _states = model.predict(obs, deterministic=True)
        print("Action is : {}".format(action))
        obs, reward, terminated, info = test_env.step(action)

        for j in range(IPPArg.NUM_DRONE):
            pass
            # logger.log(drone=j,
            #            timestamp=i / IPPArg.DECISION_FREQ,
            #            state=test_env.drone_states[j],
            #            control=np.hstack(
            #                [test_env.target_vs[j, :3],
            #                 np.zeros(9)]))
        test_env.render()
        if terminated.any():
            obs, info = test_env.reset()
        if IPPArg.DEFAULT_GUI:
            sync(i, start, 1 / IPPArg.DECISION_FREQ)

    test_env.close()
    #### plot
    logger.plot()
    logger.plot_traj()


if __name__ == "__main__":
    main()
