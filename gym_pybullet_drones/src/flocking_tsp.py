import numpy as np
import torch
from stable_baselines3.ppo import PPO
from stable_baselines3.common.env_checker import check_env
# from gym_pybullet_drones.envs.FlockingAviaryIPP import FlockingAviaryIPP
from gym_pybullet_drones.envs.FlockingAviaryIPPmarl import FlockingAviaryIPPmarl
import time
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import circle_to_yaw, normalize_radians
from gym_pybullet_drones.envs.gaussian_process.UCB.tsp_base_line import TSPBaseLine
from flocking_ipp import *

## 更改 DEFAULT_ACT_TYPE
DEFAULT_ACT_TYPE = ActionType.YAW
DEFAULT_DECISION_FREQ = 1


def main():
    # 现在和 ROS 环境中同一个 model

    INIT_XYZS = np.array([[x * 2.5, .0, DEFAULT_FLIGHT_HEIGHT]
                          for x in range(DEFAULT_NUM_DRONE)])  # 横一字排列

    INIT_RPYS = np.array([[0, 0, 0]
                          for x in range(DEFAULT_NUM_DRONE)])  # 偏航角初始化为 0

    ## planner
    planners = [
        TSPBaseLine(
            num_latent_target=DEFAULT_NUM_DRONE - 1,
            fake_fov_range=DEFAULT_FOV_CONFIG.value,
            enable_exploration=False,
        ) for i in range(DEFAULT_NUM_DRONE)
    ]
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
                      random_point=DEFAULT_RANDOM_POINT,
                      waypoint_name=IPPArg.WAYPOINT_FILE_NAME)
    test_env = FlockingAviaryIPPmarl(**env_kwargs)
    # test_env = pettingzoo_to_sb3(test_env)
    filename = "/home/lih/fromgit/gym-pybullet-drones/gym_pybullet_drones/src/tsp_results"
    logger = Logger(logging_freq_hz=DEFAULT_DECISION_FREQ,
                    num_drones=DEFAULT_NUM_DRONE,
                    output_folder=filename + '/tsp/')
    obs = test_env.reset()
    start = time.time()
    TEST_DURATION = 200
    action = np.zeros((DEFAULT_NUM_DRONE, ))

    def tsp_step(obs):
        '''
        实际不需要obs
        '''
        # planner 返回三维向量
        action_diff = np.zeros((DEFAULT_NUM_DRONE, ))
        for i in range(DEFAULT_NUM_DRONE):
            action_diff[i] = planners[i].step(
                gp_wrapper=test_env.decisions[i].GP_detection,
                curr_t=test_env.curr_time,
                ego_heading=circle_to_yaw(
                    test_env._computeHeading(i)[:2].reshape(1, 2)),
                std_at_grid=None)[-1]
        return action_diff

    for i in range(TEST_DURATION * IPPArg.DECISION_FREQ):

        action += tsp_step(obs)
        action = [normalize_radians(act) for act in action]
        print("Action is : {}".format(action))
        # action in shape (num_drone,)
        obs, reward, terminated, truncateds, info = test_env.step(action)

        for j in range(IPPArg.NUM_DRONE):
            logger.log(
                drone=j,
                timestamp=i / IPPArg.DECISION_FREQ,
                state=test_env.drone_states[j],
                control=np.hstack([test_env.target_vs[j, :3],
                                   np.zeros(9)]),
                metric=np.asarray([
                    test_env.cache['UNC_metric'][j], test_env.cache['JSD'][j]
                ]))
        test_env.render()

        if info[0].get("finish_point", False):
            print("Finish point")
            ## 更新 flocking metric
            break

        if IPPArg.DEFAULT_GUI:
            sync(i, start, 1 / IPPArg.DECISION_FREQ)

    test_env.close()
    #### plot
    logger.flocking_metircs = test_env.flocking_metrics.metric
    logger.plot()
    logger.plot_traj()
    logger.save_metric()
    logger.save_to_pickle()


if __name__ == "__main__":
    main()
