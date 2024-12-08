import numpy as np
import torch
from stable_baselines3.ppo import PPO
from flocking import *

# override
DEFAULT_GUI = True
DEFAULT_USER_DEBUG_GUI = True


def main():
    filename = "/home/lih/fromgit/gym-pybullet-drones/gym_pybullet_drones/src/results/save-12.06.2024_17.54.31"
    model_path = filename + '/best_model.zip'
    model = PPO.load(model_path)
    INIT_XYZS = np.array([[x * 2.5, .0, DEFAULT_FLIGHT_HEIGHT]
                          for x in range(DEFAULT_NUM_DRONE)])  # 横一字排列

    INIT_RPYS = np.array([[0, 0, 0]
                          for x in range(DEFAULT_NUM_DRONE)])  # 偏航角初始化为 0
    control_by_rl_mask = np.zeros((DEFAULT_NUM_DRONE, ))
    control_by_rl_mask[0] = 1

    env_kwargs = dict(drone_model=DEFAULT_DRONE,
                      num_drones=DEFAULT_NUM_DRONE,
                      control_by_RL_mask=control_by_rl_mask.astype(bool),
                      initial_xyzs=INIT_XYZS,
                      initial_rpys=INIT_RPYS,
                      pyb_freq=DEFAULT_SIMULATION_FREQ_HZ,
                      flocking_freq_hz=DEFAULT_FLOCKING_FREQ,
                      decision_freq_hz=DEFAULT_DECISION_FREQ,
                      ctrl_freq=DEFAULT_CONTROL_FREQ_HZ,
                      user_debug_gui=True,
                      gui=True,
                      default_flight_height=DEFAULT_FLIGHT_HEIGHT,
                      fov_config=DEFAULT_FOV_CONFIG,
                      obs=DEFAULT_OBS_TYPE,
                      act=DEFAULT_ACT_TYPE,
                      random_point=False)
    test_env = FlockingAviary(**env_kwargs)

    logger = Logger(logging_freq_hz=DEFAULT_DECISION_FREQ,
                    num_drones=DEFAULT_NUM_DRONE,
                    output_folder=filename + '/test/')

    obs, info = test_env.reset(seed=42)
    start = time.time()

    # 这里使用四边形场地进行验证？
    TEST_DURATION = 20
    for i in range(TEST_DURATION * test_env.DECISION_FREQ_HZ):
        action, _states = model.predict(obs, deterministic=False)
        print("Action is : {}".format(action))
        obs, reward, terminated, truncated, info = test_env.step(action)

        for j in range(test_env.NUM_DRONES):
            logger.log(drone=j,
                       timestamp=i / test_env.DECISION_FREQ_HZ,
                       state=test_env.drone_states[j],
                       control=np.hstack(
                           [test_env.target_vs[j, :3],
                            np.zeros(9)]))
        test_env.render()
        if terminated:
            obs = test_env.reset(seed=42, options={})
        if test_env.GUI:
            sync(i, start, 1 / test_env.DECISION_FREQ_HZ)

    test_env.close()
    #### plot
    logger.plot()
    logger.plot_traj()


if __name__ == "__main__":
    main()
