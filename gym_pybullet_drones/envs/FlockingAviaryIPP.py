from .FlockingAviary import *
from .IPPArguments import IPPArg
from ..utils.graph_controller import GraphController


class FlockingAviaryIPP(FlockingAviary):
    '''
    为 Flocking Aviary 添加 IPP 建模相关内容
    '''

    def __init__(self,
                 drone_model=DroneModel.CF2X,
                 num_drones=1,
                 control_by_RL_mask=None,
                 neighbourhood_radius=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics=Physics.PYB,
                 pyb_freq=240,
                 flocking_freq_hz=10,
                 decision_freq_hz=5,
                 ctrl_freq=240,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 use_reynolds=True,
                 default_flight_height=1,
                 output_folder='results',
                 fov_config=FOVType.SINGLE,
                 obs=ObservationType.GAUSSIAN,
                 act=ActionType.YAW,
                 random_point=True):
        super().__init__(drone_model, num_drones, control_by_RL_mask,
                         neighbourhood_radius, initial_xyzs, initial_rpys,
                         physics, pyb_freq, flocking_freq_hz, decision_freq_hz,
                         ctrl_freq, gui, record, obstacles, user_debug_gui,
                         use_reynolds, default_flight_height, output_folder,
                         fov_config, obs, act, random_point)
        # IPP 属性
        self.IPPEnvs: dict[int, IPPenv] = {}
        for nth in self.control_by_RL_ID:
            self.IPPEnvs[nth] = IPPenv(yaw_start=self._computeHeading(nth)[:2],
                                       act_type=act)

    def plot_online(self):
        super().plot_online()
        # 绘制图的采样
        if self.USER_DEBUG:
            for nth in self.control_by_RL_ID:
                node_coords = self.IPPEnvs[nth].node_coords
                self.plot_online_stuff[f"gp_pred_{nth}"][1].scatter(
                    node_coords[:, 0], node_coords[:, 1], c='orchid')
                plt.pause(1e-10)


class IPPenv:

    def __init__(self, yaw_start, act_type):

        self.graph_control = GraphController(start=yaw_start,
                                             k_size=IPPArg.k_size,
                                             act_type=act_type)
        self.node_coords, self.graph = self.graph_control.gen_graph(
            curr_coord=yaw_start,
            samp_num=IPPArg.sample_num,
            gen_range=IPPArg.gen_range)


if __name__ == "__main__":
    pass
