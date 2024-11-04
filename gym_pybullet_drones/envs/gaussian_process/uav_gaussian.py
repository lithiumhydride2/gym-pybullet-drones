import numpy as np
import pandas as pd
import time
import os
import matplotlib.text
import scienceplots
import matplotlib.lines
from matplotlib import axes, pyplot as plt
from matplotlib import animation as animation
from torch import layout
from tqdm import tqdm


def add_t(X, t: float):
    """
    add timestamp for measurement
    """
    return np.concatenate((X, np.zeros((X.shape[0], 1)) + t), axis=1)


class UAVAnimation:
    """
    作为未来可视化或分析的接口
    从ros中读取数据,并进行可视化
    """

    def __init__(self, **kwargs) -> None:
        self.color_map = plt.get_cmap("Set1").colors
        self.ros_step = kwargs.get("step", 1)
        self.traj_max_length = kwargs.get("traj_max_length", 400)

    def __get_xylim(self, source: str = None, key: str = None, pedding=0.5):
        """
        after the main loop, self.curr_index is max, you can use this to get xlim and ylim
        """
        assert source is not None
        assert key is not None
        xlim = [np.min(source[key + ".x"]), np.max(source[key + ".x"])]
        ylim = [np.min(source[key + ".y"]), np.max(source[key + ".y"])]
        xlim[0] = xlim[0] - pedding
        xlim[1] = xlim[1] + pedding
        ylim[0] = ylim[0] - pedding
        ylim[1] = ylim[1] + pedding
        return xlim, ylim

    def __init_animation(self, interval=40):
        self.__init_relateive_animation(interval=interval)
        # Gaussian Process Part init -----------------------------------
        self.__init_gaussian_animation(interval=interval)
        # plt.legend()

    def __init_relateive_animation(self, interval=40):
        """
        初始化 relative 需要绘制的 animation
        interval : in millisecondgs
        """
        # traj animation init --------------------------------------------------------
        self.figure = plt.figure(num="uav_traj", figsize=[5, 5])
        self.ax = plt.axes()
        ## 仅一个无人机设置边界
        xlim, ylim = [0, 0], [0, 0]
        for pose in self.poses:
            xlim_t, ylim_t = self.__get_xylim(pose,
                                              key="pose.position",
                                              pedding=0.2)
            xlim[0] = min(xlim[0], xlim_t[0])
            ylim[0] = min(ylim[0], ylim_t[0])
            xlim[1] = max(xlim[1], xlim_t[1])
            ylim[1] = max(ylim[1], ylim_t[1])
        self.ax.set_xlim(xlim[0], xlim[1])
        self.ax.set_ylim(ylim[0], ylim[1])
        self.ax.set_xlabel("$ x/(m) $")
        self.ax.set_ylabel("$ y/(m) $")
        # init an artist to update
        self.line = []
        for index in range(self.num_uav):
            self.line.append(
                self.ax.plot([], [],
                             linewidth=2,
                             color=self.color_map[index],
                             label="UAV {}".format(index + 1))[0])
        plt.legend()
        self.traj_time_label = self.ax.set_title(
            "Time : {:2f}/$(s)$".format(0))
        # 仅从 1-2-3-4-1 连线
        self.connection_line = []
        for index in range(self.num_uav):
            self.connection_line.append(
                self.ax.plot(
                    [],
                    [],
                    linewidth=1,
                    color=self.color_map[-1],
                    linestyle="--",
                )[0])
        # first in returned list
        self.anim = animation.FuncAnimation(
            fig=self.figure,
            frames=range(self.start_index, self.curr_index,
                         self.save_step),  # that is max index
            func=self.__update_animation,
            interval=interval,  # interval is in millisecond
            blit=True,
        )
        # relative pose animation init -----------------------------------------
        self.scatter_size = 100
        self.figure_relative = plt.figure(
            num="uav_{}_relative".format(self.id))
        self.ax_relative = plt.axes()
        # 设置 xylim
        xlim_relative, ylim_relative = [float("inf"),
                                        float("-inf")], [
                                            float("inf"),
                                            float("-inf"),
                                        ]
        for other in self.other_list:
            xlim_t, ylim_t = self.__get_xylim(
                source=self.relative_pose,
                key="uav{}_to_ego_pos".format(other),
                pedding=0.3,
            )
            xlim_relative[0] = min(xlim_relative[0], xlim_t[0])
            xlim_relative[1] = max(xlim_relative[1], xlim_t[1])
            ylim_relative[0] = min(ylim_relative[0], ylim_t[0])
            ylim_relative[1] = max(ylim_relative[1], ylim_t[1])
        # 需要使 (0,0) 位于中间
        xlim_abs_max = max(abs(xlim_relative[0]), abs(xlim_relative[1]))
        ylim_abs_max = max(abs(ylim_relative[0]), abs(ylim_relative[1]))
        self.ax_relative.set_xlim(-xlim_abs_max, xlim_abs_max)
        self.ax_relative.set_ylim(-ylim_abs_max, ylim_abs_max)
        self.ax_relative.set_xlabel("X")
        self.ax_relative.set_ylabel("Y")
        # init artists to update
        self.ego_scatter = self.ax_relative.scatter(
            0.0,
            0.0,
            s=self.scatter_size,
            color=self.color_map[self.id - 1],
            marker="o",
            label=f"UAV {self.id}",
        )
        # 其他无人机真值的 scatters
        self.other_scatters = [
            self.ax_relative.scatter(
                None,
                None,
                s=self.scatter_size,
                color=self.color_map[other - 1],
                marker="o",
                label=f"UAV {other}",
            ) for other in self.other_list
        ]
        self.detection_scatters = [
            self.ax_relative.scatter(
                None,
                None,
                s=self.scatter_size,
                color=list(self.color_map[self.id - 1]) + [0.3],  # add alpha
                marker="o",
                label="Detection",
            ) for bbox_num in [0, 1]
        ]
        self.anim_relative = animation.FuncAnimation(
            fig=self.figure_relative,
            frames=range(self.start_index, self.curr_index,
                         self.save_step),  # that is max index
            func=self.__update_animation_relative,
            interval=interval,  # interval is in millisecond
            blit=True,
        )

    def __init_gaussian_animation(self, interval=40):
        """ """
        grid_size = self.GP_ground_truth.grid_size
        self.grid_xx = self.GP_ground_truth.grid[:, 0].reshape(
            grid_size, grid_size)
        self.grid_yy = self.GP_ground_truth.grid[:, 1].reshape(
            grid_size, grid_size)
        self.figure_gaussian = plt.figure(
            "gaussian_process",
            layout="compressed",
            figsize=[2.2 * self.num_uav, 2.5 * 2])
        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        nrows, ncols = 2, self.num_uav

        # 关闭xytick

        self.gaussian_anim_artists = {}
        self.gaussian_anim_axes = {}
        self.gaussian_anim_artists["time_label"] = plt.title("Time")

        ###### Ground Truth part
        self.gaussian_anim_axes["ground_truth_axes"] = plt.subplot(
            nrows, ncols, self.num_uav)
        self.gaussian_anim_axes["ground_truth_axes"].set_aspect("equal")
        plt.title("Ground truth")
        plt.xlim((-1, 1))
        plt.ylim((-1, 1))

        # heat map
        self.gaussian_anim_artists["ground_truth_mesh"] = (
            self.gaussian_anim_axes["ground_truth_axes"].pcolormesh(
                self.grid_xx,
                self.grid_yy,
                self.grid_xx,
                shading="auto",
                vmin=0,
                vmax=1))
        plt.xticks(color="w")
        plt.yticks(color="w")
        # scatter
        self.gaussian_anim_artists["ground_trurh_scatter"] = [
            self.gaussian_anim_axes["ground_truth_axes"].scatter(
                None,
                None,
                s=self.scatter_size / 6.0,
                color=self.color_map[other - 1],
                marker="o",
                label=f"UAV {other}",
            ) for other in self.other_list
        ]
        plt.xticks(color="w")
        plt.yticks(color="w")
        ###### Detection part ----------------
        self.gaussian_anim_axes["detection_axes"] = [None] * len(
            self.other_list)
        self.gaussian_anim_artists["detection_mesh"] = [None] * len(
            self.other_list)
        self.gaussian_anim_artists["detection_scatter"] = [None] * len(
            self.other_list)
        for other_index in range(len(self.other_list)):
            self.gaussian_anim_axes["detection_axes"][
                other_index] = plt.subplot(nrows, ncols, other_index + 1)
            self.gaussian_anim_axes["detection_axes"][other_index].set_aspect(
                "equal")
            plt.title(f"Detection {other_index}")
            plt.xlim((-1, 1))
            plt.ylim((-1, 1))
            # heat map
            self.gaussian_anim_artists["detection_mesh"][
                other_index] = self.gaussian_anim_axes["detection_axes"][
                    other_index].pcolormesh(self.grid_xx,
                                            self.grid_yy,
                                            self.grid_xx,
                                            shading="auto",
                                            vmin=0,
                                            vmax=1)
            plt.xticks(color="w")
            plt.yticks(color="w")
            # scatters
            self.gaussian_anim_artists["detection_scatter"][
                other_index] = self.gaussian_anim_axes["detection_axes"][
                    other_index].scatter(
                        None,
                        None,
                        s=(self.scatter_size / 6.0),
                        color=list(self.color_map[self.id - 1]) + [.3],
                        marker="o",
                        label="Detection")
            plt.xticks(color="w")
            plt.yticks(color="w")
        ################# all_pred
        # 第 2 行第 1 列
        self.gaussian_anim_axes["all_pred_axes"] = plt.subplot(
            nrows, ncols, nrows + self.num_latent_target)
        self.gaussian_anim_axes["all_pred_axes"].set_aspect(1)
        plt.title("All Detection")
        plt.xlim((-1, 1))
        plt.ylim((-1, 1))
        self.gaussian_anim_artists["all_pred_mesh"] = plt.pcolormesh(
            self.grid_xx,
            self.grid_yy,
            self.grid_xx,
            shading="auto",
            vmin=0,
            vmax=1)
        plt.xticks(color="w")
        plt.yticks(color="w")
        ##### 在此图片上叠加 fov 绘制
        self.gaussian_anim_artists["fake_fov"] = [
            plt.plot([], [], 'b-'),
            plt.plot([], [], 'b-')
        ]
        # 设置axes文字颜色
        plt.xticks(color="w")
        plt.yticks(color="w")
        ################## all_std
        self.gaussian_anim_axes["all_pred_axes"] = plt.subplot(
            nrows, ncols, nrows + self.num_uav)
        self.gaussian_anim_axes["all_pred_axes"].set_aspect(1)
        plt.title("Uncertainty")
        plt.xlim((-1, 1))
        plt.ylim((-1, 1))
        self.gaussian_anim_artists["all_std_mesh"] = plt.pcolormesh(
            self.grid_xx,
            self.grid_yy,
            self.grid_xx,
            shading="auto",
            vmin=0,
            vmax=1)
        plt.xticks(color="w")
        plt.yticks(color="w")
        self.anim_gaussian = animation.FuncAnimation(
            fig=self.figure_gaussian,
            frames=range(self.start_index, self.curr_index, self.save_step),
            func=self.__update_animation_gaussian,
            interval=interval,
            blit=True,
        )

    def __update_animation(self, frame):
        """
        update traj animation
        """
        # update self.line
        data = []
        start = frame - self.traj_max_length if frame >= self.traj_max_length else 0
        for index, line in enumerate(self.line):
            line_x_data = self.poses[index].loc[start:frame]["pose.position.x"]
            line_y_data = self.poses[index].loc[start:frame]["pose.position.y"]
            data.append([
                self.poses[index].loc[frame]["pose.position.x"],
                self.poses[index].loc[frame]["pose.position.y"]
            ])
            line.set_xdata(line_x_data)
            line.set_ydata(line_y_data)
        # 头尾相连
        data.append(data[0])
        data = np.array(data).reshape(-1, 2)
        ## 自适应设置 Traj xylim
        # self.ax.set_xlim([np.min(data[:, 0]) - 1, np.max(data[:, 0]) + 1])
        # self.ax.set_ylim([np.min(data[:, 1]) - 1, np.max(data[:, 1]) + 1])
        for index, c_line in enumerate(self.connection_line):
            c_line.set_data([], [])
            c_line.set_xdata(data[index:index + 2, 0])
            c_line.set_ydata(data[index:index + 2, 1])
        self.traj_time_label.set_text("")
        self.traj_time_label.set_text("Time : {:.2f}/$(s)$".format(
            self.pose.loc[frame, "Time"]))
        return tuple(self.line) + tuple(self.connection_line) + tuple(
            [self.traj_time_label])

    def __update_animation_relative(self, frame):
        """
        update relative traj animation
        update detection animation
        ## param
        frame: begin from self.start_index
        """
        # update other scatter
        for other_scatter, other in zip(self.other_scatters, self.other_list):
            other_scatter.set_offsets([
                self.relative_pose.loc[frame]["uav{}_to_ego_pos.{}".format(
                    other, item)] for item in ["x", "y"]
            ])
        # 找到当前的时间
        pose_time = self.pose.loc[frame, "Time"]
        detection_index = self.detection_2_curr_index.get(frame, None)
        if detection_index is None:
            # 检测保持上一个结果
            pass
        else:
            # self.detection_scatters [bbox0, bbox1]
            for bbox_index, detection_scatter in enumerate(
                    self.detection_scatters):
                detection_scatter.set_offsets([
                    self.detection.loc[detection_index][
                        "bbox{}.center.position.{}".format(bbox_index, item)]
                    for item in ["x", "y"]
                ])
        return tuple(self.other_scatters) + tuple(self.detection_scatters)

    def __update_animation_gaussian(self, frame):
        '''
        frame 的 step 为 self.save_step * ros_step
        '''
        ### time label
        time_label: matplotlib.text.Text = self.gaussian_anim_artists[
            "time_label"]
        time_label_str = "Time: {:.2f} $[s]$".format(self.pose.loc[frame,
                                                                   "Time"])
        plt.figure("gaussian_process")
        plt.suptitle(time_label_str, fontsize=8, y=0.97)
        ### Ground truth part
        GP_ground_truth = self.gaussian_anim_artists["ground_truth_mesh"]
        # y_true_lists 由 0 开始
        z: np.ndarray = self.GP_ground_truth.y_true_lists[int(
            (frame - self.start_index) / self.save_step)]
        GP_ground_truth.set_array(z.max(axis=-1))
        plt.xticks(color="w")
        plt.yticks(color="w")

        # scatter
        # Ground_truth_scatters = self.gaussian_anim_artists[
        #     "ground_trurh_scatter"]
        # for other_scatter, other_id in zip(Ground_truth_scatters,
        #                                    self.other_list):
        #     data = np.array([
        #         self.relative_pose.loc[int((frame - self.start_index) /
        #                                    (self.save_step / self.ros_step))]
        #         ["uav{}_to_ego_pos.{}".format(other_id, item)]
        #         for item in ["x", "y"]
        #     ]) / 6.0  # 归一化
        #     other_scatter.set_offsets(data)
        # plt.xticks(color="w")
        # plt.yticks(color="w")
        ### Detection part
        # mesh
        GP_detection_meshes = self.gaussian_anim_artists["detection_mesh"]
        for index, mesh in enumerate(GP_detection_meshes):
            y_pred = self.gps_means_list[int(
                (frame - self.start_index) / self.save_step)][index].squeeze()
            mesh.set_array(y_pred)
            plt.xticks(color="w")
            plt.yticks(color="w")
        GP_all_pred_mesh = self.gaussian_anim_artists["all_pred_mesh"]
        GP_all_pred_mesh.set_array(self.all_means_list[int(
            (frame - self.start_index) / self.save_step)].squeeze())
        plt.xticks(color="w")
        plt.yticks(color="w")

        GP_all_std_mesh = self.gaussian_anim_artists["all_std_mesh"]
        GP_all_std_mesh.set_array(self.all_stds_list[int(
            (frame - self.start_index) / self.save_step)].squeeze())

        plt.xticks(color="w")
        plt.yticks(color="w")

        # scatter
        # GP_detection_scatters = self.gaussian_anim_artists["detection_scatter"]
        # detection_index = self.detection_2_curr_index.get(
        #     int(frame / self.save_step), None)
        # if detection_index is not None:
        #     for index, scatter in enumerate(GP_detection_scatters):
        #         x = self.detection.loc[detection_index][
        #             f"bbox{index}.center.position.x"]
        #         y = self.detection.loc[detection_index][
        #             f"bbox{index}.center.position.y"]
        #         scatter.set_offsets(np.array([x, y], dtype=float) / 6.0)  # 归一化

        #         plt.xticks(color="w")
        #         plt.yticks(color="w")
        # else:
        #     for _, scatter in enumerate(GP_detection_scatters):
        #         scatter.set_offsets([None, None])
        #         plt.xticks(color="w")
        #         plt.yticks(color="w")
        ## FOV
        def heading2line(heading):
            line_length = 2
            # 坐标反变化
            heading = heading + np.pi / 2
            x, y = line_length * np.cos(heading), line_length * np.sin(heading)
            return [0, x], [0, y]

        Fov_display = self.gaussian_anim_artists["fake_fov"]
        Fov_display = [list[0] for list in Fov_display]
        for heading, line in zip(
                self.fake_fov_list[int(
                    (frame - self.start_index) / self.save_step)],
                Fov_display):
            # 由于plot返回值默认为 list
            line.set_data(heading2line(heading))
            plt.xticks(color="w")
            plt.yticks(color="w")
        save_time = [3000]

        for time in save_time:
            if frame == time:
                plt.savefig(f"gaussian_at_{time}.png", dpi=600)

        # 绘制 ego_scatter
        for value in self.gaussian_anim_axes.values():
            if type(value) == list:
                for var in value:
                    var.scatter(0, 0, s=6, c="red", marker="o")
                    # var.axes.xaxis.label.set_color("w")
                    # var.axes.yaxis.label.set_color("w")
            else:
                value.scatter(0, 0, s=6, c="red", marker="o")
                # value.axes.xaxis.label.set_color("w")
                # value.axes.yaxis.label.set_color("w")
        return tuple([GP_ground_truth]) + tuple(GP_detection_meshes) + tuple(
            [time_label]) + tuple([GP_all_pred_mesh]) + tuple(Fov_display)

    def save_animation(self,
                       dpi=100,
                       file_format="mp4",
                       save_traj=False,
                       save_relative=False,
                       save_gaussian=False,
                       save_step=10,
                       dir=""):
        """
        save: if false , just plot, do not save
        """
        # 获取子类属性
        self.save_step = save_step * self.ros_step

        save_path = dir + "/uav_{}/".format(self.id)
        # 如果路径不存在，创建路径
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        time_str = time.strftime("_%Y%m%d-%H%M")
        fname_templete = save_path + "{}" + time_str + "." + file_format

        # 创建动画

        self.__init_animation(interval=self.save_step * 10)  #milesconds
        if save_traj:
            self.anim.save(fname_templete.format("traj"),
                           dpi=dpi,
                           writer="ffmpeg")
            plt.clf()
        if save_relative:
            save_anim_pbar = tqdm(total=(self.curr_index - self.start_index))

            def save_anim_step(curr_frame, total_frame):
                save_anim_pbar.update(self.save_step)

            plt.clf()
            self.anim_relative.save(
                fname_templete.format("relative_pos"),
                dpi=dpi,
                writer="ffmpeg",
                progress_callback=save_anim_step,
            )

        if save_gaussian:
            ## Reset detection 获取
            save_gaussian_anim_pbar = tqdm(total=(self.curr_index -
                                                  self.start_index))

            def save_gaussian_anim_step(curr_frame, total_frame):
                save_gaussian_anim_pbar.update(self.save_step)

            self.anim_gaussian.save(
                fname_templete.format("gaussian_process"),
                dpi=dpi,
                writer="ffmpeg",
                progress_callback=save_gaussian_anim_step,
            )
            plt.clf()
        #self.gaussian_anim_artists["time_label"].set_text(None)
