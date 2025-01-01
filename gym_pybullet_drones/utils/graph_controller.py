import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from .graph import Graph, dijkstra, to_array
from .enums import ActionType
from .utils import circle_angle_diff, yaw_to_circle, circle_to_yaw


class GraphController:
    """
    Graph Controller
    """

    def __init__(self, start, act_type: ActionType, random_sample=True):
        '''
        Args:
            start: 在图中的起点
            act: type of act
            random_sample: 是否随机采样，否则在 range 内均匀采样
        '''
        self.act_type = act_type
        if self.act_type in [
                ActionType.YAW, ActionType.YAW_DIFF, ActionType.YAW_RATE,
                ActionType.YAW_RATE_DISCRETE, ActionType.IPP_YAW
        ]:
            self.DIM = 2  # 以单位圆上的点表示 yaw 角

        self.start = np.array(start).reshape(1, self.DIM)
        self.node_coords = self.start
        self.random_sample = random_sample

    def gen_graph(self, curr_coord: np.ndarray, samp_num, gen_range):
        """
        Args:
            curr_coord: 当前在图中的坐标
            samp_num: 采样的数量
            gen_range: range of sample, in shape (DIM, )
        """
        self.node_coords = curr_coord.reshape(1, self.DIM)
        if self.random_sample:
            count = 1
            # 需要在单位圆上进行采样
            while count < samp_num:
                new_coord = yaw_to_circle(np.random.uniform(
                    .0, 2 * np.pi)).reshape(1, self.DIM)
                # 限制采样的范围
                if circle_angle_diff(
                        curr_coord,
                        new_coord[0]) < gen_range[1] and circle_angle_diff(
                            curr_coord, new_coord[0]) > gen_range[0]:
                    if count == 0:
                        self.node_coords = new_coord
                    else:
                        self.node_coords = np.concatenate(
                            (self.node_coords, new_coord), axis=0)
                    count += 1
        else:
            self.node_coords = yaw_to_circle(
                np.linspace(
                    -gen_range[1], gen_range[1], samp_num, endpoint=False) +
                circle_to_yaw(curr_coord.reshape(-1, 2)))

        ## 计算节点之间的距离
        self.distance_matrix = np.zeros((samp_num, samp_num))
        for i in range(samp_num):
            for j in range(samp_num):
                self.distance_matrix[i, j] = circle_angle_diff(
                    self.node_coords[i], self.node_coords[j])

        return self.node_coords, self.distance_matrix

    def findNodeIndex(self, p):
        return np.where(np.linalg.norm(self.node_coords -
                                       p, axis=1) < 1e-5)[0][0]


if __name__ == "__main__":
    pass
