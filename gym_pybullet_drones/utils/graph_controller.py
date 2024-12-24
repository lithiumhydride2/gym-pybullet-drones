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

    def __init__(self,
                 start,
                 k_size,
                 act_type: ActionType,
                 random_sample=True):
        '''
        Args:
            start: 在图中的起点
            k_size: k 近邻算法的 k 值
            act: type of act
            random_sample: 是否随机采样，否则在 range 内均匀采样
        '''
        self.graph = Graph()
        self.k_size = k_size
        self.act_type = act_type
        if self.act_type in [
                ActionType.YAW, ActionType.YAW_DIFF, ActionType.YAW_RATE,
                ActionType.YAW_RATE_DISCRETE, ActionType.IPP_YAW
        ]:
            self.DIM = 2  # 以单位圆上的点表示 yaw 角

        self.start = np.array(start).reshape(1, self.DIM)
        self.node_coords = self.start
        self.random_sample = random_sample
        self.dijkstra_dist = []
        self.dijkstra_prev = []

    def gen_graph(self, curr_coord: np.ndarray, samp_num, gen_range):
        """
        Args:
            curr_coord: 当前在图中的坐标
            samp_num: 采样的数量
            gen_range: range of sample, in shape (DIM, )
        """

        self.dijkstra_dist = []
        self.dijkstra_prev = []
        self.graph = Graph()
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

        self.findNearestNeighbour(k=self.k_size)

        return self.node_coords, self.graph.edges

    def findNearestNeighbour(self, k):
        """
        找到 k 个最近邻, 并使用 yaw 计算距离
        并构建图， self.node_coords 到 最近邻的距离

        Args:
            k: k近邻数量
        """
        X = self.node_coords
        knn = NearestNeighbors(n_neighbors=k, metric=circle_angle_diff)
        knn.fit(X)
        distances, indices = knn.kneighbors(X)

        for i, p in enumerate(X):
            # Ignoring nearest neighbour - nearest neighbour is the point itself
            for j, neighbour in enumerate(X[indices[i][:]]):
                a = str(self.findNodeIndex(p))
                b = str(self.findNodeIndex(neighbour))
                self.graph.add_node(a)
                self.graph.add_edge(a, b, distances[i, j])

    def findNodeIndex(self, p):
        return np.where(np.linalg.norm(self.node_coords -
                                       p, axis=1) < 1e-5)[0][0]

    def plotPoints(self, points):
        x = [item[0] for item in points]
        y = [item[1] for item in points]
        plt.scatter(x, y)
        plt.colorbar()


if __name__ == "__main__":
    graph_ctrl = GraphController([0, 1], k_size=5, act_type=ActionType.YAW)
    node_coord, graph = graph_ctrl.gen_graph(np.asarray([0, 1]),
                                             10,
                                             gen_range=np.deg2rad([0, 30]))
    edge_inputs = []
    for node in graph.values():
        node_edges = list(map(int, node))
        edge_inputs.append(node_edges)
    print(edge_inputs)
