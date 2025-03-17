import numpy as np
from scipy.sparse import csgraph
import os


class FlockingMetrics:

    def __init__(self, num_uav) -> None:
        self.num_uav = num_uav
        self.kSafeDisThreshold = 1.5
        self.metric = {"connectivity": [], "union": [], "safety": []}
        self.distances = []

    def step(self, sense_graphs=None, ground_truth_pose=None):
        '''
        sense_graph: 每个无人机的感知图
        '''
        # 计算邻接矩阵
        A_matrix = np.zeros((self.num_uav, self.num_uav))

        for index, sense_graph in enumerate(sense_graphs):
            if sense_graph is None:
                continue
            for key, val in sense_graph.items():
                # 此处的 key 已经修改为从 0 开始的 index
                A_matrix[index][key] = val

        # 计算入度矩阵
        D_in_matrix = np.zeros((self.num_uav, self.num_uav))
        np.fill_diagonal(D_in_matrix, np.sum(A_matrix, axis=0))
        L = D_in_matrix - A_matrix
        eigen_value, eigen_vector = np.linalg.eig(L)
        idx = eigen_value.argsort()

        # connectivity metric
        lambda_2 = np.real(eigen_value[idx[1]]) / self.num_uav
        self.metric["connectivity"].append(lambda_2)

        # union metrics
        n_components, _ = csgraph.connected_components(A_matrix)
        union = 1 - (n_components - 1) / (self.num_uav - 1)
        self.metric["union"].append(union)

        # safety metrics
        assert len(ground_truth_pose) == self.num_uav
        n_s = 0
        for i in range(self.num_uav):
            for j in list(range(0, i)) + list(range(i + 1, self.num_uav)):
                distance = np.linalg.norm(ground_truth_pose[i] -
                                          ground_truth_pose[j])
                self.distances.append(distance)
                if distance <= self.kSafeDisThreshold:
                    n_s = n_s + 1
        safety = 1 - n_s / (self.num_uav * (self.num_uav - 1))
        self.metric["safety"].append(safety)
        return np.asarray([lambda_2, union, safety])

    def save_report(self, dir):
        result = []

        def add_result(name, result):
            result += ["## {} ".format(name)]
            result += ["- mean : {} ".format(np.mean(self.metric[name]))]
            result += ["- std : {} ".format(np.std(self.metric[name]))]
            result += ["- max : {}".format(np.max(self.metric[name]))]
            result += ["- min : {}".format(np.min(self.metric[name]))]

        for key in self.metric.keys():
            add_result(key, result)

        result += ["## {} ".format("distance")]
        result += ["- mean : {} ".format(np.mean(self.distances))]
        result += ["- std : {} ".format(np.std(self.distances))]
        result += ["- min : {} ".format(np.min(self.distances))]
        result += ["- max : {} ".format(np.max(self.distances))]

        file = dir + "/report.md"
        if not os.path.exists(dir):
            os.mkdir(dir)
        with open(file, "a") as f:
            for line in result:
                f.write(line + "\n")

        f.close()
