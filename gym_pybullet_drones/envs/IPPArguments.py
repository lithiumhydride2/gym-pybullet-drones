import numpy as np


class IPPArguments:

    def __init__(self):
        #### graph
        self.graph_size = 10
        self.k_size = 5  # knn
        self.sample_num = 10
        self.gen_range = np.deg2rad([0, 90])  # 限制采样的范围
        self.budget_range = np.array([5, 10])  # budget 为所经过路径的最大成本，在初始化时随机采样


IPPArg = IPPArguments()
