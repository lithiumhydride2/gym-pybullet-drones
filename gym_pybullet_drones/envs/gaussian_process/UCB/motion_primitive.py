import numpy as np


class MotionPrimitive:
    " 生成运动基元，为一组轨迹 "

    def __init__(
        self,
        time_span=1.8,
        time_interval=0.3,
        num_primitive=10,
    ):
        """
        # description :
         --------------- 
        # param :
         - num_primitive: 运动基元的数量
         - time_span: 一个运动基元的时间长度 [s]
         - time_interval: 运动基元中两 segment 的时间间隔 [s]
         --------------- 
        # returns :
         --------------- 
        """
        self.num_primitive = num_primitive
        self.time_span = time_span
        self.time_interval = time_interval
        self.num_segment = int(time_span / time_interval)
        self.time_range = np.linspace(self.time_interval, self.time_span,
                                      self.num_segment)
        self.action_dim = 3  # 基元的维度， [vx,vy,heading]
        self.headings = None
        self.motion_primitive = None  # shoule in shape [num_primitive,num_segment,action_dim]

        self.__GenerateHeading()

    def __GenerateHeading(self):
        '''
        仅生成航向角的运动基元
        heading 的逆时针为正方向
        '''
        # headings = [0, 5, 10, 15, 20, -5, -10, -15, -20]
        # headings = [0, 10, 20, 30, 40, -10, -20, -30, -40]  # in degree
        headings = [0, 20, 40, 60, 80, -20, -40, -60, -80]  # in degree
        # headings = [0, 30, 60, 90, 120, -30, -60, -90, -120]  # in degree
        self.headings = np.deg2rad(headings)
        self.num_primitive = len(headings)

        self.motion_primitive = np.zeros(
            (self.num_primitive, self.num_segment, self.action_dim))

        # 遍历设置 heading angle
        for primitive_idx in range(self.num_primitive):
            # no include index zero
            self.motion_primitive[primitive_idx][:, -1] = self.headings[
                primitive_idx] / self.num_segment

    def plot():
        pass
