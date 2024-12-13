from enum import Enum


class DroneModel(Enum):
    """Drone models enumeration class."""

    CF2X = "cf2x"  # Bitcraze Craziflie 2.0 in the X configuration
    CF2P = "cf2p"  # Bitcraze Craziflie 2.0 in the + configuration
    RACE = "racer"  # Racer drone in the X configuration
    VSWARM_QUAD = "vswarm_quad/vswarm_quad_dae"  # vswarm_quad 其实是 craziflie 换皮， 可以使用这二者控制器


################################################################################


class Physics(Enum):
    """Physics implementations enumeration class."""

    PYB = "pyb"  # Base PyBullet physics update
    DYN = "dyn"  # Explicit dynamics model
    PYB_GND = "pyb_gnd"  # PyBullet physics update with ground effect
    PYB_DRAG = "pyb_drag"  # PyBullet physics update with drag
    PYB_DW = "pyb_dw"  # PyBullet physics update with downwash
    PYB_GND_DRAG_DW = "pyb_gnd_drag_dw"  # PyBullet physics update with ground effect, drag, and downwash


################################################################################


class ImageType(Enum):
    """Camera capture image type enumeration class."""

    RGB = 0  # Red, green, blue (and alpha)
    DEP = 1  # Depth
    SEG = 2  # Segmentation by object id
    BW = 3  # Black and white


################################################################################


class ActionType(Enum):
    """Action type enumeration class."""
    RPM = "rpm"  # RPMS
    PID = "pid"  # PID control
    VEL = "vel"  # Velocity input (using PID control)
    YAW = "yaw"  # yaw angle in world coordinate, 连续性的
    YAW_DIFF = "yaw_diff"  # diff 形式的 yaw_angle
    YAW_RATE = "yaw_rate"  # 角速度的形式
    YAW_RATE_DISCRETE = "yaw_rate_discrete"  # 离散的角速度选择
    ONE_D_RPM = "one_d_rpm"  # 1D (identical input to all motors) with RPMs
    ONE_D_PID = "one_d_pid"  # 1D (identical input to all motors) with PID control


################################################################################


class ObservationType(Enum):
    """Observation type enumeration class."""
    KIN = "kin"  # Kinematic information (pose, linear and angular velocities)
    RGB = "rgb"  # RGB camera capture in each drone's POV
    GAUSSIAN = "gaussian"  # gaussian process of each drone in world and non rotation corrdinate
    IPP = "ipp"  # 以 IPP 形式获得观测
    POSE = "pose"  # 直接相对位置估计结果


class FOVType(Enum):
    '''
        无人机 fov 的角度， 以无人机 ego_heading 为中心，数组两项分别为：
        [negative, positive]
    '''
    SINGLE = [-1.1757, 1.1757]
    DOUBLE = [-2.77999, 1.20765]
    FOUR = [0, 0]
