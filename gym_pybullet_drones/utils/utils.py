"""General use functions.
"""
import time
import argparse
import numpy as np
from scipy.optimize import nnls

################################################################################


def sync(i, start_time, timestep):
    """Syncs the stepped simulation with the wall-clock.

    Function `sync` calls time.sleep() to pause a for-loop
    running faster than the expected timestep.

    Parameters
    ----------
    i : int
        Current simulation iteration.
    start_time : timestamp
        Timestamp of the simulation start.
    timestep : float
        Desired, wall-clock step of the simulation's rendering.

    """
    if timestep > .04 or i % (int(1 / (24 * timestep))) == 0:
        elapsed = time.time() - start_time
        if elapsed < (i * timestep):
            time.sleep(timestep * i - elapsed)


################################################################################


def str2bool(val):
    """Converts a string into a boolean.

    Parameters
    ----------
    val : str | bool
        Input value (possibly string) to interpret as boolean.

    Returns
    -------
    bool
        Interpretation of `val` as True or False.

    """
    if isinstance(val, bool):
        return val
    elif val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(
            "[ERROR] in str2bool(), a Boolean value is expected")


def in_fov(point, fov_vector):
    '''
        在水平面上判断 point 是否在 fov_vector 之内
    '''
    point = point[:2]  # 仅在水平面上进行判断
    fov_vector = fov_vector[:, :2]
    # 由于 fov 可能是大于 pi 的，因此需要这个判断
    return_val = (np.cross(fov_vector[0], fov_vector[1]) *
                  np.dot(fov_vector[0], fov_vector[1])) < 0
    if ~return_val:
        fov_vector = fov_vector[::-1]
    if np.cross(fov_vector[0], point) >= 0 and np.cross(point,
                                                        fov_vector[1]) >= 0:
        return return_val
    return ~return_val


def normalize_radians(angle):
    angle = angle % (2 * np.pi)
    if angle > np.pi:
        angle -= 2 * np.pi
    elif angle < -np.pi:
        angle += 2 * np.pi
    return angle


def point_heading(point: np.ndarray):
    point = point.squeeze()
    angle = np.arctan2(point[1], point[0])
    return normalize_radians(angle)


def yaw_to_circle(yaw):
    '''
    将任意 yaw 角以单位圆上的点来表示
    '''
    yaw = np.array(yaw).reshape(-1, 1)
    circle = np.hstack([np.cos(yaw), np.sin(yaw)])
    return circle


def circle_to_yaw(circile):
    '''
    将单位圆上任意点 circle 转换为 (-pi,pi] 内的 yaw 角
    '''
    circile = np.array(circile).reshape(-1, 2)
    yaw = np.arctan2(circile[:, 1], circile[:, 0])
    return yaw


def circle_angle_diff(p1: np.ndarray, p2: np.ndarray):
    '''
    计算以单位圆方式 表示航向角下 两个角度的偏差 的绝对值
    '''
    p1 = p1.reshape(-1, 2)
    p2 = p2.reshape(-1, 2)
    delta_cos = p1[:, 0] * p2[:, 0] + p1[:, 1] * p2[:, 1]
    delta_sin = p1[:, 1] * p2[:, 0] - p1[:, 0] * p2[:, 1]
    return np.abs(np.arctan2(abs(delta_sin), delta_cos))


def add_t(X, t: float):
    return np.concatenate((X, np.zeros((X.shape[0], 1)) + t), axis=1)
