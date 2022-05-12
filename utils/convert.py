import quaternion
import numpy as np
from scipy.spatial.transform import Rotation as R

class Convert:
    @staticmethod
    def rotvec_to_quaternion(rotvec):
        return quaternion.from_rotation_vector(rotvec)

    @staticmethod
    def quaternion_to_rotvec(quaternions):
        return quaternion.as_rotation_vector(quaternions)
        
    @staticmethod
    def quaternion_to_euler(quaternions, order):
        quaternions = quaternion.as_float_array(quaternions)
        shape = quaternions.shape
        quaternions = quaternions.reshape(-1, 4)[:, [1, 2, 3, 0]]
        rotation = R.from_quat(quaternions)
        eulers = rotation.as_euler(order[::-1], degrees = True)
        return np.array(eulers)[:, [2, 1, 0]].reshape(*shape[:-1], 3)  

    @staticmethod
    def euler_to_quaternion(eulers, order):
        eulers = np.array(eulers)[..., [2, 1, 0]]
        shape, order = eulers.shape, order[::-1]
        eulers = eulers.reshape(-1, 3)
        rotations = R.from_euler(order, eulers, degrees = True)
        quaternions = np.array(rotations.as_quat())[:, [3, 0, 1, 2]]
        quaternions = quaternions.reshape(*shape[:-1], 4)
        quaternions = quaternion.as_quat_array(quaternions)
        return quaternions     