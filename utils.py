import quaternion
import numpy as np
from scipy.spatial.transform import Rotation as R

def vector_to_string(vector):
    result = ''
    for value in vector: result += str(value) + ' '
    return result[:-1]
    
def vequal(vector_a, vector_b, epsilon):
    if vector_a.shape != vector_b.shape: return False
    if len(vector_a) == 0 and len(vector_b) == 0: return True
    return np.max(np.abs(vector_a - vector_b)) < epsilon

def rotation_apply(rotations, vectors):
    vectors = vectors[..., np.newaxis]
    matrixs = quaternion.as_rotation_matrix(rotations)
    return np.matmul(matrixs, vectors)[..., 0]

def rotation_to_euler(rotations, order):
    quaternions = quaternion.as_float_array(rotations)
    shape = quaternions.shape
    quaternions = quaternions.reshape(-1, 4)[:, [1, 2, 3, 0]]
    rotation = R.from_quat(quaternions)
    eulers = rotation.as_euler(order[::-1], degrees = True)
    return np.array(eulers)[:, [2, 1, 0]].reshape(*shape[:-1], 3)  

def euler_to_rotation(eulers, order):
    eulers = np.array(eulers)[..., [2, 1, 0]]
    shape, order = eulers.shape, order[::-1]
    eulers = eulers.reshape(-1, 3)
    rotations = R.from_euler(order, eulers, degrees = True)
    quaternions = np.array(rotations.as_quat())[:, [3, 0, 1, 2]]
    quaternions = quaternions.reshape(*shape[:-1], 4)
    quaternions = quaternion.as_quat_array(quaternions)
    return quaternions