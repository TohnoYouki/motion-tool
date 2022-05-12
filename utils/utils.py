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