import numpy as np
from bvh import BVH
import quaternion as quater
from skeleton import Skeleton

class SkeletonTransition:
    @staticmethod
    def __clerp_rotation__(rotation_a, rotation_b, factors):
        return rotation_a * (1 - factors) + rotation_b * factors

    @staticmethod
    def __slerp_rotation__(rotation_a, rotation_b, factors):
        assert(isinstance(factors, np.ndarray))
        return quater.slerp(rotation_a, rotation_b, 0, 1, factors)

if __name__ == '__main__':
    bvh = BVH.load('/home/tohnoyouki/Desktop/CGF/CGF2021-v2/rawdata/data/Vasso_Tired-BVH.bvh')
    skeleton = Skeleton.generate(bvh)
    #skeleton[-1]
    print(skeleton[3:20:2].frame)