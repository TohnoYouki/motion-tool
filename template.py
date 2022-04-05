import numpy as np
from bvh import BVH
from skeleton import Skeleton

class SkeletonTemplate:    
    def __init__(self, skeleton, joint_map):
        assert(isinstance(skeleton, Skeleton))
        self.skeleton = skeleton
        self.joint_map = joint_map
        self.__joint_map_init__()

    def __joint_map_init__(self):
        raise NotImplementedError