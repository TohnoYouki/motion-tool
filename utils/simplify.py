import numpy as np
import numpy.linalg as linalg
from skeleton.skeleton import Skeleton

class SkeletonSimplify:
    def __init__(self, threshold):
        self.threshold = threshold

    def __remove_redundant_endsite__(self, skeleton):
        assert(isinstance(skeleton, Skeleton))
        change, skeleton = False, skeleton
        for i in range(len(skeleton.joints)):
            if len(skeleton.end_sites[i]) <= 0: continue
            endsite = skeleton.end_sites[i]
            zero_offset = np.sum(endsite ** 2, -1) > self.threshold
            skeleton.end_sites[i] = endsite[zero_offset]
            if not np.all(zero_offset): change = True
        return change

    def __add_endsite__(self, skeleton):
        assert(isinstance(skeleton, Skeleton))        
        childrens = skeleton.__children__()
        for i in range(len(skeleton.joints)):
            if len(childrens[i]) == 0 and len(skeleton.end_sites[i]) == 0:
                skeleton.end_sites[i] = np.zeros((1, 3))

    def __collapse_redundant_joint__(self, skeleton, i):
        assert(isinstance(skeleton, Skeleton))
        parent = skeleton.parents[i]
        skeleton.parents[i] = skeleton.parents[parent]
        skeleton.offsets[i] = skeleton.offsets[parent]
        self.__remove_redundant_joint__(parent)

    def __remove_redundant_joint__(self, skeleton, joint):
        assert(isinstance(skeleton, Skeleton))
        for i in range(len(skeleton.parents)):
            if skeleton.parents[i] > joint:
                skeleton.parents[i] -= 1
        del skeleton.joints[joint]
        skeleton.offsets = np.delete(skeleton.offsets, joint, 0)    
        skeleton.parents = np.delete(skeleton.parents, joint, 0)
        del skeleton.end_sites[joint]

    def remove_redundant_parts(self, skeleton):
        assert(isinstance(skeleton, Skeleton))
        i, skeleton = 0, skeleton.copy()
        while i < len(skeleton.joints):
            if self.__remove_redundant_endsite__(skeleton): continue
            zero = skeleton.parents[i] != -1
            zero &= linalg.norm(skeleton.offsets[i]) <= self.threshold
            zerochild = skeleton.__child_num__(i) == 0
            onechild = skeleton.__child_num__(skeleton.parents[i]) == 1
            if zero and zerochild: 
                self.__remove_redundant_joint__(skeleton, i)
            elif zero and onechild: 
                self.__collapse_redundant_joint__(skeleton, i)
            i = 0 if zero and (zerochild or onechild) else i + 1
        self.__add_endsite__()