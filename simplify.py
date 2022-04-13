import numpy as np
import numpy.linalg as linalg
from skeleton import Skeleton

class SkeletonSimplify:
    def __init__(self, skeleton, threshold):
        assert(isinstance(skeleton, Skeleton))
        self.skeleton = skeleton
        self.threshold = threshold

    def __child_num__(self, joint):
        children = self.skeleton.__children__()
        end_site = self.skeleton.end_sites
        return len(children[joint]) + len(end_site[joint])

    def __add_endsite__(self):
        skeleton = self.skeleton
        childrens = skeleton.__children__()
        for i in range(len(skeleton.joints)):
            if len(childrens[i]) == 0 and len(skeleton.end_sites[i]) == 0:
                skeleton.end_sites[i] = np.zeros((1, 3))

    def __remove_redundant_endsite__(self):
        change, skeleton = False, self.skeleton
        for i in range(len(skeleton.joints)):
            if len(skeleton.end_sites[i]) <= 0: continue
            endsite = skeleton.end_sites[i]
            zero_offset = np.sum(endsite ** 2, -1) > self.threshold
            skeleton.end_sites[i] = endsite[zero_offset]
            if not np.all(zero_offset): change = True
        return change

    def __remove_redundant_joint__(self, joint):
        skeleton = self.skeleton
        for i in range(len(skeleton.parents)):
            if skeleton.parents[i] > joint:
                skeleton.parents[i] -= 1
        del skeleton.joints[joint]
        skeleton.offsets = np.delete(skeleton.offsets, joint, 0)    
        skeleton.parents = np.delete(skeleton.parents, joint, 0)
        skeleton.rotations = np.delete(skeleton.rotations, joint, 1)
        del skeleton.end_sites[joint]

    def __collapse_redundant_joint__(self, i):
        skeleton = self.skeleton
        parent = skeleton.parent[i]
        skeleton.parent[i] = skeleton.parent[parent]
        skeleton.offset[i] = skeleton.offset[parent]
        rotation = skeleton.rotations[:, parent]
        rotation *= skeleton.rotations[:, i]
        skeleton.rotations[:, i] = rotation
        self.__remove_redundant_joint__(parent)

    def remove_redundant_skeleton(self):
        i, skeleton = 0, self.skeleton
        while i < len(skeleton.joints):
            if self.__remove_redundant_endsite__(): continue
            zero = skeleton.parent[i] != -1
            zero &= linalg.norm(skeleton.offset[i]) <= self.threshold
            zerochild = self.__child_num__(i) == 0
            onechild = self.__child_num__(skeleton.parent[i]) == 1
            if zero and zerochild: self.__remove_redundant_joint__(i)
            elif zero and onechild: self.__collapse_redundant_joint__(i)
            i = 0 if zero and (zerochild or onechild) else i + 1
        self.__add_endsite__()