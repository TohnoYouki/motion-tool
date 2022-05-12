import numpy as np
from skeleton.bvh import BVH
from utils.utils import vequal

class Skeleton:
    epsilon = 1e-8

    def __init__(self, joints, parents, offsets, end_sites = None, order = 'zyx'):
        self.order = order
        self.joints = [x for x in joints]
        self.offsets = offsets.copy()
        self.parents = parents.copy()
        if end_sites is None:
            end_sites = np.ones(len(parents), bool)
            for parent in parents:
                if parent != -1: end_sites[parent] = False
            end_sites = [np.zeros((1,3)) if end_sites[i] else np.array([]) 
                         for i in range(len(end_sites))]
        self.end_sites = [x.copy() for x in end_sites]

    def copy(self):
        return Skeleton(self.joints, self.parents, 
               self.offsets, self.offsets, self.order)

    def __child_num__(self, joint):
        children = self.__children__()
        end_site = self.end_sites
        return len(children[joint]) + len(end_site[joint])

    def __children__(self):
        children = []
        for i in range(len(self.joints)):
            children.append([])
            if self.parents[i] != -1:
                children[self.parents[i]].append(i)
        return children

    @staticmethod
    def generate(bvh):
        assert(isinstance(bvh, BVH))
        order = ''
        for name in bvh.channels[0]:
            if 'rotation' in name: order += name[0].lower()
        assert(len(order) == 3)
        joints = [x for x in bvh.names]
        parents = np.array(bvh.parents).copy()
        offsets = np.array(bvh.offsets).copy()
        endsite = [x.copy() if x is not None else None 
                   for x in bvh.end_offsets]
        return Skeleton(joints, parents, offsets, endsite, order)

    def __eq__(self, other):
        if self is other: return True
        if self.order != other.order: return False
        if len(self.joints) != len(other.joints): return False
        if any([self.joints[i] != other.joints[i] 
                for i in range(len(self.joints))]): return False
        if not vequal(self.offsets, other.offsets, self.epsilon) or \
           not vequal(self.parents, other.parents, self.epsilon): return False
        if len(self.end_sites) != len(other.end_sites): return False
        for i, endsite in enumerate(self.end_sites):
            oendsite = other.end_sites[i]
            if not vequal(endsite, oendsite, self.epsilon): return False
        return True