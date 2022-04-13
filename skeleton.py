import numpy as np
from bvh import BVH
from utils import rotation_apply

class Skeleton:
    def __init__(self, joints, parents, offsets, end_sites, order = 'zyx'):
        self.order = order
        self.joints = [x for x in joints]
        self.offsets = offsets.copy()
        self.parents = parents.copy()
        self.end_sites = [x.copy() for x in end_sites]

    def __children__(self):
        children = []
        for i in range(len(self.joints)):
            children.append([])
            if self.parents[i] != -1:
                children[self.parents[i]].append(i)
        return children

    def position(self, joint, root_pos, rotations):
        frames = len(root_pos)
        if isinstance(joint, str):
            joint = self.joints.index(joint)
        assert(joint >= 0 and joint < len(self.joints))
        position = np.zeros((len(frames), 3))
        while self.parents[joint] != -1:
            position += self.offsets[joint][np.newaxis, :]
            joint = self.parents[joint]
            rotation = rotations[joint]
            position = rotation_apply(rotation, position)
        position += root_pos
        return np.array(position)

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