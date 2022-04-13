import numpy as np
from bvh import BVH
from skeleton import Skeleton
from utils import euler_to_rotation, rotation_apply

class MotionClip:
    def __init__(self, skeleton, positions, rotations):
        assert(isinstance(skeleton, Skeleton))
        self.frame = len(positions)
        self.skeleton = skeleton
        self.root_pos = positions
        self.rotations = rotations

    @staticmethod
    def generate(bvh):
        assert(isinstance(bvh, BVH))
        index, channel_datas = 0, {}
        for i, channel in enumerate(bvh.channels):
            for name in channel:
                if channel_datas.get(name, None) is None:
                    channel_datas[name] = []
                data = bvh.data_block[:, index]
                channel_datas[name].append((i, data))
                index += 1
        skeleton = Skeleton.generate(bvh)
        names = ['Xposition', 'Yposition', 'Zposition']
        pos = [channel_datas[x][0][1][:, np.newaxis] for x in names]
        positions = np.concatenate((pos[0], pos[1], pos[2]), 1)
        names = [axis.upper() + 'rotation' for axis in skeleton.order]
        rots = [[x[1] for x in channel_datas[name]] for name in names]
        rotations = np.array(rots).transpose(2, 1, 0)
        rotations = euler_to_rotation(rotations, skeleton.order)
        return MotionClip(skeleton, positions, rotations)
    '''

class Skeleton:
    def __init__(self, joints, parents, position, orientation, 
                       offset, end_site, order = 'zyx'):
        self.order = order
        self.frame = len(position)
        self.joints = joints
        self.offset = offset
        self.parent = parents
        self.root_pos = position
        self.rotations = orientation
        self.end_site = end_site
    

    def __add__(self, other):
        return None

    def __iadd__(self, other):
        self.root_pos = np.concatenate((self.root_pos, other.root_pos))
        self.rotations = np.concatenate((self.rotations, other.rotations))
        return self

    def __getitem__(self, item):
        assert(isinstance(item, slice))
        root_pos, rotations = self.root_pos[item], self.rotations[item]
        assert(len(root_pos) > 0)
        return Skeleton(self.joints, self.parent, root_pos,
                rotations, self.offset, self.end_site, self.order)

    def all_position(self):
        frames = np.arange(self.frame)
        for i, joint in enumerate(self.joints):
            pos = self.position(joint, frames)[:, np.newaxis, :]
            if i == 0: result = pos
            else: result = np.concatenate((result, pos), 1)
        return result
    '''