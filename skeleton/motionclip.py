import quaternion
import numpy as np
from .bvh import BVH
from .skeleton import Skeleton
from utils.convert import Convert

class MotionClip:
    def __init__(self, skeleton, positions, rotations):
        assert(isinstance(skeleton, Skeleton))
        self.frame = len(positions)
        self.skeleton = skeleton
        self.root_pos = positions
        self.rotations = rotations

    def __setitem__(self, key, value):
        assert(isinstance(value, MotionClip))
        if isinstance(key, int):
            key = slice(key, key + 1, None)
        self.root_pos[key] = value.root_pos
        self.rotations[key] = value.rotations

    def __getitem__(self, key):
        if isinstance(key, int):
            key = slice(key, key + 1, None)
        assert(isinstance(key, slice))
        root_pos, rotations = self.root_pos[key], self.rotations[key]
        assert(len(root_pos) > 0)
        return MotionClip(self.skeleton, root_pos, rotations)  

    @staticmethod
    def concat(clips):
        assert(len(clips) >= 2)
        skeleton = clips[0].skeleton
        assert(all([skeleton == clip.skeleton for clip in clips]))
        root_pos = [clip.root_pos for clip in clips]
        rotations = [clip.rotations for clip in clips]
        root_pos = np.concatenate(root_pos)
        rotations = np.concatenate(rotations)
        return MotionClip(skeleton, root_pos, rotations) 

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
        rotations = Convert.euler_to_quaternion(rotations, skeleton.order)
        return MotionClip(skeleton, positions, rotations)

    def align(self, mode = 'dot'):
        velocity, alpha = np.zeros((len(self.rotations[0]) ,4)), 0.8
        rotations = quaternion.as_float_array(self.rotations)
        rotations[:, rotations[0, :, 0] < 0.0, :] *= -1
        for i in range(1, len(rotations)):
            prev, next = rotations[i - 1], rotations[i]
            if mode == 'dot':
                diff = np.einsum('ij, ij -> i', prev, next) < 0
            elif mode == 'distance':
                positive = np.sum((next - prev) ** 2, -1)
                negative = np.sum((next + prev) ** 2, -1)
                diff = positive > negative
            elif mode == 'velocity':
                positive = np.sum((next - prev - velocity) ** 2, -1)
                negative = np.sum((prev + velocity + next) ** 2, -1)
                diff = positive > negative
            else: assert(False)
            rotations[i, diff] = -rotations[i, diff]
            velocity *= (1 - alpha)
            velocity += (rotations[i] - rotations[i - 1]) * alpha
        self.rotations = quaternion.from_float_array(rotations)