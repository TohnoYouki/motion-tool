import numpy as np
from bvh import BVH
from scipy.spatial.transform import Rotation as R

class Skeleton(object):
    def __init__(self, joints, parents, position, orientation, 
                       offset, end_site, order = 'zyx'):
        assert(order == 'zyx')
        self.joints = [x for x in joints]
        self.offset, self.parent = offset.copy(), parents.copy()
        self.frame, self.root_pos = len(position), position.copy()
        self.orientation = []
        for i in range(len(orientation)):
            rotations = self.euler_to_rotation(orientation[i])
            self.orientation.append(rotations)
        self.end_site = end_site

    def euler_to_rotation(self, eulers):
        rotations = []
        for euler in eulers:
            euler = [euler[2], euler[1], euler[0]]
            rotations.append(R.from_euler('yxz', euler, degrees = True))
        return rotations

    def rotation_to_euler(self, rotations):
        eulers = []
        for rotation in rotations:
            euler = rotation.as_euler('yxz', degrees = True)
            eulers.append([euler[2], euler[1], euler[0]])
        return eulers
    
    def euler_angle_orientation(self):
        return [self.rotation_to_euler(x) for x in self.orientation]

    def global_orientation(self, joint, frame):
        joint = self.joints.index(joint)
        rotation, joint = R.identity(), self.parent[joint]
        while joint != -1:
            rotation = self.orientation[frame][joint] * rotation
            joint = self.parent[joint]
        return rotation

    def add_global_rotation(self, joint, index, global_rotation):
        rotation = self.global_orientation(joint, index)
        rotation = rotation.inv() * global_rotation * rotation
        ori_rotation = self.orientation[index][joint]
        self.orientation[index][joint] = rotation * ori_rotation

    def position(self, joint, frame):
        joint = self.joints.index(joint)
        position = [0, 0, 0]
        while self.parent[joint] != -1:
            position = np.add(position, self.offset[joint])
            joint = self.parent[joint]
            position = self.orientation[frame][joint].apply(position)
        position = np.add(position, self.root_pos[frame])
        return np.array(position)

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
        names = ['Xposition', 'Yposition', 'Zposition']
        pos = [channel_datas[x][0][1][:, np.newaxis] for x in names]
        positions = np.concatenate((pos[0], pos[1], pos[2]), 1)
        names = ['Zrotation', 'Yrotation', 'Xrotation']
        rots = [[x[1] for x in channel_datas[name]] for name in names]
        rotations = np.array(rots).transpose(2, 1, 0)
        return Skeleton(bvh.names, bvh.parents, positions, rotations,
                        bvh.offsets, bvh.end_offsets, 'zyx')