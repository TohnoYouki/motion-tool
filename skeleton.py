import quaternion
import numpy as np
from bvh import BVH
from scipy.spatial.transform import Rotation as R

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
        if self.rotations.shape == 3:
            self.rotations = self.euler_to_rotation(orientation)
        self.end_site = end_site

    def __getitem__(self, item):
        assert(isinstance(item, slice))
        root_pos = self.root_pos[item].copy()
        rotations = self.rotations[item].copy()
        assert(len(root_pos) > 0)
        return Skeleton(self.joints, self.parent, root_pos,
                rotations, self.offset, self.end_site, self.order)

    def __children__(self):
        children = []
        for i in range(len(self.joints)):
            children.append([])
            if self.parent[i] != -1:
                children[self.parent[i]].append(i)
        return children

    def euler_to_rotation(self, eulers):
        eulers = np.array(eulers)[..., [2, 1, 0]]
        shape, order = eulers.shape, self.order[::-1]
        eulers = eulers.reshape(-1, 3)
        rotations = R.from_euler(order, eulers, degrees = True)
        quaternions = np.array(rotations.as_quat())[:, [3, 0, 1, 2]]
        quaternions = quaternions.reshape(*shape[:-1], 4)
        quaternions = quaternion.as_quat_array(quaternions)
        return quaternions

    def rotation_to_euler(self, rotations):
        quaternions = quaternion.as_float_array(rotations)
        shape = quaternions.shape
        quaternions = quaternions.reshape(-1, 4)[:, [1, 2, 3, 0]]
        rotation = R.from_quat(quaternions)
        eulers = rotation.as_euler(self.order[::-1], degrees = True)
        return np.array(eulers)[:, [2, 1, 0]].reshape(*shape[:-1], 3)

    def __rotation_apply__(self, rotations, vectors):
        vectors = vectors[..., np.newaxis]
        matrixs = quaternion.as_rotation_matrix(rotations)
        return np.matmul(matrixs, vectors)[..., 0]

    def all_position(self):
        frames = np.arange(self.frame)
        for i, joint in enumerate(self.joints):
            pos = self.position(joint, frames)[:, np.newaxis, :]
            if i == 0: result = pos
            else: result = np.concatenate((result, pos), 1)
        return result

    def position(self, joint, frames):
        if isinstance(joint, str):
            joint = self.joints.index(joint)
        position = np.zeros((len(frames), 3))
        while self.parent[joint] != -1:
            position += self.offset[joint][np.newaxis, :]
            joint = self.parent[joint]
            rotation = self.rotations[frames, joint]
            position = self.__rotation_apply__(rotation, position)
        position += self.root_pos[frames, :]
        return np.array(position)

    @staticmethod
    def generate(bvh):
        assert(isinstance(bvh, BVH))
        index, order, channel_datas = 0, '', {}
        for i, channel in enumerate(bvh.channels):
            for name in channel:
                if channel_datas.get(name, None) is None:
                    if 'rotation' in name: order += name[0].lower()
                    channel_datas[name] = []
                data = bvh.data_block[:, index]
                channel_datas[name].append((i, data))
                index += 1
        names = ['Xposition', 'Yposition', 'Zposition']
        pos = [channel_datas[x][0][1][:, np.newaxis] for x in names]
        positions = np.concatenate((pos[0], pos[1], pos[2]), 1)
        names = [axis.upper() + 'rotation' for axis in order]
        rots = [[x[1] for x in channel_datas[name]] for name in names]
        rotations = np.array(rots).transpose(2, 1, 0)
        joints = [x for x in bvh.names]
        parents = np.array(bvh.parents).copy()
        offsets = np.array(bvh.offsets).copy()
        endsite = [x.copy() if x is not None else None 
                   for x in bvh.end_offsets]
        return Skeleton(joints, parents, positions, 
                        rotations, offsets, endsite, order)