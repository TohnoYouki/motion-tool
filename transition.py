import numpy as np
from bvh import BVH
import quaternion as quater
from skeleton import Skeleton

'''
class SkeletonTransition:
    @staticmethod
    def __clerp_position__(position_a, position_b, factors):
        return position_a * (1 - factors) + position_b * factors

    @staticmethod
    def __slerp_rotation__(rotation_a, rotation_b, factors):
        assert(isinstance(factors, np.ndarray))
        return quater.slerp(rotation_a, rotation_b, 0, 1, factors)








    @staticmethod
    def __slerp_pose__(pose_a, pose_b, number):
        rot = []
        (src_pos, src_rot), (dst_pos, dst_rot) = pose_a, pose_b
        factors = [i / (number - 1) for i in range(number)]
        params = (src_pos, dst_pos, factors)
        pos = SkeletonTransition.__clerp_position__(*params)
        for i in range(len(src_rot)):
            params = (src_rot[i], dst_rot[i], factors)
            rot.append(SkeletonTransition.__slerp_rotation__(*params))
        rot = np.array(rot).transpose(1, 0)
        return (pos, rot)
    
    @staticmethod
    def __slerp_concat__(clip_a, clip_b, window_a, window_b):
        window =  window_a + window_b
        src_clip, dst_clip = clip_a[:-window_a], clip_b[window_b:]
        src_pos = clip_a.root_pos[-window_a]
        dst_pos = clip_b.root_pos[window_b - 1]
        src_rot = clip_a.rotations[-window_a]
        dst_rot = clip_b.rotations[window_b - 1]
        params = ((src_pos, src_rot), (dst_pos, dst_rot), window)
        pos, rots = SkeletonTransition.__slerp_pose__(*params)
        src_clip += 
'''


if __name__ == '__main__':
    bvh = BVH.load('/home/tohnoyouki/Desktop/CGF/CGF2021-v2/rawdata/data/Vasso_Tired-BVH.bvh')
    skeleton = Skeleton.generate(bvh)
    #test = SkeletonTransition.__slerp_rotation__(skeleton.rotations[0], skeleton.rotations[30],
    #                                      np.array([0.3, 0.4, 0.5]))
    test = [1,2,3,4,5,6]
    test2 = test[3:5]
    test2[0] = 10
    print(test2, test)