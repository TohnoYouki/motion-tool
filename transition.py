import numpy as np
import quaternion as quater
from motionclip import MotionClip

class Transition:
    @staticmethod
    def __clerp_position__(position_a, position_b, factors):
        position_a = position_a[np.newaxis, :]
        position_b = position_b[np.newaxis, :]
        factors = factors[:, np.newaxis]
        return position_a * (1 - factors) + position_b * factors

    @staticmethod
    def __slerp_rotation__(rotation_a, rotation_b, factors):
        assert(isinstance(factors, np.ndarray))
        return quater.slerp(rotation_a, rotation_b, 0, 1, factors)

    @staticmethod
    def __slerp_transition__(clip_a, clip_b, tran_num):
        rotation = []
        factors = np.array([i / (tran_num + 1) for i in range(tran_num + 2)])
        params = (clip_a.root_pos[-1], clip_b.root_pos[0], factors)
        root_pos = Transition.__clerp_position__(*params)[1:-1]
        for i in range(len(clip_a.rotations[-1])):
            params = (clip_a.rotations[-1, i], clip_b.rotations[0, i], factors)
            rotation.append(Transition.__slerp_rotation__(*params))
        rotation = np.array(rotation).transpose(1, 0)[1:-1]
        clip_tran = MotionClip(clip_a.skeleton, root_pos, rotation)
        return MotionClip.concat((clip_a, clip_tran, clip_b))
    
    @staticmethod
    def __slerp_concat__(clip_a, clip_b, window_a, window_b):
        window =  window_a + window_b
        src_clip, dst_clip = clip_a[:-window_a], clip_b[window_b:]
        params = (clip_a[-window_a:], clip_b[:window_b], window - 2)
        tran_clip = Transition.__slerp_transition__(*params)
        return MotionClip.concat((src_clip, tran_clip, dst_clip))