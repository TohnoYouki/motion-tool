import os
import numpy as np
from skeleton.bvh import BVH
from utils.convert import Convert
from skeleton.skeleton import Skeleton
from skeleton.motionclip import MotionClip

def read_data(path):
    joints = ['Hip', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase',
              'RightToeBaseEndSite', 'LeftUpLeg', 'LeftLeg', 'LeftFoot',
              'LeftToeBase', 'LeftToeBaseEndSite', 'Spine', 'Spine1',
              'Neck', 'Head', 'HeadEndSite', 'LeftShoulder', 'LeftArm',
              'LeftForeArm', 'LeftHand', 'LeftHandThumb', 'LeftHandThumbEndSite',
              'LeftWristEnd', 'LeftWristEndEndSite', 'RightShoulder', 'RightArm',
              'RightForeArm', 'RightHand', 'RightHandThumb', 'RightHandThumbEndSite',
              'RightWristEnd', 'RightWristEndEndSite']
    offsets = np.array([[0, 0, 0], [-132.948591, 0, 0], [0, -442.894612, 0], 
                        [0, -454.206447, 0], [0, 0, 162.767078], [0, 0, 74.999437],
                        [132.948826, 0, 0], [0, -442.894413, 0], [0, -454.206590, 0], 
                        [0, 0, 162.767426], [0, 0, 74.999948], [0, 0.1, 0],
                        [0, 233.383263, 0], [0, 257.077681, 0], [0, 121.134938, 0],
                        [0, 115.002227, 0], [0, 257.077681, 0], [0, 151.034226, 0],
                        [0, 278.882773, 0], [0, 251.733451, 0], [0, 0, 0],
                        [0, 0, 99.999627], [0, 100.000188, 0], [0, 0, 0],
                        [0, 257.077681, 0], [0, 151.031437, 0], [0, 278.892924, 0],
                        [0, 251.728680, 0], [0, 0, 0], [0, 0, 99.999888],
                        [0, 137.499922, 0], [0, 0, 0]]) * 0.1
    parents = np.array([-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
                        16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30])
    skeleton = Skeleton(joints, parents, offsets)
    lines = open(path).readlines()
    data = np.array([list(map(float, x.strip().split(','))) for x in lines])
    root_pos = data[:, :3]
    exp_map = data[:, 3:].reshape(len(data), -1, 3)
    quaternions = Convert.rotvec_to_quaternion(-exp_map)
    return MotionClip(skeleton, root_pos, quaternions)

def transfer_whole_dataset(loadpath, savepath, fps = 50):
    loadpath = loadpath + 'dataset/'
    for subject in os.listdir(loadpath):
        subpath = loadpath + subject + '/'
        for file in os.listdir(subpath):
            clip = read_data(subpath + file)
            bvh = BVH.generate(clip, fps)
            bvh_path = savepath + subject + '_' + file.split('.')[0] + '.bvh'
            bvh.bvh_save(bvh_path)

def aggregation_dataset(bvhpath):
    train_subject = [1, 6, 7, 8, 9, 11]
    test_subject = [5]
    train, test = {}, {}
    for file in os.listdir(bvhpath):
        clip = MotionClip.generate(BVH.load(bvhpath + file))
        subject, action, subact = file.split('.')[0].split('_')
        rotvec = Convert.quaternion_to_rotvec(clip.rotations)
        subject, subact = int(subject[1:]), int(subact)
        if subject in train_subject:
            train[(subject, action, subact)] = rotvec
        elif subject in test_subject:
            test[(subject, action, subact)] = rotvec
    np.save('train.npy', train)
    np.save('test.npy', test)