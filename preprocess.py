import numpy as np
from skeleton import Skeleton

def remove_redundant_joint(skeleton, joint):
    assert(isinstance(skeleton, Skeleton))
    for i in range(len(skeleton.parent)):
        if skeleton.parent[i] > joint:
            skeleton.parent[i] -= 1
    del skeleton.joints[joint]
    skeleton.offset = np.delete(skeleton.offset, joint, 0)    
    skeleton.parent = np.delete(skeleton.parent, joint, 0)
    skeleton.rotations = np.delete(skeleton.rotations, joint, 1)
    del skeleton.end_site[joint]

def remove_redundant_skeleton(skeleton):
    assert(isinstance(skeleton, Skeleton))
    skeleton = skeleton.__skeleton_copy__()
    change = True
    while change:
        childrens = skeleton.__children__()
        change, threshold = False, 1e-10
        for i, parent in enumerate(skeleton.parent):
            if len(skeleton.end_site[i]) > 0:
                endsite = skeleton.end_site[i]
                zero_offset = np.sum(endsite ** 2, -1) > threshold
                skeleton.end_site[i] = endsite[zero_offset]
                if not np.all(zero_offset): change = True
            if np.linalg.norm(skeleton.offset[i]) > threshold: continue
            if parent == -1: continue
            endsite = skeleton.end_site
            if len(childrens[i]) == 0 and len(endsite[i]) == 0:
                remove_redundant_joint(skeleton, i)
            elif len(childrens[parent]) == 1 and len(endsite[parent]) == 0:
                skeleton.parent[i] = skeleton.parent[parent]
                skeleton.offset[i] = skeleton.offset[parent]
                rotation = skeleton.rotations[:, parent]
                rotation *= skeleton.rotations[:, i]
                skeleton.rotations[:, i] = rotation
                remove_redundant_joint(skeleton, parent)
            if len(skeleton.joints) != len(childrens): change = True
            if change: break
    childrens = skeleton.__children__()
    for i in range(len(skeleton.joints)):
        if len(childrens[i]) == 0 and len(skeleton.end_site[i]) == 0:
            skeleton.end_site[i] = np.zeros((1, 3))
    return skeleton