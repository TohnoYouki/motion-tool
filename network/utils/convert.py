import torch

def rotvec2quaternion(rotvec):
    shape = rotvec.shape[:-1]
    rotvec = rotvec.reshape(-1, 3)
    theta = torch.norm(rotvec, p = 2, dim = -1, keepdim = True)
    w = torch.cos(0.5 * theta)
    xyz = torch.sin(0.5 * theta) / (theta + 1e-6) * rotvec
    quaternion = torch.cat([w, xyz], dim = -1)
    return quaternion.view(*shape, -1)