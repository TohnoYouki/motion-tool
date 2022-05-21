import torch.nn as nn
import torch.utils.checkpoint as cp

class CheckpointWrap(nn.Module):
    def __init__(self, module):
        super(CheckpointWrap, self).__init__()
        self.use_checkpoint = False
        self.module = module
    
    def forward(self, *features):
        forward_fn = self.module.forward
        if self.use_checkpoint and self.training:
            return cp.checkpoint(forward_fn, *features)
        else: return forward_fn(*features)