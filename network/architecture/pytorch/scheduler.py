import torch.nn.utils.clip_grad as clip
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class SchedulerOptimizer():
    def __init__(self, optimizer, scheduler, gradclip):
        assert(isinstance(optimizer, Optimizer))
        self.optimizer = optimizer
        self.scheduler, self.clip_grad_norm = None, None
        if scheduler is not None:
            assert(isinstance(scheduler, _LRScheduler))
            assert(scheduler.optimizer is optimizer)
            self.scheduler = scheduler
        if gradclip[0]: self.clip_grad_norm = gradclip[1]

    def __clip_gradient__(self):
        if self.clip_grad_norm is None: return 
        for group in self.optimizer.param_groups:
            max_norm = self.clip_grad_norm
            clip.clip_grad_norm_(group['params'], max_norm)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self, gradscaler = None):
        if gradscaler is not None:
            gradscaler.unscale_(self.optimizer)
        self.__clip_gradient__()
        if gradscaler is not None:
            gradscaler.step(self.optimizer)
        else: self.optimizer.step()

    def update(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state['scheduler'])
    
    def state_dict(self, state = {}):
        state['optimizer'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state['scheduler'] = self.scheduler.state_dict()
        return state