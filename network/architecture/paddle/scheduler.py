from paddle.optimizer import Optimizer
from paddle.optimizer.lr import LRScheduler

class SchedulerOptimizer():
    def __init__(self, optimizer, scheduler, gradclip_norm = None):
        assert(isinstance(optimizer, Optimizer))
        self.optimizer = optimizer
        if scheduler is not None:
            assert(isinstance(scheduler, LRScheduler))
            assert(self.optimizer._learning_rate is scheduler)
            self.scheduler = scheduler
        else:
            assert(isinstance(self.optimizer._learning_rate, float))
            self.scheduler = None
        self.clip_grad_norm = gradclip_norm
        if self.clip_grad_norm is not None:
            assert(self.optimizer._grad_clip is not None)

    def clear_grad(self):
        self.optimizer.clear_grad()

    def step(self, gradscaler = None):
        if gradscaler is not None:
            gradscaler.step(self.optimizer)
        else: self.optimizer.step()

    def update(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def set_state_dict(self, state):
        self.optimizer.set_state_dict(state['optimizer'])
        if self.scheduler is not None:
            self.scheduler.set_state_dict(state['scheduler'])

    def state_dict(self, state = {}):
        state['optimizer'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state['scheduler'] = self.scheduler.state_dict()
        return state