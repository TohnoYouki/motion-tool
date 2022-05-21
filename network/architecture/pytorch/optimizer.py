from torch.cuda import amp
from .scheduler import SchedulerOptimizer

class OptimizationWrap():
    def __init__(self, optimizer, enable_amp = False, accumulate_step = 1):
        self.optimizer = optimizer
        if enable_amp:
            self.gradscaler = amp.grad_scaler.GradScaler()
        else: self.gradscaler = None
        self.step_num = 0
        self.accumulate_step = accumulate_step

    def calculate(self, function, data):
        if self.gradscaler is not None:
            with amp.autocast_mode.autocast():
                result = function(data)
        else: result = function(data)
        return result

    def __inner_step__(self):
        if self.gradscaler is not None:
            if isinstance(self.optimizer, SchedulerOptimizer):
                self.optimizer.step(self.gradscaler)
            else: self.gradscaler.step(self.optimizer)            
            self.gradscaler.update()
        else: self.optimizer.step()
        self.optimizer.zero_grad()

    def step(self, loss):
        self.step_num += 1
        self.backward(loss / self.accumulate_step)
        if self.step_num % self.accumulate_step == 0:
            self.step_num = 0
            self.__inner_step__()

    def backward(self, loss):
        if self.gradscaler is not None:
            self.gradscaler.scale(loss).backward()
        else: loss.backward()

    def update(self):
        if self.step_num != 0:
            raise NotImplementedError
        if isinstance(self.optimizer, SchedulerOptimizer):
            self.optimizer.update()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)
        if self.gradscaler is not None:
            self.gradscaler.load_state_dict(state['gradscaler'])

    def state_dict(self, state = {}):
        if isinstance(self.optimizer, SchedulerOptimizer):
            state = self.optimizer.state_dict(state)
        else: state['optimizer'] = self.optimizer.state_dict()
        if self.gradscaler is not None:
            state['gradscaler'] = self.gradscaler.state_dict()
        return state