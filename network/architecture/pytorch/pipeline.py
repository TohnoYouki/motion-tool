from .task import Task
from .optimizer import OptimizationWrap

class TrainPipeline():
    def __init__(self, optimizer, task):
        self.task = task
        assert(isinstance(task, Task))
        self.optimizer = optimizer
        assert(isinstance(optimizer, OptimizationWrap))

    def update(self):
        self.optimizer.update()

    def step(self, data):
        self.task.train()
        train_fn = self.task.forward
        reports, loss = self.optimizer.calculate(train_fn, data)
        self.optimizer.step(loss)
        return reports, loss

    def load_state_dict(self, state):
        self.task.load_state_dict(state)
        self.optimizer.load_state_dict(state)

    def state_dict(self, state = {}):
        state = self.task.state_dict(state)
        state = self.optimizer.state_dict(state)
        return state