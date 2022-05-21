import torch
import shutil
import torch.optim as optim
from .process import train_epoch
from .recoder import TrainRecoder
from .pipeline import TrainPipeline
from .optimizer import OptimizationWrap
from .scheduler import SchedulerOptimizer

class DefaultPipeline(TrainPipeline):
    def __init__(self, task, args):
        params = task.parameters()
        optimizer = self.default_optimizer(params, args['Optimizer'])
        scheduler = self.default_scheduler(optimizer, args['Scheduler'])
        clip = args['Gradient Clipping']
        optimizer = SchedulerOptimizer(optimizer, scheduler, clip)
        amp = args['Auto Mixed Precision']
        gradient_accumulation = args['Gradient Accumulation']
        optimizer = OptimizationWrap(optimizer, amp, gradient_accumulation)
        earlystop = args['Early Stopping']
        params = (None, earlystop[0], earlystop[1][0], earlystop[1][1])
        self.recoder = TrainRecoder(*params)
        super(DefaultPipeline, self).__init__(optimizer, task)

    def default_optimizer(self, parameters, args):
        lr, wd = args[0], args[1]
        return optim.Adam(parameters, lr = lr, weight_decay = wd)

    def default_scheduler(self, optimizer, args):
        if not args[0]: return None
        if args[1] == 'MultiStepLR':
            milestones, gamma = args[2][0], args[2][1]
            params = (optimizer, milestones, gamma)
            scheduler = optim.lr_scheduler.MultiStepLR(*params)
        elif args[1] == 'StepLR':
            step_size, gamma = args[2][0], args[2][1]
            params = (optimizer, step_size, gamma)
            scheduler = optim.lr_scheduler.StepLR(*params)
        return scheduler

    def load_state_dict(self, state):
        self.recoder.load_state_dict(state)
        super(DefaultPipeline, self).load_state_dict(state)

    def state_dict(self, state = {}):
        state = self.recoder.state_dict(state)
        return super(DefaultPipeline, self).state_dict(state)

    def default_train_process(self, traindata, testdata, epoch, checkpath, bestpath):
        self.recoder.end_epoch = epoch
        while self.recoder.epoch < self.recoder.end_epoch:
            params = (self, self.recoder, traindata, testdata)
            stop, ifbest, state = train_epoch(*params)
            torch.save(state, checkpath)
            if ifbest: shutil.copyfile(checkpath, bestpath)
            if stop: break