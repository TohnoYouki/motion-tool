import shutil
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from .process import train_epoch
from .recoder import TrainRecoder
from .pipeline import TrainPipeline
from .optimizer import OptimizationWrap
from .scheduler import SchedulerOptimizer

class DefaultPipeline(TrainPipeline):
    def __init__(self, task, args):
        params = task.parameters()
        init_lr = args['Optimizer'][0]
        scheduler = self.default_scheduler(init_lr, args['Scheduler'])
        clip_setting = args['Gradient Clipping']
        optimizer = self.default_optimizer(params, init_lr, scheduler,
                                      args['Optimizer'], clip_setting)
        optimizer = SchedulerOptimizer(optimizer, scheduler, None)
        amp = args['Auto Mixed Precision']
        gradient_accumulation = args['Gradient Accumulation']
        optimizer = OptimizationWrap(optimizer, amp, gradient_accumulation)
        earlystop = args['Early Stopping']
        params = (None, earlystop[0], earlystop[1][0], earlystop[1][1])
        self.recoder = TrainRecoder(*params)
        super(DefaultPipeline, self).__init__(optimizer, task)

    def default_scheduler(self, lr, args):
        if not args[0]: return None
        if args[1] == 'MultiStepLR':
            milestones, gamma = args[2][0], args[2][1]
            scheduler = optim.lr.MultiStepDecay(learning_rate = lr,
                            milestones = milestones, gamma = gamma)
        elif args[1] == 'StepLR':
            step_size, gamma = args[2][0], args[2][1]
            scheduler = optim.lr.StepDecay(learning_rate = lr,
                              step_size = step_size, gamma = gamma)
        return scheduler

    def default_optimizer(self, params, lr, scheduler, optim_setting, clip_setting):
        weight_decay, clip = optim_setting[1], None
        if clip_setting[0]:
            clip_grad_norm = clip_setting[1]
            clip = nn.ClipGradByValue(clip_grad_norm, -clip_grad_norm)
        learn_rate = lr if scheduler is None else scheduler
        optimizer = optim.Adam(parameters = params, grad_clip = clip,
            learning_rate = learn_rate, weight_decay = weight_decay)   
        return optimizer

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
            paddle.save(state, checkpath)
            if ifbest: shutil.copyfile(checkpath, bestpath)
            if stop: break
