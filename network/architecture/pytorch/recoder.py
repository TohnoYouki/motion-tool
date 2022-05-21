import numpy as np

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = None
        self.sum = None
        self.count = 0

    def update(self, vals, n = 1):
        if self.count == 0:
            self.avg = [0 for _ in range(len(vals))]
            self.sum = [0 for _ in range(len(vals))]
        self.count += n
        for i in range(len(vals)):
            self.sum[i] += vals[i].item() * n
            self.avg[i] = self.sum[i] / self.count

class TrainRecoder():
    def __init__(self, epoch, earlystop, threshold, patience):
        self.epoch = 0
        self.end_epoch = epoch
        self.if_best = False
        self.loss_info = []
        self.best_val_loss = np.inf
        self.earlystop = earlystop
        self.threshold = threshold
        self.patience = patience
        self.counter = 0

    def exit_train(self):
        return self.earlystop and self.counter > self.patience

    def load_state_dict(self, state):
        self.epoch = state['start_epoch']
        self.end_epoch = state['end_epoch']
        self.counter = state['counter']
        self.loss_info = state['loss_info']
        self.best_val_loss = state['best_val_loss']

    def state_dict(self, state = {}):
        state['start_epoch'] = self.epoch
        state['end_epoch'] = self.end_epoch
        state['counter'] = self.counter
        state['loss_info'] = self.loss_info
        state['best_val_loss'] = self.best_val_loss
        return state

    def step(self, train_loss, val_loss):
        self.epoch += 1
        self.loss_info.append([train_loss, val_loss])
        if val_loss > self.best_val_loss * self.threshold:
            self.counter += 1
        elif val_loss < self.best_val_loss:
            self.counter = 0
        self.if_best = self.best_val_loss > val_loss
        self.best_val_loss = min(val_loss, self.best_val_loss)