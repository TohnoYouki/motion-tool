import torch
import numpy as np
from genericpath import isfile
from .task.pvred import PvredTask
from .dataset.sample_dataset import MotionSampleDataset
from .architecture.pytorch.default_template import DefaultPipeline

setting = {
    'device': 'cuda',
    'batch': 128,
    'epoch': 50000,
    'input_dim': 63,
    'hidden_dim': 1024,
    'rnn_layer': 1,
    'dropout': 0.1,
    'input_length': 50,
    'output_length': 25,
    'start_sample': 16,
    'train_data_path': '/home/tohnoyouki/Desktop/Motion/data/traindata/train.npy',
    'test_data_path': '/home/tohnoyouki/Desktop/Motion/data/traindata/test.npy',
    'load_path': '/home/tohnoyouki/Desktop/Motion/network/bestpoint1.pth.tar',
    'check_path': '/home/tohnoyouki/Desktop/Motion/network/checkpoint.pth.tar',
    'best_path': '/home/tohnoyouki/Desktop/Motion/network/bestpoint.pth.tar'
}

train_setting = {
    'Gradient Checkpoint': False,
    'Auto Mixed Precision': [False, 1024],
    'Gradient Clipping': [True, 0.1],
    'Scheduler': [True, 'StepLR', [30000, 0.1]],
    'Early Stopping': [False, [1.02, 10]],
    'Gradient Accumulation': 1,
    'Optimizer': [1e-4, 1e-4]
}

def load_data(path):
    data = np.load(path, allow_pickle = True).item()
    motions = list(data.values())
    number = len(motions) * 2
    motions = [motions[i // 2][i % 2::2] for i in range(number)]
    motions = [x[setting['start_sample']:] for x in motions]
    std = [np.std(np.std(x, axis = 0), axis = 1) for x in motions]
    mask = np.any(np.array(std) > 1e-4, axis = 0)
    motions = [x[:, mask, :] for x in motions]
    motions = [x.reshape(len(x), -1) for x in motions]
    length = setting['input_length'] + setting['output_length'] + 1
    dataset = MotionSampleDataset(motions, length)
    return dataset

def main():
    train_data = load_data(setting['train_data_path'])
    test_data = load_data(setting['test_data_path'])
    train_data = train_data.dataloader(True, setting['batch'])
    test_data = test_data.dataloader(False, setting['batch'])
    
    task = PvredTask(setting['input_dim'], setting['hidden_dim'],
                     setting['rnn_layer'], setting['dropout'], 
                     setting['input_length'], setting['output_length'])
    task = task.to_device(setting['device'])
    train_pipeline = DefaultPipeline(task, train_setting)
    if isfile(setting['load_path']):
        state = torch.load(setting['load_path'])
        train_pipeline.load_state_dict(state)
    train_pipeline.default_train_process(train_data, test_data, 
        setting['epoch'], setting['check_path'], setting['best_path'])