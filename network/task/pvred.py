import torch
import torch.nn as nn
from ..architecture.pytorch.task import Task
from ..utils.convert import rotvec2quaternion
from ..utils.embedding import position_embedding

class PvredNetwork(nn.Module):
    def __init__(self, input, hidden, output, layers, dropout):
        super(PvredNetwork, self).__init__()
        self.gru = nn.GRU(input, hidden, layers)
        self.linear = nn.Linear(hidden, output)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        if hidden is None:
            output, hidden = self.gru(input)
        else: output, hidden = self.gru(input, hidden)
        output = self.dropout(output)
        output = self.linear(output)
        return output, hidden

class PvredTask(Task):
    def __init__(self, input, hidden, layers, dropout, input_l, output_l):
        self.input_length = input_l
        self.output_length = output_l
        self.embedding = position_embedding(input, input_l + output_l + 1)
        network = PvredNetwork(input * 3, hidden, input, layers, dropout)
        super(PvredTask, self).__init__([network])
    
    def __set_checkpoint__(self, state): pass

    def __predict_motion__(self, motion, velocity, hidden, frame):
        embedding = self.embedding[frame:frame + len(motion)]
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, motion.size(1), 1)
        embedding = embedding.to(self.device)
        feature = torch.cat((motion, velocity, embedding), dim = -1)  
        result, hidden = self.models[0](feature, hidden)
        return result + motion, hidden      

    def __output__(self, motion, length):
        args = (motion[1:], motion[1:] - motion[:-1], None, 0)
        pred, hidden = self.__predict_motion__(*args)
        result = torch.zeros(length + 1, *pred.shape[1:])
        result = result.to(self.device)
        result[0], result[1] = motion[-1], pred[-1] 
        for i in range(length - 1):
            position = result[i + 1:i + 2]
            velocity = position - result[i:i + 1]
            args = (position, velocity, hidden, len(motion) + i - 1)
            pred, hidden = self.__predict_motion__(*args)
            result[i + 2:i + 3] = pred
        return result[1:]

    def forward(self, motion):
        motion = motion.to(self.device).permute(1, 0, 2).float()
        input = motion[:self.input_length + 1]
        target = motion[self.input_length + 1:]
        criterion = nn.L1Loss()
        predict = self.__output__(input, len(target))
        target = rotvec2quaternion(target)
        predict = rotvec2quaternion(predict)
        padding = torch.zeros((25, target.shape[1], 44)).to(self.device)
        predict = torch.cat((predict, padding), dim = -1)
        target = torch.cat((target, padding), dim = -1)
        loss = criterion(predict, target)
        return [], loss