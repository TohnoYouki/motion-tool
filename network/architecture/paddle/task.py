import paddle.nn as nn

class Task(nn.Layer):
    def __init__(self, models):
        super(Task, self).__init__()
        self.models = nn.LayerList(models)

    def to_device(self, device):
        self.device = device
        return self.to(device)
    
    def __set_checkpoint__(self, state):
        raise NotImplementedError

    def forward(self, data):
        raise NotImplementedError

    def evaluate(self, data):
        return self.forward(data)

    def test(self, data, index):
        return self.forward(data)

    def load_state_dict(self, state):
        for i, model in enumerate(self.models):
            model.set_state_dict(state['model_' + str(i)])

    def state_dict(self, state = {}):
        for i, model in enumerate(self.models): 
            state['model_' + str(i)] = model.state_dict()
        return state