import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MotionSampleDataset(Dataset):
    def __init__(self, motions, sample_length):
        super(MotionSampleDataset, self).__init__()
        self.length = sample_length
        self.motions = motions

    def __len__(self):
        return len(self.motions)
    
    def __getitem__(self, index):
        motion = self.motions[index]
        end = len(motion) - self.length
        start = np.random.randint(0, end)
        clip = motion[start:start + self.length]
        return clip

    def dataloader(self, shuffle, batch, pin_memory = True):
        return DataLoader(self, shuffle = shuffle, 
                batch_size = batch, pin_memory = pin_memory)