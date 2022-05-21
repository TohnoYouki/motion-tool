import math
import torch

def position_embedding(dimension, length):
    padding_dimension = dimension + dimension % 2
    embedding = torch.zeros(length, dimension).float()
    embedding.require_grad = False
    position = torch.arange(0, length).float().unsqueeze(1)
    div_term = torch.arange(0, padding_dimension, 2)
    div_term = div_term.float().unsqueeze(0)
    div_term = (div_term * -(math.log(10000) / dimension)).exp()
    term = position * div_term
    embedding[:, 0::2] = torch.sin(term)
    embedding[:, 1::2] = torch.cos(term)[:, :dimension // 2]
    return embedding