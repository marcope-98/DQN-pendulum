# User files
import conf as conf

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# numpy
import numpy as np
# other
from collections import * 
import random


# ------------------------ Replay buffer -----------------------
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# ----------------------- Neural Network------------------------

class Network(nn.Module):
    def __init__(self, input_size, output_size):
        super(Network, self).__init__() # calls to constructor of base class
        # general attributes
        self.i_dim = input_size         # input dim
        self.h_dim = [16,32,64,64]      # hidden layer dims
        self.o_dim = output_size        # output dim (aka number of actions)

        self.network = nn.Sequential(
            nn.Linear(self.i_dim, self.h_dim[0]),
            nn.ReLU(),
            nn.Linear(self.h_dim[0], self.h_dim[1]),
            nn.ReLU(),
            nn.Linear(self.h_dim[1], self.h_dim[2]),
            nn.ReLU(),
            nn.Linear(self.h_dim[2], self.h_dim[3]),
            nn.ReLU(),
            nn.Linear(self.h_dim[3], self.o_dim)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=conf.LEARNING_RATE)
        

    # forward propagation
    def forward(self, x):
        return self.network(x)

    # copy weights and bias
    def copy(self, other):
        self.load_state_dict(other.state_dict())

