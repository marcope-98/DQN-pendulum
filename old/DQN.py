import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import dqn_conf as conf
from collections import * 
import random


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))


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


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__() # calls to constructor of base class
        # general attributes
        self.i_dim = input_size
        self.h_dim = [16,32,64,64]
        self.o_dim = output_size        # number of actions

        # nn architecture

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

    def copy(self, other):
        self.load_state_dict(other.state_dict())


if __name__ == "__main__":
    rp = ReplayMemory(5)
    rp.push(0,1,2,3)

    res = rp.sample(3)
    print(res)
    print(res[0].state)
    