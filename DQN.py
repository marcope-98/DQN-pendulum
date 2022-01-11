import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import * 
import random
'''
# hyper-parameters
minibatch_size = 32
replay_memory_size = 1_000_000
agent_history_length = 4
target_network_update_frequency = 10_000
discount_factor = 0.99
action_repeat = 4
update_frequency = 4
learning_rate = 0.00025
gradient_momentum = 0.95
squared_gradient_momentum = 0.95
min_squared_gradient = 0.01
initial_exploration = 1
final_exploration = 0.1
final_exploration_frame = 1_000_000
replay_start_size = 50_000
noop_max = 30
'''

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


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
    def __init__(self, learning_rate, input_size, output_size):
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
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        
        # misc
        #self.device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu') # if gpu is available
        #self.to(self.device)

    # forward propagation
    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    rp = ReplayMemory(5)
    rp.push(0,1,2,3, True)
    rp.push(4,5,6,7, True)
    rp.push(8,9,10,11, True)
    rp.push(12,13,14,15, True)
    rp.push(16,17,18,19, True)

    res = rp.sample(1)
    print(res)
    print(res[0].state)
    