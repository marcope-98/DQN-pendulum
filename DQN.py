import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

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



class ReplayBuffer:
    def __init__(self, mem_size):
        self.mem_size = mem_size
        self.st   = np.zeros(shape=(self.mem_size, 1))   # current state
        self.at   = np.zeros(shape=(self.mem_size, 1))   # current action
        self.rt   = np.zeros(shape=(self.mem_size, 1))   # current reward
        self.stp1 = np.zeros(shape=(self.mem_size, 1))   # next state
        self.mem_counter = 0

    def update(self, st, at, rt, stp1):
        index = self.mem_counter % self.mem_size         # get index: this will wrap around and overwrite the start of the np.array once it reaches the end
        # insert values into memory
        self.st[index] = st
        self.at[index] = at
        self.rt[index] = rt
        self.stp1[index] = stp1

        self.mem_counter += 1                           # increment counter



class DQN(nn.Module):
    def __init__(self, learning_rate, input_size, hidden_sizes, output_size):
        super(DQN, self).__init__() # calls to constructor of base class
        # general attributes
        self.i_dim = input_size
        self.h_dim = hidden_sizes
        self.o_dim = output_size        # number of actions

        # nn architecture
        self.lay_1 = nn.Linear(self.i_dim, self.h_dim[0])
        self.lay_2 = nn.Linear(self.h_dim[0], self.h_dim[1])
        self.lay_3 = nn.Linear(self.h_dim[1], self.h_dim[2])
        self.lay_4 = nn.Linear(self.h_dim[2], self.h_dim[3])
        self.lay_out = nn.Linear(self.h_dim[3], self.o_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        
        # misc
        self.device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu') # if gpu is available
        self.to(self.device)

    # forward propagation
    def forward(self, state):
        s = F.relu(self.lay_1(state))
        s = F.relu(self.lay_2(s))
        s = F.relu(self.lay_3(s))
        s = F.relu(self.lay_4(s))
        actions = self.lay_out(s)
        
        return actions

