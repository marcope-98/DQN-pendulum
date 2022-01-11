from pendulum import Pendulum
import torch
from torch import functional as F
import numpy as np
from numpy import pi
from numpy.random import randint, uniform
from DQN import DQN, ReplayMemory
import time

MINIBATCH_SIZE = 32
REPLAY_MEMORY_SIZE = 1_000_000                  # N in the algorithm pseudocode
AGENT_HISTORY_LENGTH = 4
TARGET_NETWORK_UPDATE_FREQUENCY = 10_000
GAMMA = 0.99                                    # discount factor
ACTION_REPEAT = 4
UPDATE_FREQUENCY = 4
LEARNING_RATE = 0.00025
GRADIENT_MOMENTUM = 0.95
SQUARED_GRADIENT_MOMENTUM = 0.95
MIN_SQUARED_GRADIENT = 0.01
INITIAL_EXPLORATION = 1.
FINAL_EXPLORATION = 0.1
FINAL_EXPLORATION_FRAME = 1_000_000
REPLAY_START_SIZE = 50_000
NOOP_MAX = 30




class CDPendulum:
    ''' Continuous(-state)Discrete(-action) Pendulum environment. Joint angle and velocity are continuous and torque is discretized
        with the specified step. Torque is saturated. 
        Guassian noise can be added in the dynamics. 
    '''

    # Constructor
    def __init__(self, nu=11, uMax=5, dt=5e-2, ndt=1, noise_stddev=0):
        # model variables
        
        self.pendulum = Pendulum(1, noise_stddev)
        self.pendulum.DT = dt                       # time step lenght
        self.pendulum.NDT = ndt                     # number of euler steps per integration
        
        self.nu = nu                                # number of discretization steps for joint torque
        self.umax = uMax                            # max torque
        self.DU = 2*uMax/nu                         # discretization resolution for joint torque
        # NNs
            # I imagine that the inputs are both number of actions and number of states
            # and the output is the number of actions
        self.Q = DQN(LEARNING_RATE,   self.pendulum.nx + self.nu, self.nu) # for weights 
        self.Q_target = DQN(LEARNING_RATE, self.pendulum.nx + self.nu, self.nu) # for fixed target

    # RL
    def update_Q_target(self):
        self.Q_target.load_state_dict(self.Q.state_dict())
        
    def train_step(self, state_transitions):
        current_states = torch.stack([s.state       for s in state_transitions])
        rewards =        torch.stack([s.reward      for s in state_transitions])
        actions =        torch.stack([s.action      for s in state_transitions])
        next_states =    torch.stack([s.next_state  for s in state_transitions])
        mask =           torch.stack([0 if s.done else 1 for s in state_transitions])
        
        # compute loss
        with torch.no_grad():
            qvals_next = self.Q_target(next_states).max(-1)



    # Dynamics
    # Continuous to discrete (prolly never gonna use this but whatever)
    def c2du(self, u):
        u = np.clip(u,-self.uMax+1e-3,self.uMax-1e-3)
        return int(np.floor((u+self.uMax)/self.DU))

    # Discrete to continuous
    def d2cu(self, iu):
        iu = np.clip(iu,0,self.nu-1) - (self.nu-1)/2
        return iu*self.DU


    # this enforces epsilon greedy 
    def choose_action(self, obs):
        if uniform(0,1) > self.epsilon: # be greedy
            state = torch.tensor([obs]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else: # exploration
            action = randint(self.nu) # int between 0 and self.nu

        return action

    # methods for simulation
    def step(self,iu):
        cost = self.dynamics(self.x, iu)
        return self.pendulum.obs(self.x), cost

    def render(self):
        self.pendulum.render() # prolly just leave it as default given that state is continuous

    def dynamics(self, x, iu):
        u = self.d2cu(iu)
        _, cost = self.pendulum.dynamics(x, u)
        return cost
        


