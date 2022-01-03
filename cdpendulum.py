from pendulum import Pendulum
import torch
import numpy as np
from numpy import pi
from numpy.random import randint, uniform
import time

nn_opt = {
    "minibatch-size"                    : 32,
    "replay-memory-size"                : 1_000_000,
    "agent-history-length"              : 4,
    "target-network-update-frequency"   : 10_000,
    "discount-factor"                   : 0.99,
    "action-repeat"                     : 4,
    "update-frequency"                  : 4,
    "learning-rate"                     : 0.00025,
    "gradient-momentum"                 : 0.95,
    "squared-gradient-momentum"         : 0.95,
    "min-squared-gradient"              : 0.01,
    "initial-exploration"               : 1,
    "final-exploration"                 : 0.1,
    "final-exploration-frame"           : 1_000_000,
    "replay-start-size"                 : 50_000,
    "noop-max"                          : 30
}



class CDPendulum:
    ''' Continuous(-state)Discrete(-action) Pendulum environment. Joint angle and velocity are continuous and torque is discretized
        with the specified step. Torque is saturated. 
        Guassian noise can be added in the dynamics. 
    '''

    # Constructor
    def __init__(self, nu=11, uMax=5, dt=5e-2, ndt=1, noise_stddev=0, nn_opt={}):
        # model variables
        self.pendulum = Pendulum(1, noise_stddev)
        self.pendulum.DT = dt                       # time step lenght
        self.pendulum.NDT = ndt                     # number of euler steps per integration
        self.nu = nu                                # number of discretization steps for joint torque
        self.umax = uMax                            # max torque
        self.DU = 2*uMax/nu                         # discretization resolution for joint torque

        # DQN variables
        self.opt = nn_opt
        
        # NNs
        self.Q = ... # for weights 
        self.Q_target = ... # for fixed target

        



        
    
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
        
