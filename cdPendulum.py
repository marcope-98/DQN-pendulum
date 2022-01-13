# User files
from random import paretovariate
from pendulum import Pendulum
import conf as conf
# pytorch
import torch
from torch import functional as F
# numpy
import numpy as np
# other
import time

class CDPendulum():
    # Constructor
    def __init__(self,  nu=11, uMax=5, dt=5e-2, ndt=1, noise_stddev=0):
        self.pendulum = Pendulum(1, noise_stddev)
        self.pendulum.DT = dt                       # time step lenght
        self.pendulum.NDT = ndt                     # number of euler steps per integration
        self.dt = dt                                # time step
        self.nx = self.pendulum.nx                  # state dim
        self.nu = nu                                # number of discretization steps for joint torque
        self.uMax = uMax                            # max torque
        self.DU = 2*uMax/nu                         # discretization resolution for joint torque
        self.nv = self.pendulum.nv

    # Dynamics
    # Continuous to discrete (prolly never gonna use this but whatever)
    def c2du(self, u):
        u = np.clip(u,-self.uMax+1e-3,self.uMax-1e-3)
        return int(np.floor((u+self.uMax)/self.DU))

    # Discrete to continuous
    def d2cu(self, iu):
        iu = np.clip(iu,0,self.nu-1) - (self.nu-1)/2
        return iu*self.DU


    # methods for simulation
    def step(self,iu):
        cost = self.dynamics(self.pendulum.x, iu)
        return self.pendulum.obs(self.pendulum.x), cost

    def render(self):
        self.pendulum.render() # prolly just leave it as default given that state is continuous

    def dynamics(self, x, iu):
        u = self.d2cu(iu)
        _, cost = self.pendulum.dynamics(x, u, display=False) 
        return cost

    def decode_action(self, a):
        res = []
        rem = a
        for i in np.arange(0, self.nv - 1):
            res.insert(0, int(rem / self.nu**(self.nv - i - 1))) # append at the beginning
            rem = (rem % self.nu**(self.nv - i - 1))
        res.insert(0, rem)
        return res

    def encode_action(self, a):
        res = 0
        for i in np.arange(0, self.nv):
            res += a[i] * self.nu**i

        return res
            
        
if __name__ == '__main__':
    test = CDPendulum()
    a = test.decode_action(120)
    a = test.encode_action(a)
    pass