# User files
from .pendulum import Pendulum
from . import conf
# pytorch
import torch
from torch import functional as F
# numpy
import numpy as np
# other
import time

class CDPendulum():
    # Constructor
    '''dt = 0.2 ndt = 1'''
    def __init__(self, nbJoint=1, nu=11, uMax=5.0, dt=0.1, ndt=1, noise_stddev=0, withSinCos=False):  # dt = 0.05 should imply 20 fps
        self.pendulum = Pendulum(nbJoint,noise_stddev, withSinCos)
        self.pendulum.DT = dt                       # time step lenght
        self.pendulum.NDT = ndt                     # number of euler steps per integration
        self.pendulum.umax = uMax
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

    def reset(self, x=None):
        return self.pendulum.reset(x)
        
    def step(self,iu):
        u = self.d2cu(iu)
        self.x , cost = self.pendulum.step(u)
        return self.x, cost

    def render(self):
        self.pendulum.render() # prolly just leave it as default given that state is continuous

    def dynamics(self, x, iu):
        u = self.d2cu(iu)
        _, cost = self.pendulum.dynamics(x, u, display=False) 
        return cost

    def decode_action(self, a):
        res = []
        zero = self.c2du(0)
        for i in np.arange(0, self.nv -1):
            res.insert(0, zero)
        res.insert(0,a)
        return res

            
        
if __name__ == '__main__':
    test = CDPendulum()
    test.reset(x= np.array([[np.pi,0.],[0.,0.]]))
    for i in np.arange(200):
        test.render()
        time.sleep(0.05)
        test.step(np.array([0, 5]))

    


    
