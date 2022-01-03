from numpy.core.arrayprint import dtype_is_implied
from pendulum import Pendulum
import numpy as np
from numpy import pi
import time



class CDPendulum:
    ''' Continuous(-state)Discrete(-action) Pendulum environment. Joint angle and velocity are continuous and torque is discretized
        with the specified step. Torque is saturated. 
        Guassian noise can be added in the dynamics. 
    '''

    # Constructor
    def __init__(self, nu=11, uMax=5, dt=5e-2, ndt=1, noise_stddev=0):
        self.pendulum = Pendulum(1, noise_stddev)
        self.pendulum.DT = dt                       # time step lenght
        self.pendulum.NDT = ndt                     # number of euler steps per integration
        self.nu = nu                                # number of discretization steps for joint torque
        self.umax = uMax                            # max torque
        self.DU = 2*uMax/nu                         # discretization resolution for joint torque
    
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
        cost = self.dynamics(self.x, iu)
        return self.pendulum.obs(self.x), cost

    def render(self):
        self.pendulum.render() # prolly just leave it as default given that state is continuous

    def dynamics(self, x, iu):
        u = self.d2cu(iu)
        _, cost = self.pendulum.dynamics(x, u)
        return cost
        
