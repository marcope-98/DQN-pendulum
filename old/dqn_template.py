import dqn_conf as conf
from cdpendulum import CDPendulum
from DQN import DQN, ReplayMemory
import numpy as np
from numpy.random import randint, uniform


if __name__ == "__main__":
    # Initialize physical model for simulating actions and states
    model    = CDPendulum()
    # Initialize replay memory D to capacity N
    rb       = ReplayMemory(conf.REPLAY_MEMORY_SIZE)
    # Initialize action-value function Q with random weights \theta
    Q        = DQN(model.nx + model.nu, model.nu)
    # Initialize target action-value function Q_hat with weights \theta^- = \theta 
    Q_target = DQN(model.nx + model.nu, model.nu)
    Q_target.copy(Q)

    epsilon = conf.INITIAL_EXPLORATION
    # for episode = 1, M
    for epison in np.arange(1, conf.N_EPISODES):
        # initilize sequence s_1 = {x_1} and preprocessed sequence \phi_1 = \phi(s_1)
        for t in np.arange(1, conf.EPISODES_LENGTH):
            x = model.pendulum.reset()
        # with probability \epsilon select a random action a_t
            if uniform(0,1) < epsilon:
                a_t = randint(model.nu) # sample random discrete action in the range given
            # otherwise select a_t = \argmax_a Q(\phi(s_t), a; \theta)
            else:
                u = np.argmin(Q[x,:])

            # Execute action a_t in simulator and observe reward r_t and image x_t+1



    