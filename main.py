# User files
import conf as conf
from cdPendulum import CDPendulum
from Network import Network, ReplayMemory
# pytorch
import torch
import torch.nn.functional as F
import torch.optim as optim
# numpy
import numpy as np
from numpy.random import randint, uniform
import time
    
test = CDPendulum()
Q_gepetto_gui = Network(test.nx, test.nu**test.nv)

epsilon = conf.INITIAL_EXPLORATION
counter = 0
ctg_global = []
best_ctg = -np.inf


# Initialize environment
model      = CDPendulum()
# Initilalize replay memory D to capacity N
rb         = ReplayMemory(conf.REPLAY_MEMORY_SIZE) 
# Initialize action-value function Q with random weights \theta
Q_function = Network(model.nx, model.nu**model.nv)
# Initialize target action-value function \hat{Q} 
Q_target   = Network(model.nx, model.nu**model.nv)
#                                                 with weights \theta^- = \theta
Q_target.copy(Q_function)
optimizer = optim.Adam(Q_function.parameters(), lr=conf.LEARNING_RATE)

        
# for episode 1, M do
for episode in np.arange(0, conf.N_EPISODES):
    print(episode)
    # Initialize sequence s_1 = \{x_1\} ...
    ctg = 0.
    gamma = 1.
    s = model.reset().reshape(model.nx)
    # for t = 1, T do 
    for t in np.arange(0, conf.MAX_EPISODE_LENGTH):
        counter += 1
        # With probability \epsilon select a random action a_t
        # otherwise select a_t = \argmin_a Q(s_t, a; \theta)
        if uniform(0,1) < epsilon:
            a = randint(model.nu, size=model.nv)
            a_encoded = model.encode_action(a)
        else:
            ''' TODO: this does not work with double pendulum, can be solved recursively '''
            with torch.no_grad(): # needed for inference
                a_encoded = int(torch.argmin(Q_function(torch.Tensor(s)))) # first model.nu elements refer to the values (u_1i, u_2) with u_2 fixed
            a = model.decode_action(a_encoded)

        # Execute action a_t in the emulator and observe reward r_t and image x_{t+1}
        s_next, r = model.step(a)
        ctg += gamma * r
        gamma *= conf.GAMMA
        done = 0 if (episode == conf.N_EPISODES - 1) else 1 # mask to check if the next state is the one that terminates the episode
        s_next = s_next.reshape(model.nx)
        #print(s_next)
        # Set s_{t+1} = s_t , a_t, x_{t+1}
        # Store experience (s_t, a_t, r_t, s_{t+1}) in D
        rb.push(s, a_encoded, r, s_next, done)

        s = s_next
        # Sample random minibatch of experiences (s_j, a_j, r_j, s_{j+1}) from D
        if len(rb) >= conf.REPLAY_START_SIZE and counter % conf.REPLAY_SAMPLE_STEP == 0:
            counter = 0
            minibatch = rb.sample(conf.MINIBATCH_SIZE)
        else:
            continue

        # Set y_i = r_j if episode terminates at step j+1
        # Set y_i = r_j + \gamma \max_{a'} \hat{Q}(s_{j+1}, a'; \theta^-) otherwise
        t_states      = torch.stack([torch.Tensor(m.state) for m in minibatch])          # 32, 2, self.nv
        t_actions     = torch.LongTensor([m.action for m in minibatch] )        # 32, 1 (the action is encoded)
        t_rewards     = torch.Tensor([-m.reward for m in minibatch])       # 32, 1
        t_next_states = torch.stack([torch.Tensor(m.next_state) for m in minibatch])     # 32, 1
        t_dones       = torch.Tensor([m.done for m in minibatch])        # 32, 2, self.nv
        
        with torch.no_grad(): # needed for inference
            q_target_values = Q_target(t_next_states).min(-1)[0] # min or max, i believe min  # this is MINIBATCH_SIZE x model.nu**model.nv
        
        optimizer.zero_grad()
        q_values = Q_function(t_states)
        q_selector = F.one_hot(t_actions, model.nu**model.nv) # 32, 121 
        
        # Perform a gradient descent step on (y_i - Q(s_j, a_j; \theta))^2 with respect to the weights \theta
        loss = ( ( t_rewards - torch.sum(q_selector * q_values, -1) + conf.GAMMA*t_dones*(q_target_values) )**2 ).mean()
        loss.backward()
        optimizer.step()
        
        # Every C steps reset \hat{Q} = Q
        #        if (counter == conf.NETWORK_RESET):
        #    counter = 0                         # by resetting to 0 the next iteration will change it back to 1 and correctly count to 4
    if ctg > best_ctg:
        torch.save(Q_function.state_dict(), "model/model.pth")
        best_ctg = ctg
    
    if (episode % conf.GEPETTO_GUI_VISUAL == 0):
        Q_gepetto_gui.load_state_dict(torch.load("model/model.pth"))
        s_simu = test.reset(x=np.array([[np.pi],[0.]])).reshape(test.nx)
        test.render()
        time.sleep(0.5)
        for _ in np.arange(0, conf.MAX_EPISODE_LENGTH):
            test.render()
            with torch.no_grad(): # needed for inference
                a_simu = int(torch.argmin(Q_gepetto_gui(torch.Tensor(s_simu)))) # first model.nu elements refer to the values (u_1i, u_2) with u_2 fixed
            a_simu = model.decode_action(a_simu)
            s_next_simu, _ = test.step(a_simu)
            s_simu = s_next_simu.reshape(test.nx)
            time.sleep(0.01)


    Q_target.copy(Q_function)  
    #print(episode, best_ctg)
    # end for
    epsilon = max(conf.FINAL_EXPLORATION, 
                               np.exp(-conf.EXPLORATION_DECAY*episode))
# end for

# final simulation
Q_function.load_state_dict(torch.load("model/model.pth"))
test = CDPendulum()
s = test.reset(x=np.array([[np.pi],[0.]])).reshape(test.nx)
for _ in np.arange(0, conf.MAX_EPISODE_LENGTH):
    test.render()
    with torch.no_grad(): # needed for inference
        a_encoded = int(torch.argmin(Q_function(torch.Tensor(s)))) # first model.nu elements refer to the values (u_1i, u_2) with u_2 fixed
    a = model.decode_action(a_encoded)
    s_next, _ = test.step(a)
    s = s_next.reshape(test.nx)
    time.sleep(0.001)