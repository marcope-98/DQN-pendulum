# User files
import conf as conf
from cdPendulum import CDPendulum
from Network import Network, ReplayMemory
# pytorch
import torch
import torch.nn.functional as F
# numpy
import numpy as np
from numpy.random import randint, uniform
import time
    





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
exploration_prob = conf.INITIAL_EXPLORATION
'''TODO:  change this variable back when finished debugging'''
#exploration_prob = 1.
counter = 0
# for episode 1, M do
for episode in np.arange(0, conf.N_EPISODES):
    print(episode)
    # Initialize sequence s_1 = \{x_1\} ...
    prev = 10.0
    s = model.pendulum.reset().reshape(model.nx)
    # for t = 1, T do 
    for t in np.arange(0, conf.MAX_EPISODE_LENGTH):
        counter += 1
        # With probability \epsilon select a random action a_t
        if uniform(0,1) < exploration_prob:
            a = randint(model.nu, size=model.nv)
            a_encoded = model.encode_action(a)
        # otherwise select a_t = \argmin_a Q(s_t, a; \theta)
        else:
            ''' TODO: this does not work with double pendulum, can be solved recursively '''
            with torch.no_grad(): # needed for inference
                a_encoded = int(torch.argmin(Q_function(torch.Tensor(s)))) # first model.nu elements refer to the values (u_1i, u_2) with u_2 fixed
            a = model.decode_action(a_encoded)

        # Execute action a_t in the emulator and observe reward r_t and image x_{t+1}
        s_next, r = model.step(a)
        #done = 0 if (episode == conf.N_EPISODES - 1) else 1 # mask to check if the next state is the one that terminates the episode
        
        #done = 0 if (np.abs(r) < 0.001) else 1
        s_next = s_next.reshape(model.nx)
        done = 1
        if ( np.abs(s_next[0]) < 0.01 and np.abs(s_next[1]) < 0.001  and np.abs(r) < prev):
            prev = np.abs(r)
            done = 0
            print("DONE!!!")
        else:
            done = 1
        # Set s_{t+1} = s_t , a_t, x_{t+1}
        # Store experience (s_t, a_t, r_t, s_{t+1}) in D
        rb.push(s, a_encoded, r, s_next, done)

        s = s_next
        # Sample random minibatch of experiences (s_j, a_j, r_j, s_{j+1}) from D
        try:
            minibatch = rb.sample(conf.MINIBATCH_SIZE)
        except ValueError:
            # print is here just for debugging
            print("Sample larger than population or is negative")
            continue
        # Set y_i = r_j if episode terminates at step j+1
        # Set y_i = r_j + \gamma \max_{a'} \hat{Q}(s_{j+1}, a'; \theta^-) otherwise
        t_states      = torch.stack([torch.Tensor(m.state) for m in minibatch])          # 32, 2, self.nv
        t_actions     = torch.LongTensor([m.action for m in minibatch] )        # 32, 1 (the action is encoded)
        t_rewards     = torch.Tensor([m.reward for m in minibatch])       # 32, 1
        t_next_states = torch.stack([torch.Tensor(m.next_state) for m in minibatch])     # 32, 1
        t_dones       = torch.Tensor([m.done for m in minibatch])        # 32, 2, self.nv

        with torch.no_grad(): # needed for inference
            q_target_values = Q_target(t_next_states).min(-1)[0] # min or max, i believe min  # this is MINIBATCH_SIZE x model.nu**model.nv
        Q_function.optimizer.zero_grad()
        q_values = Q_function(t_states)
        q_selector = F.one_hot(t_actions, model.nu**model.nv) # 32, 121 

        # Perform a gradient descent step on (y_i - Q(s_j, a_j; \theta))^2 with respect to the weights \theta
        loss = ( ( t_rewards - torch.sum(q_selector * q_values, -1) + conf.GAMMA*t_dones*(q_target_values) )**2 ).mean()


        loss.backward()
        Q_function.optimizer.step()
        #model.render()

        # Every C steps reset \hat{Q} = Q
        if (counter == conf.NETWORK_RESET):
            counter = 0                         # by resetting to 0 the next iteration will change it back to 1 and correctly count to 4
            Q_target.copy(Q_function)
        if done == 0:
            break
# end for
    exploration_prob = max(conf.FINAL_EXPLORATION, 
                               np.exp(-conf.EXPLORATION_DECAY*episode))
    if(episode % 50 == 0):

        s_alt = model.pendulum.reset(x0=np.array([[np.pi],[0.]])).reshape(model.nx)
        for _ in np.arange(0, conf.MAX_EPISODE_LENGTH):
            model.render()
            with torch.no_grad(): # needed for inference
                a_encoded_alt = int(torch.argmin(Q_function(torch.Tensor(s_alt)))) # first model.nu elements refer to the values (u_1i, u_2) with u_2 fixed
            a_alt = model.decode_action(a_encoded_alt)
            s_next_alt, _ = model.step(a_alt)
            s_alt = s_next_alt.reshape(model.nx)
            time.sleep(0.01)
# end for

torch.save(Q_function.state_dict(), "/home/marcope/Desktop/test.pth")
test = CDPendulum()
s = test.pendulum.reset(x0=np.array([[0.],[0.]])).reshape(test.nx)
for _ in np.arange(0, conf.MAX_EPISODE_LENGTH):
    test.render()
    with torch.no_grad(): # needed for inference
        a_encoded = int(torch.argmin(Q_function(torch.Tensor(s)))) # first model.nu elements refer to the values (u_1i, u_2) with u_2 fixed
    a = model.decode_action(a_encoded)
    s_next, _ = test.step(a)
    s = s_next.reshape(test.nx)
    time.sleep(0.01)
'''
TODO:
Question 1: from the explanation of the professor it seems like we need to sample from the replay buffer every C steps and keep the same experience for these iteration,
            from the pseudocode provided it seems like we need to sample from the replay buffer every step

Question 2: when evaluating the Q_target do I have to consider every possible actions or every action among those that were sampled??

Question 3: Do we need to use SGD optimizer or Adam?

Question 4: i believe that the neural network should accept dim of states and output a Q function for each action
            However I find it difficult to implement it in case of double pendulum since i'm not sure how the network will behave
'''

'''
TODO:
 - implement exploitation in epsilon greedy
 - stack minibatch samples and compute loss
'''