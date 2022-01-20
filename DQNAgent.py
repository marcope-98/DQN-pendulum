# User Files
from env.Network import Network, ReplayMemory, Transition
from env import CDPendulum
from env import conf
# numpy
import numpy as np
# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# others
import csv
import time
import random

class DQNAgent():
    def __init__(self):
        self.buffer = ReplayMemory(conf.REPLAY_MEMORY_SIZE)
        self.model = CDPendulum(nbJoint=conf.N_JOINTS,nu=conf.NU, uMax=conf.UMAX, dt=conf.DT, ndt=conf.NDT, noise_stddev=conf.NOISE, withSinCos=conf.WITHSinCos)
        self.ns = self.model.nx + self.model.nv if (self.model.pendulum.withSinCos) else self.model.nx

        self.Q_function = Network(self.ns, self.model.nu)
        self.optimizer = optim.Adam(self.Q_function.parameters(), lr = conf.LEARNING_RATE)
        self.Q_target  = Network(self.ns, self.model.nu)
        self.Q_target.eval()
        self.Q_target.copy(self.Q_function)

        self.eps          = [] # keeps track of the episode
        self.ctgs         = [] # keeps track of the cost-to-go
        self.decay        = [] # keeps track of the epsilon-greedy decay
        self.running_loss = [] # keeps track of the loss

        self.best_ctg = -np.inf
        self.episode = -1

        self.epsilon = conf.INITIAL_EXPLORATION
        self.filename = None
        self.Q_gepetto_gui = Network(self.ns, self.model.nu)
        self.display_model = CDPendulum(nu=conf.NU, uMax=conf.UMAX, dt=conf.DT, ndt=conf.NDT, noise_stddev=conf.NOISE)
        
    # reset environment, returns state as row vector
    def reset(self, x=None):
        return torch.tensor(self.model.reset(x), dtype=torch.float32).view(1,self.ns)

    def step(self, a):
        n_s, r = self.model.step(a)
        return torch.tensor(n_s, dtype=torch.float32).view(1,self.ns), torch.tensor([r], dtype=torch.float32)

    def failed(self, state, next_state):
        threashold = np.sin(5.0*np.pi/180.0) # threashold of 5.0 degrees
        sin_1 = np.abs(state[1]) + np.abs(state[4])
        sin_2 = np.abs(next_state[1]) + np.abs(next_state[4])
        if sin_1 < threashold and sin_2 > threashold:
            return True
        return False

    # update target
    def target_update(self):
        self.Q_target.copy(self.Q_function)

    def epsilon_decay(self):
        self.epsilon = conf.FINAL_EXPLORATION + (conf.INITIAL_EXPLORATION-conf.FINAL_EXPLORATION)*np.exp(-self.episode/conf.EXPLORATION_DECAY)
        self.decay.append(self.epsilon)

    # train step
    def update(self):
        transitions = self.buffer.sample(conf.MINIBATCH_SIZE)
        minibatch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          minibatch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in minibatch.next_state
                                                if s is not None])
        state_batch = torch.cat(minibatch.state)
        action_batch = torch.cat(minibatch.action)
        reward_batch = torch.cat(minibatch.reward)
        
        state_action_values = self.Q_function(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(conf.MINIBATCH_SIZE)
        next_state_values[non_final_mask] = self.Q_target(non_final_next_states).max(-1)[0].detach()
        expected_state_action_values = reward_batch + next_state_values * conf.GAMMA

        # Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.Q_function.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.detach().item()

    # epsilon-greedy
    def choose_action(self, state):
        if random.random() < self.epsilon:
            a_int = torch.tensor([[random.randrange(self.model.nu)]], dtype=torch.long)
        else:
            with torch.no_grad():
                a_int = self.Q_function(state).max(-1)[1].view(1,1)

        a = self.model.decode_action(a_int.item())

        return a, a_int  # a is an array, a_int is an integer (returned as singleton tensor)

    def display(self):
        print(self.episode)
        self.Q_gepetto_gui.load_state_dict(torch.load(self.filename))
        temp_s = self.display_model.reset(np.array([[np.pi + 1e-3, 0.],[0., 0.]])).reshape(self.ns)
        
        self.display_model.render()
        time.sleep(0.5)
        for _ in np.arange(0, conf.MAX_EPISODE_LENGTH):
            with torch.no_grad():
                a_simu = int(torch.argmax(self.Q_gepetto_gui(torch.Tensor(temp_s))))
            a_simu = self.display_model.decode_action(a_simu)
            s_next_simu, _ = self.display_model.step(a_simu)
            temp_s = s_next_simu.reshape(self.ns)

    def save_model(self, ctg):
        self.filename = "results/model/model_" + str(self.episode) + ".pth"
        if ctg > self.best_ctg:
            self.eps.append(self.episode)
            self.ctgs.append(ctg)
            self.best_ctg = ctg
            print(self.episode, ctg)
        torch.save(self.Q_function.state_dict(), self.filename)

    def save_csv(self):
        # stores all cost to go improvements and the episode at which they happened in the same csv file
        with open('results/csv/cost_to_go.csv', 'w', newline='') as csvfile:
            file = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i in np.arange(0, len(self.eps)):
                file.writerow([self.eps[i], self.ctgs[i]])

        # stores epsilon decay and loss in the same csv file
        with open('results/csv/loss.csv', 'w', newline='') as csvfile:
            file = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i in self.running_loss:
                file.writerow([i])

        with open('results/csv/decay.csv', 'w', newline='') as csvfile:
            file = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i in self.decay:
                file.writerow([i])

if __name__ == "__main__":
    agent = DQNAgent()

    Q_function_alt = Network(agent.ns, agent.model.nu)
    try: 
        step = 0
        while True:
            ctg = 0.
            gamma = 1.
            agent.episode += 1
            s = agent.reset()
            for t in np.arange(0, conf.MAX_EPISODE_LENGTH):
                step += 1

                a, a_int = agent.choose_action(s)
                s_next, r = agent.step(a)
                ctg += gamma * r
                gamma *= conf.GAMMA
                agent.buffer.push(s, a_int, r, s_next)
                s = s_next
                
                if len(agent.buffer) >= conf.REPLAY_START_SIZE and step % conf.REPLAY_SAMPLE_STEP:
                    l = agent.update()
                    #agent.running_loss.append(l.detach().item())                

                if (step % conf.NETWORK_RESET == 0):
                    agent.target_update()
                
            if ctg > agent.best_ctg:
                agent.save_model(ctg)
            
            agent.epsilon_decay()

            if agent.episode % 100 == 0:
                print(agent.episode)

            if agent.episode % conf.GEPETTO_GUI_VISUAL == 0 and agent.filename is not None:
                #print(agent.episode)
                Q_function_alt.load_state_dict(torch.load(agent.filename))

                s = agent.model.reset(x=np.array(conf.X_0)).reshape(agent.ns)
                time.sleep(1)
                for _ in np.arange(conf.MAX_EPISODE_LENGTH):
                    agent.model.render()
                    with torch.no_grad(): # needed for inference
                        a_encoded = int(torch.argmax(Q_function_alt(torch.Tensor(s))))
                    a = agent.model.decode_action(a_encoded)
                    s_next, _ = agent.model.step(a)
                    s = s_next.reshape(agent.ns)
                    time.sleep(0.03)
    except KeyboardInterrupt:
        agent.save_csv()
