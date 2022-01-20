# User Files
from math import ceil
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
        self.Q_target  = Network(self.ns, self.model.nu)
        self.Q_target.eval()
        self.Q_target.copy(self.Q_function)

        self.optimizer = optim.Adam(self.Q_function.parameters(), lr = conf.LEARNING_RATE)

        self.eps          = [] # keeps track of the episode
        self.ctgs         = [] # keeps track of the cost-to-go
        self.decay        = [] # keeps track of the epsilon-greedy decay
        self.running_loss = [] # keeps track of the loss

        self.best_ctg = -np.inf
        self.episode = -1

        self.epsilon = conf.INITIAL_EXPLORATION
        self.filename = None
        
    # reset environment, returns state as row vector
    def reset(self, x=None):
        return torch.tensor(self.model.reset(x), dtype=torch.float32).view(1,self.ns)

    def step(self, a):
        s, r = self.model.step(self.model.decode_action(a))
        return torch.tensor(s, dtype=torch.float32).view(1,self.ns), torch.tensor([r], dtype=torch.float32)

    def failed(self, state, next_state):
        threshold = np.sin(5.0*np.pi/180.0) # threshold of 5.0 degrees
        sin_1 = np.abs(state[1]) + np.abs(state[4])
        sin_2 = np.abs(next_state[1]) + np.abs(next_state[4])
        if sin_1 < threshold and sin_2 > threshold:
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
            a = torch.tensor([[random.randrange(self.model.nu)]], dtype=torch.long)
        else:
            with torch.no_grad():
                a = self.Q_function(state).max(-1)[1].view(1,1)

        return a # singleton tensor of encoded action

    def display(self):
        print("Displaying progress")
        s = agent.reset(conf.X_0)
        agent.model.render()
        time.sleep(1)
        for _ in np.arange(conf.MAX_EPISODE_LENGTH):
            agent.model.render()
            a = self.Q_target(s).max(-1)[1].view(1,1)
            s_next, _ = agent.step(a.item())
            s = s_next
            time.sleep(conf.DT)

    def save_model(self, ctg):
        quant_ctg = ceil(ctg * 1000)
        self.filename = "results/model/model_" + str(quant_ctg) + "_" + str(self.episode) + ".pth"
        self.eps.append(self.episode)
        self.ctgs.append(ctg)
        print("Saving model episode:", self.episode, "cost-to-go:", ctg)
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
    
    def track_improvement(self):
        ctg = 0.
        gamma = 1.
        s = self.reset(conf.X_0)
        for t in np.arange(0, conf.MAX_EPISODE_LENGTH):
            a = self.Q_target(s).max(-1)[1].view(1,1)
            s_next, r = agent.step(a.item())
            ctg += gamma * r
            gamma *= conf.GAMMA
            agent.buffer.push(s, a, r, s_next)
            s = s_next
        if ctg > self.best_ctg:
            self.best_ctg = ctg
            self.save_model(-ctg.item())
        return -ctg.item()
        

if __name__ == "__main__":
    # Create agent
    agent = DQNAgent()
    try: 
        step = 0
        #Training loop
        while True:
            agent.episode += 1
            episode_loss = []
            # Reset pendulum and get initial state
            s = agent.reset()
            # Episode loop
            for t in np.arange(0, conf.MAX_EPISODE_LENGTH):
                step += 1
                # Choose action with epsilon-greedy policy, then update state and save transition in replay buffer
                a = agent.choose_action(s)
                s_next, r = agent.step(a.item())
                agent.buffer.push(s, a, r, s_next)
                s = s_next

                # Run model optimization step and track training loss
                if len(agent.buffer) >= conf.REPLAY_START_SIZE and step % conf.REPLAY_SAMPLE_STEP:
                    l = agent.update()
                    episode_loss.append(l)                

                # Update target network weights
                if (step % conf.NETWORK_RESET == 0):
                    agent.target_update()

            # Re-evaluate epsilon value
            agent.epsilon_decay()

            # Track average training loss
            agent.running_loss.append(np.mean(episode_loss))

            # Print track advancement
            if agent.episode % 100 == 0:
                print("Episode:", agent.episode)
                ctg = agent.track_improvement()

            # Display progress in 3D view
            if agent.episode % conf.GEPETTO_GUI_VISUAL == 0:
                agent.display()

    except KeyboardInterrupt:
        agent.save_csv()
        ctg = agent.track_improvement()
        agent.save_model(ctg)
