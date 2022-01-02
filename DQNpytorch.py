import numpy as np
from numpy.random import randint, uniform
import torch
import torch.nn as nn


def get_critic(nx, nu):
    ''' Create the neural network to represent the Q function '''
    input_size = nx+nu
    hidden_sizes = [16,32,64,64]
    output_size = 1

    model = nn.Sequential(
        nn.Linear(input_size, hidden_sizes[0]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[0],hidden_sizes[1]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[1],hidden_sizes[2]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[2],hidden_sizes[3]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[3],output_size),
    )
    return model

def update(xu_batch, cost_batch, xu_next_batch):
    ''' Update the weights of the Q network using the specified batch of data '''
    target_values = Q_target(xu_next_batch)
    y = cost_batch + DISCOUNT*target_values
    Q_value = Q(xu_batch)
    loss = nn.MSELoss()                     # operand 
    Q_loss = loss(y, Q_value)               # actual loss
    
    critic_optimizer.zero_grad()
    Q_loss.backward()
    critic_optimizer.step()


if __name__ == "__main__":
    QVALUE_LEARNING_RATE = 1e-3
    DISCOUNT = 1e-3
    nx = 3
    nu = 2
    
    # initialize Neural Networks
    Q = get_critic(nx,nu)
    Q_target = get_critic(nx, nu)

    # set biases to zero
    for name, param in Q_target.named_parameters():
        if "bias" in name:
            param.data = torch.zeros(param.shape)

    # make sure that both neural network have the same bias and weights before training 
    Q.load_state_dict(Q_target.state_dict())

    # I did not really think about this but probably this should be inserted inside the training loop
    critic_optimizer = torch.optim.Adam(Q.parameters(),lr=QVALUE_LEARNING_RATE)
    