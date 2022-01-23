# User Files
from xml.etree.ElementTree import PI
from env import CDPendulum
from env import Network
from env import conf
# pytorch
import torch
# numpy
import numpy as np
# others
import time
import math
import csv
import matplotlib.pyplot as plt

FILENAME = "results/model/model_47042_8900.pth"

model = CDPendulum(nbJoint=conf.N_JOINTS, 
                   nu=conf.NU, 
                   uMax=conf.UMAX, 
                   dt=conf.DT/conf.NDT, 
                   ndt=1, 
                   noise_stddev=conf.NOISE, 
                   withSinCos=conf.WITHSinCos)
if conf.WITHSinCos:
    ns = model.nx + model.nv
else:
    ns = model.nx
Q_function = Network(ns, model.nu).eval()
Q_function.load_state_dict(torch.load(FILENAME))

# Display simulation
s = model.reset(conf.X_0)
model.render()
time.sleep(1)
for _ in np.arange(30*conf.NDT/conf.DT): # Simulate for 30 seconds
    model.render()
    a_int = Q_function(torch.tensor(s, dtype=torch.float32).view(1,-1)).max(-1)[1].item()
    a = model.decode_action(a_int)
    s_next, _ = model.step(a)
    s = s_next
    time.sleep(conf.DT/conf.NDT)

def plot_V_table(qmin, qmax, dqmin, dqmax, Q_function):
    ''' Plot Value table '''
    Q = np.linspace(qmin, qmax, num=100)
    DQ = np.linspace(dqmin, dqmax, num=100)
    V = np.zeros((DQ.size, Q.size))
    for i in range(Q.size):
        for j in range(DQ.size):
            with torch.no_grad():
                V[j,i] = Q_function(torch.tensor([Q[i], DQ[j]], dtype=torch.float32)).max(-1)[0].item()
    plt.pcolormesh(Q,DQ,V, cmap=plt.cm.get_cmap('Blues'))
    plt.colorbar()
    plt.title('V table')
    plt.xlabel("q")
    plt.ylabel("dq")
    plt.show()

def plot_policy_table(qmin, qmax, dqmin, dqmax, Q_function):
    ''' Plot policy table '''
    Q = np.linspace(qmin, qmax, num=100)
    DQ = np.linspace(dqmin, dqmax, num=100)
    U = np.zeros((DQ.size, Q.size))
    for i in range(Q.size):
        for j in range(DQ.size):
            with torch.no_grad():
                a = Q_function(torch.tensor([Q[i], DQ[j]], dtype=torch.float32)).max(-1)[1].item()
                U[j,i] = model.d2cu(a)
    plt.pcolormesh(Q,DQ,U, cmap=plt.cm.get_cmap('Blues'))
    plt.colorbar()
    plt.title('Policy table')
    plt.xlabel("q")
    plt.ylabel("dq")
    plt.show()

if conf.N_JOINTS == 1:
    qmin = -math.pi
    qmax = math.pi
    dqmin = -math.pi
    dqmax = math.pi
    plot_V_table(qmin, qmax, dqmin, dqmax, Q_function)
    plot_policy_table(qmin, qmax, dqmin, dqmax, Q_function)

with open('results/csv/cost_to_go.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    ep = []
    ctg = []
    for row in reader:
        ep.append(float(row[0]))
        ctg.append(float(row[1]))
plt.plot(ep, ctg)
plt.title('Cost-to-go over episodes')
plt.xlabel("Episode")
plt.ylabel("Cost-to-go")
plt.show()

with open('results/csv/loss.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    loss = []
    for row in reader:
        loss.append(float(row[0]))
plt.plot(np.arange(1, len(loss)), loss[1:])
plt.title('Temporal difference error')
plt.xlabel("Episode")
plt.ylabel("e")
plt.show()
