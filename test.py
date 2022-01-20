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

FILENAME = "results/model/model_4376_5529.pth"

model = CDPendulum(nbJoint=conf.N_JOINTS, 
                   nu=conf.NU, 
                   uMax=conf.UMAX, 
                   dt=conf.DT, 
                   ndt=conf.NDT, 
                   noise_stddev=conf.NOISE, 
                   withSinCos=conf.WITHSinCos)
if conf.WITHSinCos:
    ns = model.nx + model.nv
else:
    ns = model.nx
Q_function = Network(ns, model.nu).eval()
Q_function.load_state_dict(torch.load(FILENAME))

# Display simulation
s = model.reset()
model.render()
time.sleep(1)
for _ in np.arange(300):
    model.render()
    a_int = Q_function(torch.tensor(s, dtype=torch.float32).view(1,-1)).max(-1)[1].item()
    a = model.decode_action(a_int)
    s_next, _ = model.step(a)
    s = s_next
    time.sleep(conf.DT)

def plot_V_table(qmin, qmax, dqmin, dqmax, Q_function):
    ''' Plot Value table '''
    Q = np.linspace(qmin, qmax, num=100)
    DQ = np.linspace(dqmin, dqmax, num=100)
    V = np.zeros((Q.size, DQ.size))
    for i in range(Q.size):
        for j in range(DQ.size):
            with torch.no_grad():
                V[i,j] = Q_function(torch.tensor([Q[i], DQ[j]], dtype=torch.float32)).max(-1)[0].item()
    plt.pcolormesh(Q, DQ, V, cmap=plt.cm.get_cmap('Blues'))
    plt.colorbar()
    plt.title('V table')
    plt.xlabel("q")
    plt.ylabel("dq")
    plt.show()

def plot_policy_table(qmin, qmax, dqmin, dqmax, Q_function):
    ''' Plot policy table '''
    Q = np.linspace(qmin, qmax, num=100)
    DQ = np.linspace(dqmin, dqmax, num=100)
    U = np.zeros((Q.size, DQ.size))
    for i in range(Q.size):
        for j in range(DQ.size):
            with torch.no_grad():
                a = Q_function(torch.tensor([Q[i], DQ[j]], dtype=torch.float32)).max(-1)[1].item()
                U[i,j] = model.d2cu(a)
    plt.pcolormesh(Q, DQ, U, cmap=plt.cm.get_cmap('Blues'))
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

# with open('results/csv/decay.csv', newline='') as csvfile:
#     reader = csv.reader(csvfile, delimiter=',', quotechar='|')
#     t = []
#     for row in reader:
#         t.append(float(row[0]))

# plt.plot(np.arange(0, len(t)), t)
# plt.show()
