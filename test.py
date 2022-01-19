# User Files
from env import CDPendulum
from env import Network
from env import conf
# pytorch
import torch
# numpy
import numpy as np
# others
import time
import csv
import matplotlib.pyplot as plt

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
Q_function = Network(ns, model.nu)

s = model.reset().reshape(ns)
model.render()

filename = "results/model/model_7049.pth"
Q_function.load_state_dict(torch.load(filename))
time.sleep(1)

for _ in np.arange(300):
    model.render()
    with torch.no_grad(): # needed for inference
        a_int = int(torch.argmin(Q_function(torch.Tensor(s))))
    a = model.decode_action(a_int)
    s_next, _ = model.step(a)
    s = s_next.reshape(ns)
    time.sleep(0.02)

with open('results/csv/decay.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    t = []
    for row in reader:
        t.append(float(row[0]))


plt.plot(np.arange(0, len(t)), t)
plt.show()
