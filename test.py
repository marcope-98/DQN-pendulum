import torch
from cdPendulum import CDPendulum
from Network import ReplayMemory, Network
import numpy as np
import conf as conf
import time
import csv
import matplotlib.pyplot as plt

model = CDPendulum()
Q_function = Network(6, model.nu)
Q_function.load_state_dict(torch.load("model/model_430.pth"))

s = model.reset(x=np.array([[np.pi+0.1, 0.],[0.,0.]])).reshape(6)

for _ in np.arange(conf.MAX_EPISODE_LENGTH*3):
    model.render()
    with torch.no_grad(): # needed for inference
        a_encoded = int(torch.argmin(Q_function(torch.Tensor(s))))
    a = model.decode_action(a_encoded)
    s_next, _ = model.step(a)
    s = s_next.reshape(6)
    time.sleep(0.03)
"""
with open('loss.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    t = []
    for row in reader:
        t.append(float(row[0]))


plt.plot(np.arange(0, len(t)), t)
plt.show()
"""