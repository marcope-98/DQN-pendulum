import torch
from cdPendulum import CDPendulum
from Network import ReplayMemory, Network
import numpy as np
import conf as conf
import time
import csv
import matplotlib.pyplot as plt

files = [0, 1, 2, 3, 5, 74, 81, 116, 144, 145, 303, 453, 503, 941, 1026, 1059, 1589, 1754, 1879, 3225, 4396, 7049]
times = [150]*len(files)
times[-1] = 500
model = CDPendulum(conf.NU)
Q_function = Network(6, model.nu)

for i in files:
    s = model.reset(x=np.array([[np.pi+1e-2, 0.],[0.,0.]])).reshape(6)
    model.render()
    filename = "model/model_" + str(i) + ".pth"
    Q_function.load_state_dict(torch.load(filename))
    time.sleep(1)
    for _ in np.arange(175):
        model.render()
        with torch.no_grad(): # needed for inference
            a_encoded = int(torch.argmin(Q_function(torch.Tensor(s))))
        a = model.decode_action(a_encoded)
        s_next, _ = model.step(a)
        s = s_next.reshape(6)
        time.sleep(0.02)
"""
with open('loss.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    t = []
    for row in reader:
        t.append(float(row[0]))


plt.plot(np.arange(0, len(t)), t)
plt.show()
"""