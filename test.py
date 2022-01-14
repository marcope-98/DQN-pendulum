import torch
from cdPendulum import CDPendulum
from Network import ReplayMemory, Network
import numpy as np
import conf as conf
import time

model = CDPendulum()
Q_function = Network(model.nx, model.nu)
Q_function.load_state_dict(torch.load("model/model.pth"))

s = model.reset(x=np.array([[np.pi],[0.]])).reshape(model.nx)
while True:
    model.render()
    with torch.no_grad(): # needed for inference
        a_encoded = int(torch.argmin(Q_function(torch.Tensor(s))))
    a = model.decode_action(a_encoded)
    print(s[0], s[1], a, a_encoded, model.d2cu(a_encoded))
    s_next, _ = model.step(a)
    s = s_next.reshape(model.nx)
    time.sleep(0.01)
