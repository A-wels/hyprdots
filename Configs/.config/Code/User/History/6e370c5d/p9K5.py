import numpy as np

v = np.fromfile('1.v', 'float32')
data =  torch.from_numpy(np.reshape(v,[344,344,127], order='F')).float()

print(v.shape)
# reduce size
v = v[20:-20, 40:-40, :]
print(v.shape)
