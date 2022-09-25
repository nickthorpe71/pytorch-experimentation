import torch

x = torch.arange(1,10).reshape(1,3,3)
print(x)
print(x.shape)

print(x[0])
print(x[0][0])
print(x[0, 0])
print(x[0][0][1])
print(x[0, 0, 1])
