import torch
import numpy as np

x = torch.ones(2,2, dtype=torch.float16)
print(x.size())

x_from_list = torch.tensor([2, 5, 0.1])
print(x_from_list)

# random tensors
x = torch.rand(2,2)
y = torch.rand(2,2)
print(x)
print(y)

z = x + y
z = torch.add(x,y)
print(z)

# print all rows and first column
print(x[:, 0])

# print all columns and second row
print(x[1, :])

# print element at pos 1,1
print(x[1,1])

# resize tensors
x = torch.rand(4,4)
print(x)
y = x.view(16, 1)
print(y)
y = x.view(2, 8)

# numpy / torch conversions
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
# a and b both point to the same memory location

c = np.ones(5)
print(c)
d = torch.from_numpy(c)
print(d)

# create a tensor on the gpu
if torch.cuda.is_available():
  device = torch.device("cuda")
  x = torch.ones(5, device=device)
  # or
  y = torch.ones(5)
  y = y.to(device)