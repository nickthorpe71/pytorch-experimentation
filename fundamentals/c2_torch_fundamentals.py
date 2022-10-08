import torch

x = torch.ones(2,2, dtype=torch.float16)
print(x.size())

x_from_list = torch.tensor([2, 5, 0.1])
print(x_from_list)

# random tensors
x = torch.rand(2,2)
y = torch.rand(2,2)
print(x)
print(y)