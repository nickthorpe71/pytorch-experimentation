import torch

# Create a tensor
x = torch.arange(0,100,10) # 0 - 100 with step of 10

# Find min
print(torch.min(x))
# or
print(x.min())

