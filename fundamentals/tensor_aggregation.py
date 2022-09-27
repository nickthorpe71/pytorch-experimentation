import torch

# Create a tensor
x = torch.arange(0,100,10) # 0 - 100 with step of 10

# Find min
print(torch.min(x))
# or
print(x.min())

# Find max
print(torch.max(x))
# or
print(x.max())

# Find mean
print(torch.mean(x.type(torch.float32)))
# or
print(x.type(torch.float32).mean())

# Find sum
print(torch.sum(x))
# or
print(x.sum())

