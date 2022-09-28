import torch


# Create known parameters
weight = 0.7
bias   = 0.3

# Create
start = 0
end   = 1
step  = 0.02
X     = torch.arange(start, end, step).unsqueeze(dim=1) # add an extra dimension
y     = weight * X  + bias

print(X[:10])
print(y[:10])
print(len(X))
print(len(y))

