import torch

## Reshaping, stacking squeezing and unsqueezing tensors

# Rechaping - reshapes an input tensor to a defined shape
# View      - return a view of an input tensor of certain chape but keep the same memory as the original tensor
# Stacking  - combine multiple tensors on top of each other (vstack aka vertical stack) or side by side (hstack aka horizontal stack)
# Squeeze   - removes all `1` dimensions from a tensor
# Unsqueeze - adds a `1` dimenstion to a target tensor
# Permute   - Return a view of the input with dimenstions permuted (swapped) in a certain way

x = torch.arange(1., 10.)
print(x)
print(x.shape)

# Add an extra dimenstion
# * The reshape has to be compatable with the original size *
# this will fail because we can't fit a 9 vector into a 1,7 matrix
# x_reshaped = x.reshape(1,7)

# this adds an extra dimension making it a 1 by 9 matrix
x_reshaped = x.reshape(1,9) 
print(x_reshaped)
print(x_reshaped.shape)

# this adds an extra dimension making it a 9 by 1 matrix
x_reshaped = x.reshape(9,1) 
print(x_reshaped)
print(x_reshaped.shape)

# this adds an extra dimension making it a 3 by 3 matrix
x_reshaped = x.reshape(3,3) 
print(x_reshaped)
print(x_reshaped.shape)

