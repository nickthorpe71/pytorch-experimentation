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

#* Add an Extra Dimenstion
#* The reshape has to be compatable with the original size *
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


#* Change the View
x = torch.arange(1., 10.)
z = x.view(1,9)
print(x)
print(z)

# A view is just a reference to a tensor, ∴ changing z will change x 
z[:, 0] = 5
print(x)
print(z)


#* Stack tensors
x = torch.arange(1., 10.)
# stack vertically
x_stacked = torch.stack([x, x ,x, x], dim=0)
print(x)
print(x_stacked)

# stack horizontally
x_stacked = torch.stack([x, x ,x, x], dim=1)
print(x)
print(x_stacked)


#* Squeeze / Unsqueeze
# Squeeze
# removes single dimenstions
x = torch.zeros(2, 1, 2, 1, 2)
print(x.size())
y = torch.squeeze(x)
print(y.size())
y = torch.squeeze(x, 0)
print(y.size())
y = torch.squeeze(x, 1)
print(y.size())

# unsqueeze
# adds a single dimention
x = torch.tensor([1, 2, 3, 4])
print(torch.unsqueeze(x, 0))
print(torch.unsqueeze(x, 1))


#* Permute
# returns view (ref) of the original tensor input with its dimensions permuted
x = torch.randn(2, 3, 5)
print(x)
print(x.size())
print(torch.permute(x, (2, 0, 1)).size())
print(torch.permute(x, (2, 0, 1)))