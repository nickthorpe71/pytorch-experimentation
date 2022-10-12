import torch

x = torch.randn(3, requires_grad=True)
print(x)

y = x+2
print(y)

z = y*y*2
z = z.mean()
print(z)

# backward needs to be called on a scalar value which is why we used z.mean() above
# or pass it a vector like below:
# v = torch.tensor([0.01, 1.0, 0.0001], dtype=torch.float32)
# z.backward(v)
z.backward() # will calculate the gradient of z in respect to x
# Tensors will not have the grad property if "requires_grad=True" is not defined in tensor creation
print(x.grad)


# Three ways to prevent gradients from being tracked
print(x)

# remove the gradient tracking from x
x.requires_grad_(False) 
print(x)

# create a new tensor with the same data but without gradient tracking
y = x.detach()
print(y)

# wrap the tensor in a "with torch.no_grad()" block
with torch.no_grad():
    y = x + 2
    print(y)




