import torch
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# Intro to Tensors

print("--- Scalar ---")
scalar = torch.tensor(7)
print(scalar)
print(f"Dimensions: {scalar.ndim} -- A scalar has no dimensions.")
print(scalar.item()) # give us the scalar as a refular python int


print("--- Vector ---")
vector = torch.tensor([7,7])
print(vector)
print(f"Dimensions: {vector.ndim}")
print(f"Shape: {vector.shape}")


print("--- MATRIX ---")
MATRIX = torch.tensor([[1,2],[3,4]])
print(MATRIX)
print(f"Dimensions: {MATRIX.ndim}")
print(f"Shape: {MATRIX.shape}")


print("--- TENSOR ---")
TENSOR = torch.tensor([[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]],[[13,14],[15,16]]])
print(TENSOR)
print(f"Dimensions: {TENSOR.ndim}")
print(f"Shape: {TENSOR.shape}")
print(TENSOR[3][0])


print("--- Random TENSOR ---")
random_tensor = torch.rand(12, 6)
print(random_tensor)
# random_tensor_4d = torch.rand(10, 10, 10, 10)


print("--- Random Image TENSOR ---")
random_image_size_tensor = torch.rand(size=(224, 224, 3)) # height, width, color channel
print(random_image_size_tensor)
print(f"Dimensions: {random_image_size_tensor.ndim}")
print(f"Shape: {random_image_size_tensor.shape}")


print("--- Zeros and Ones TENSOR ---")
zeros = torch.zeros(size=(12,6))
print(zeros)
print(zeros * random_tensor)

ones = torch.ones(size=(3,4))
print(ones)
print(ones.dtype) # check the data type


print("--- Range of Tensors and tensor-like ---")
print(torch.range(0, 10))  # deprecated
print(torch.arange(0, 11)) # update

# with step
print(torch.arange(start=0, end=1000, step=77))

# tensor-like (create tensor in same shape as another tensor)
one_to_ten = torch.arange(0, 11)
ten_zeros = torch.zeros_like(input=one_to_ten)
print(ten_zeros)


print("--- Datatypes ---")
# Float 32 tensor (default)
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                                dtype=None, # defines the datatype
                                device=None, # default is "cpu" 
                                requires_grad= False) # whether or not to track gradients with this tensors operations
print(float_32_tensor.dtype)

# Float 16 tensor (default)
float_16_tensor = torch.tensor([3.0, 6.0, 9.0], dtype=torch.float16)
# or
float_16_tensor = float_32_tensor.type(torch.float16)
# or 
float_16_tensor = float_32_tensor.type(torch.half)
print(float_16_tensor.dtype)


print("--- Math Operators ---")
tensor_a = torch.tensor([[1,2], [3,4]])
tensor_b = torch.tensor([[5,6], [7,8]])

# basic
print(f"Add tensor+tensor: {torch.add(tensor_a, tensor_b)}")
print(f"Add tensor+int: {torch.add(tensor_a, 2)}")
print(f"Subtract tensor+tensor: {torch.sub(tensor_a, tensor_b)}")
print(f"Subtract tensor+int: {torch.sub(tensor_a, 2)}")
print(f"Multiply tensor+tensor: {torch.mul(tensor_a, tensor_b)}")
print(f"Multiply tensor+int: {torch.mul(tensor_a, 2)}")
print(f"Divide tensor+tensor: {torch.divide(tensor_a, tensor_b)}")
print(f"Divide tensor+int: {torch.divide(tensor_a, 2)}")

# matrix
print(f"Multiply tensor+tensor: {torch.matmul(tensor_a, tensor_b)}")

