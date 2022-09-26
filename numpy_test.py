import torch
import numpy as np

# Change np data into torch tensor
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array) # creates a copy of the original array
print(array)
print(tensor)

# Change torch tensor into np data 
tensor = torch.ones(7)
numpy_tensor = tensor.numpy()
print(tensor)
print(numpy_tensor)
