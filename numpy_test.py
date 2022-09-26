import torch
import numpy as np

# Change np data into torch tensor
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array)
print(array)
print(tensor)

# Change torch tensor into np data 

