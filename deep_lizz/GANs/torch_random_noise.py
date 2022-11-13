import torch
# import numpy as np
import pandas as pd

torch.manual_seed(0)
torch.set_printoptions(linewidth=120)

# Create a tensor of 100 random numbers from a random uniform distribution over the interval [0, 1)]
noise_vector_1 = torch.rand(100)
print(noise_vector_1)

# Create a tensor of 100 random numbers from a random standard normal distribution with mean 0 and standard deviation 1
noise_vector_2 = torch.randn(100)
print(noise_vector_2)

pd.DataFrame(noise_vector_1).plot.hist(bins=10).get_figure().savefig('noise_vector_1.png')
pd.DataFrame(noise_vector_2).plot.hist(bins=10).get_figure().savefig('noise_vector_2.png')