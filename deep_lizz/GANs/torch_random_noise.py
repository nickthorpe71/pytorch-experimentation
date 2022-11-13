import torch
# import numpy as np
import pandas as pd

torch.manual_seed(0)
torch.set_printoptions(linewidth=120)

# Create a tensor of 100 random numbers
noise_vector_1 = torch.randn(100)
print(noise_vector_1)

noise_vector_2 = torch.randn(100)
print(noise_vector_2)

pd.DataFrame(noise_vector_1).plot.hist(bins=10).get_figure().savefig('noise_vector_1.png')
pd.DataFrame(noise_vector_2).plot.hist(bins=10).get_figure().savefig('noise_vector_2.png')