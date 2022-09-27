import torch


# These are random
random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)

print(random_tensor_A == random_tensor_B)

# Create random but reproducible tensors

# set random seed
RANDOM_SEED = 42  # <-- answer to life
torch.manual_seed(RANDOM_SEED)

random_tensor_C = torch.rand(3, 4)
random_tensor_D = torch.rand(3, 4)

print(random_tensor_C == random_tensor_D)

# need to set the seed before reach call or it will not work
torch.manual_seed(RANDOM_SEED)
random_tensor_C = torch.rand(3, 4)

torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand(3, 4)

print(random_tensor_C == random_tensor_D)
