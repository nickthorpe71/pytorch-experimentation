import torch
# import torch.nn as nn

from torchvision import datasets, transforms
# import torchvision
# import matplotlib.pyplot as plt

# import numpy as np
# import os

# from training_stats_manager import TrainingStatsManager

# Device configuration (use GPU)
device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu'
)
print(device)

# Hyper-parameters
batch_size = 128
num_epochs = 1
image_height = 452
image_width = 616
num_channels = 3
z_size = 100
num_g_filters = 64
num_d_filters = 64
lr = 0.0002
betas = (0.5, 0.999)
num_workers = 1

# prep the dataset
transform = transforms.Compose([transforms.Resize(
    (image_height, image_width)), transforms.ToTensor(), transforms.Normalize(mean=(0.5), std=(0.5))])

dataset = datasets.ImageFolder(
    'data/scryfall/card_images/cropped/sample', transform=transform)

# load the dataset
dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)

# plot some images from the training set to get a quick visualization
# real_batch = next(iter(dataloader))
# images, labels = real_batch

# grid = torchvision.utils.make_grid(images, nrow=10, normalize=True)

# plt.figure(figsize=(50, 50))
# plt.axis("off")
# plt.imshow(grid.permute(1, 2, 0))
# plt.savefig('sample.png')

# [ ] model
# [ ] train
# 	- load data
# 	- create model
# 	- train
# 	- save
# [ ] create_card
# 	- load model
# 	- generate images
