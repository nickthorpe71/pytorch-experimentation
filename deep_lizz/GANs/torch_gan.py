import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
# import matplotlib.animation
import matplotlib.pyplot as plt

# import numpy as np
import os
# from IPython.display import HTML

# for consistancy on each run
torch.manual_seed(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Device configuration (use GPU)
device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu'
)
print(device)

# Hyper-parameters
batch_size = 128
num_epochs = 5
image_size = 64
num_channels = 1
z_size = 100
num_g_filters = 64
num_d_filters = 64
lr = 0.0002
betas = (0.5, 0.999)
num_workers = 1


# download and prepare the MNIST dataset
dataset = torchvision.datasets.MNIST(
    root=os.path.relpath('data/mnist'),
    download=True,
    transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5), std=(0.5))
    ])
)

# load the dataset
dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)

# plot some images from the training set to get a quick visualization
real_batch = next(iter(dataloader))
images, labels = real_batch

grid = torchvision.utils.make_grid(images, nrow=10, normalize=True)

plt.figure(figsize=(10, 10))
plt.axis("off")
plt.imshow(grid.permute(1, 2, 0))
plt.savefig('sample.png')


def weights_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)


netG = nn.Sequential(
    nn.ConvTranspose2d(
        in_channels=z_size,
        out_channels=num_g_filters * 8,
        kernel_size=4,
        stride=1,
        padding=0,
        bias=False
    ),
    nn.BatchNorm2d(num_features=num_g_filters * 8),
    nn.ReLU(inplace=True),

    nn.ConvTranspose2d(
        in_channels=num_g_filters * 8,
        out_channels=num_g_filters * 4,
        kernel_size=4,
        stride=2,
        padding=1,
        bias=False
    ),
    nn.BatchNorm2d(num_features=num_g_filters * 4),
    nn.ReLU(inplace=True),

    nn.ConvTranspose2d(
        in_channels=num_g_filters * 4,
        out_channels=num_g_filters * 2,
        kernel_size=4,
        stride=2,
        padding=1,
        bias=False
    ),
    nn.BatchNorm2d(num_features=num_g_filters * 2),
    nn.ReLU(inplace=True),

    nn.ConvTranspose2d(
        in_channels=num_g_filters * 2,
        out_channels=num_g_filters,
        kernel_size=4,
        stride=2,
        padding=1,
        bias=False
    ),
    nn.BatchNorm2d(num_features=num_g_filters),
    nn.ReLU(inplace=True),

    nn.ConvTranspose2d(
        in_channels=num_g_filters,
        out_channels=num_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        bias=False
    ),
    nn.Tanh()
)

# initialize the generator and apply the weights_init function
netG = netG.to(device)
# apply recursively applies the weights_init function to every
# PyTorch submodule in the network
netG.apply(weights_init)
