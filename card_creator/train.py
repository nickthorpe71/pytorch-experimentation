import torch
import torch.nn as nn

from torchvision import datasets, transforms
import torchvision
import matplotlib.pyplot as plt

# import numpy as np
# import os

from models.generator import Generator
from models.discriminator import Discriminator

from training_stats_manager import TrainingStatsManager

torch.manual_seed(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Device configuration (use GPU)
device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu'
)
print(device)

# Hyper-parameters
batch_size = 128  # 128
num_epochs = 1
image_size = 32
num_channels = 3
z_size = 100
num_g_filters = image_size
num_d_filters = image_size
lr = 0.0002
betas = (0.5, 0.999)
num_workers = 1

# prep the dataset
transform = transforms.Compose([
    # transforms.CenterCrop((458, image_size)),
    # transforms.Pad((0, 84)),
    # transforms.Resize(image_size),
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

dataset = datasets.ImageFolder(
    'data/scryfall/card_images/cropped/full', transform=transform)

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


def weights_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)


# ------ GENERATOR ------ #
# initialize the generator and apply the weights_init function
netG = Generator(z_size, num_g_filters, num_channels)
netG = netG.to(device)
netG.apply(weights_init)
# apply recursively applies the weights_init function to every
# PyTorch submodule in the network


# ------ DISCRIMINATOR ------ #
# initialize the discriminator and apply the weights_init function
netD = Discriminator(num_channels, num_d_filters)
netD = netD.to(device)
netD.apply(weights_init)
# apply recursively applies the weights_init function to every
# PyTorch submodule in the network


# initialize the loss function
bce_loss = nn.BCELoss()


def discriminator_loss(real_output, fake_output):
    real_loss = bce_loss(real_output, torch.ones_like(real_output))
    fake_loss = bce_loss(fake_output, torch.zeros_like(fake_output))
    return real_loss + fake_loss


def generator_loss(fake_output):
    return bce_loss(fake_output, torch.ones_like(fake_output))


# initialize the optimizers
d_optimizer = torch.optim.Adam(netD.parameters(), lr=lr, betas=betas)
g_optimizer = torch.optim.Adam(netG.parameters(), lr=lr, betas=betas)

# to track training progress
g_losses = []
d_losses = []
fake_img_grids = []

fixed_noise = torch.randn(64, z_size, 1, 1, device=device)
with torch.no_grad():
    fixed_fakes = netG(fixed_noise).cpu()
fake_img_grids.append(torchvision.utils.make_grid(
    fixed_fakes, padding=2, normalize=True))

# Save noise images pre trianing
# plt.figure(figsize=(8, 8))
# plt.axis("off")
# plt.imshow(fake_img_grids[0].permute(1, 2, 0))
# plt.savefig('pre_training.png')

stats = TrainingStatsManager(batches_per_epoch=len(dataloader))
stats.begin_run()

for epoch in range(num_epochs):
    stats.begin_epoch()

    for batch in dataloader:

        # TRAIN DISCRIMINATOR
        netD.zero_grad()

        # in each batch, index 0 is the image data
        # and index 1 is the corrisponding label
        real_images = batch[0].to(device)
        real_output = netD(real_images)

        noise = torch.randn(real_images.size(0), z_size, 1, 1).to(device)
        fake_images = netG(noise)
        fake_output = netD(fake_images.detach())

        d_loss = discriminator_loss(real_output, fake_output)
        d_loss.backward()  # calculate gradients
        d_optimizer.step()  # update weights

        stats.track('real_mean', real_output.mean().item())
        stats.track('fake_mean1', fake_output.mean().item())
        stats.track('d_loss', d_loss.item())

        # TRAIN GENERATOR
        netG.zero_grad()
        fake_output = netD(fake_images)
        g_loss = generator_loss(fake_output)
        g_loss.backward()
        g_optimizer.step()

        stats.track('g_loss', g_loss.item())
        stats.track('fake_mean2', fake_output.mean().item())
        stats.progress.update()

    # # Display training stats
    # with torch.no_grad():
    #     fixed_fakes = netG(fixed_noise).cpu()
    # fake_img_grids.append(torchvision.utils.make_grid(
    #     fixed_fakes, padding=2, normalize=True))

    g_losses.append(sum(stats.epoch_data['g_loss'])/len(dataloader))
    d_losses.append(sum(stats.epoch_data['d_loss'])/len(dataloader))

    stats.add_result('D loss', sum(stats.epoch_data['d_loss'])/len(dataloader))
    stats.add_result('G loss', sum(stats.epoch_data['g_loss'])/len(dataloader))
    stats.add_result('avg D(x)', sum(
        stats.epoch_data['real_mean'])/len(dataloader))
    stats.add_result('avg D(G(z)) pre', sum(
        stats.epoch_data['fake_mean1'])/len(dataloader))
    stats.add_result('avg D(G(z)) post', sum(
        stats.epoch_data['fake_mean2'])/len(dataloader))

    stats.end_epoch()

stats.end_run()


with torch.no_grad():
    fixed_fakes = netG(fixed_noise).cpu()
fake_img_grids.append(torchvision.utils.make_grid(
    fixed_fakes, padding=2, normalize=True))

# Save png of training stats
# plt.figure(figsize=(10, 5))
# plt.title("Generator and Discriminator Loss")
# plt.plot(g_losses, label="G")
# plt.plot(d_losses, label="D")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.xticks(ticks=np.arange(len(g_losses)),
#            labels=np.arange(1, len(g_losses)+1))
# plt.legend()
# plt.show()
# plt.savefig('gan_training.png')


# Save a png of a real and fake image from the last epoch
real_batch = next(iter(dataloader))
images, labels = real_batch

real_grid = torchvision.utils.make_grid(images[:64], nrow=8, normalize=True)

plt.figure(figsize=(15, 15))

plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(real_grid.permute(1, 2, 0))

plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(fake_img_grids[-1].permute(1, 2, 0))

plt.show()
plt.savefig('gan_result.png')

# [ ] model
# [ ] train
# 	- load data
# 	- create model
# 	- train
# 	- save
# [ ] create_card
# 	- load model
# 	- generate images
