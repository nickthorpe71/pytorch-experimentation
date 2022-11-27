import torch.nn as nn


def Generator(z_size, num_g_filters, num_channels):
    """
      # As input, the generator accepts batches of noise vectors,
      # each generally referred to as z and of length z_size, with values
      # randomly sampled from a normal distribution. The output of the
      # generator is a num_channels x num_g_filters x num_g_filters image. 
      # This is an image with num_channels color channels (RGB) with a 
      # height and width of num_g_filters pixels.
    """

    return nn.Sequential(
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
