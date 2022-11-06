import torch
import torch.nn as nn

# An example of taking a 2x2 input tensor and using transpose convolution to
# upsample the input to a 3x4 output tensor. The filter size is 2x2 and the
# stride is 1.

t = torch.arange(1,5).float()
t = t.reshape(1,1,2,2)
print(t)

pt_transconv_layer = nn.ConvTranspose2d(
  in_channels = 1,
  out_channels=1,
  kernel_size=2,
  stride=1,
  bias=False
)

pt_filter= torch.arange(1,5).reshape(1,1,2,2).float()
# print(pt_filter)

pt_transconv_layer.weight = nn.Parameter(pt_filter)

t_transconv = pt_transconv_layer(t)
print(t_transconv)