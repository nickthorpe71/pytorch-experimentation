import torch
import torch.nn as nn
import torch.nn.functional as F

t = torch.arange(1, 5).float()
print(t)
t = t.reshape(1, 1, 2, 2)
print(t)

upsample_nearest = nn.Upsample(scale_factor=2, mode='nearest')
t_nearest = upsample_nearest(t)
print(t_nearest)

upsample_bilinear = nn.Upsample(scale_factor=2, mode='bilinear')
t_bilinear = upsample_bilinear(t)
print(t_bilinear)

t_interpolate = F.interpolate(t, scale_factor=2, mode='bilinear')
print(t_interpolate)
