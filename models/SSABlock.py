import torch
import torch.nn as nn


class SalientBlock(nn.Module):
    def __init__(self, in_planes):
        super(SalientBlock, self).__init__()
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        size_n, _, size_h, size_w = list(x.size())
        w = self.globalAvgPool(x)
        w = torch.mean(w*x, 1).view((size_n, 1, size_h, size_w))
        spatial_w = self.sigmoid(self.bn(w))
        out = spatial_w * x
        return out


# get 1 attention map by every k feature maps. 
# If in_planes=256, channels_per_maps=16, we could get 256/16=16 attention maps.
class GroupSalientBlock(nn.Module):
    def __init__(self, in_planes, channels_per_map):
        super(GroupSalientBlock, self).__init__()
        assert channels_per_map>in_planes or in_planes%channels_per_map==0
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.groups = in_planes//channels_per_map if in_planes//channels_per_map>0 else 1
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        size_n, size_c, size_h, size_w = list(x.size())
        w = self.globalAvgPool(x).view((size_n, self.groups, -1, 1, 1))
        x = x.view((size_n, self.groups, -1, size_h, size_w))
        w = torch.mean(w*x, 2).view((size_n, self.groups, 1, size_h, size_w))
        spatial_w = self.sigmoid(self.bn(w))
        out = spatial_w * x
        out = out.view((size_n, size_c, size_h, size_w))
        return out
