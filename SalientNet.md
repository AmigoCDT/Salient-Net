[TOC]

# Salient-Net

一种类似于SeNet那样的在ResNet上挂载一个网络挂件结构。

## ResNet



## SeNet

### SeBlock

本质上是给卷积后的每个feature map学习一个权重。

认为在每个卷积后的feature maps中，每个通道对于整个网络的重要性不同(可能是基于稀疏理论？)。

![SEBlock](.\SalientNet-pics\SEBlock.jpg)

以上是SEBlock的结构，假设input的shape为(N, C, H, W)，其中N为batch_size， C为通道数，H和W分别为每个通道的图像的高和宽。通过Global_Pooling降维到(N, C, 1, 1)，再通过全连接层，学习到W=(N, C, 1, 1)，即为input的每个通道的权重，再将W数乘(broadcast_mult)到input中。

![SE_ResNet_Inception](.\SalientNet-pics\SE_ResNet_Inception.jpg)



### 参数量和计算量分析

输入为input=(1, C, H, W)

参数量：C*C/8

计算量：C\*C/8 + C\*H*W



## Salient-Net

### 思想来源和启发

Salient-Net的思想来源于传统计算机视觉中的图像显著性分析，对一张图像，学习三种特征图（亮度图，轮廓图和SIFT特征图），再将这三种特征图进行加权求和，得到的特征图即为显著图(Salient-Map).

将这种思想应用到CNN中，我们就可以得到一种新的网络结构Salient-Block。

对比于Se-Block是学习到(N, C, 1, 1)作为每个通道的权重，Salient-Block是学习到(N, 1, H, W)作为所有通道每个像素的权重。

实现：

```python
class SailentBlock(nn.Module):
    def __init__(self, in_planes):
        super(SailentBlock, self).__init__()
        self.conv = nn.Conv2d(in_planes, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(1)
    
    def forward(self, x):
        w = F.sigmoid(self.bn(self.conv(x)))
        out = w * x # boradcasting multime
        return out
```



***参数量少***，相比于SE-Block的C*C/8，Salient-Block只需要C\*1的参数量。对于32*\*32的输入图像，分别将Salient-Block和SE-block挂载于ResNet-50上，Salient-Net增加了3776个参数，而SE-Block增加了891136个参数。

***效果好***

|           | ResNet50 | SeNet50 | SalientNet50 |
| --------- | -------- | ------- | ------------ |
| cifar-100 | 77.20%   | 77.45%  | 78.25%       |
|           |          |         |              |

