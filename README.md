# Salient-Net, SENet 以及 ResNet 在 cifar10 和 cifar100 效果比较
  在cifar-100和cifar-10上训练，batch_size均为100，学习率最开始为0.1，每过50个epoch学习率除以10。总共训练240个epoch。选取在测试集上的最大的ACC作为结果。

|                      | ResNet50 | SeNet50 | SalientNet50 |
| -------------------- | -------- | ------- | ------------ |
| cifar-100(Top-1 ACC) | 77.260%  | 77.130% | 78.350%      |
| cifar-10(Top-1 ACC)  | 94.380%  | 94.830% | 94.930%      |



# Train CIFAR with PyTorch

I'm playing with [PyTorch](http://pytorch.org/) on the cifar10 or cifar100 dataset.

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Accuracy
This is not my result, this result is from [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)

| Model             | Acc.        |
| ----------------- | ----------- |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 92.64%      |
| [ResNet18](https://arxiv.org/abs/1512.03385)          | 93.02%      |
| [ResNet50](https://arxiv.org/abs/1512.03385)          | 93.62%      |
| [ResNet101](https://arxiv.org/abs/1512.03385)         | 93.75%      |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)       | 94.43%      |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431)  | 94.73%      |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)  | 94.82%      |
| [DenseNet121](https://arxiv.org/abs/1608.06993)       | 95.04%      |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)    | 95.11%      |
| [DPN92](https://arxiv.org/abs/1707.01629)             | 95.16%      |

## Learning rate adjustment
I manually change the `lr` during training:
- `0.1` for epoch `[0,50)`
- `0.01` for epoch `[50,100)`
- `0.001` for epoch `[100,150)`
- `0.0001` for epoch `[150,200)`
- `0.00001` for epoch `[200,250)`

train from scratch `python main.py --lr 0.1 >./log/logfile.log`

Resume the training with `python main.py --resume --lr=0.1 --ckpt-dir ./checkpoint --pretrain-model ./ckpt,133 >./log/resume_logfile.log`
