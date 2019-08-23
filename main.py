'''Train CIFAR10 with PyTorch.'''
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

def seed_torch(seed=1205):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_torch()


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument("--ckpt-dir", default="./checkpoint", type=str)
parser.add_argument("--pretrain-model", default="", type=str, help="type like: ckpt")
parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100"],\
                     default="cifar100", help="choose a dataset in (cifar10, cifar100)")
parser.add_argument("--batch-size", type=int, default=100, help="batch size")
parser.add_argument("--backbone", type=str, choices=["resnet", "senet", "salientnet"],\
                     help="choose backbone in (resnet, senet, salientnet)")
parser.add_argument("--layers-num", type=int, choices=[18, 34, 50, 101, 152], default=50,\
                     help="choose backbone layers num in (18, 34, 50, 101, 152)")
parser.add_argument("--gpu", type=str, default="0", help="use which gpu, usage: '0' or '0,1,2,3'")
args = parser.parse_args()
print("args are {}".format(args))

# set gpu: CUDA_VISIBLE_DEVICES
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("devices are {}".format(device))
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])



# preparing dataset
print("preparing dataset: {}".format(args.dataset))
num_classes = 0
if args.dataset == "cifar10":
    trainset = torchvision.datasets.CIFAR10(root='../Datasets/cifar10', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='../Datasets/cifar10', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    num_classes = 10
else args.dataset == "cifar100":
    trainset = torchvision.datasets.CIFAR100(root='../Datasets', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR100(root='../Datasets', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    num_classes = 100


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = None
if args.backbone=="resnet":
    if args.layers_num == 18:
        net = ResNet18(num_classes)
    elif args.layers_num==34:
        net = ResNet34(num_classes)
    elif args.layers_num==50:
        net = ResNet50(num_classes)
    elif args.layers_num==101:
        net = ResNet101(num_classes)
    elif args.layers_num==152:
        net = ResNet152(num_classes)
    else:
        print("no such net called {}, implement it yourself, please".format(args.backbone + args.layers_num))
elif args.backbone=="senet":
    if args.layers_num == 18:
        net = SENet18(num_classes)
    elif args.layers_num==34:
        net = SENet34(num_classes)
    elif args.layers_num==50:
        net = SENet50(num_classes)
    elif args.layers_num==101:
        net = SENet101(num_classes)
    elif args.layers_num==152:
        net = SENet152(num_classes)
    else:
        print("no such net called {}, implement it yourself, please".format(args.backbone + args.layers_num))
elif args.backbone=="salientnet":
    if args.layers_num == 18:
        net = SalientResNet18(num_classes)
    elif args.layers_num==34:
        net = SalientResNet34(num_classes)
    elif args.layers_num==50:
        net = SalientResNet50(num_classes)
    elif args.layers_num==101:
        net = SalientResNet101(num_classes)
    elif args.layers_num==152:
        net = SalientResNet152(num_classes)
    else:
        print("no such net called {}, implement it yourself, please".format(args.backbone + args.layers_num))
else:
    print("no such backbone in my implements, check ./models to find it or implement it yourself, please")

# net = VGG('VGG19')
# net = ResNet50()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet50()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = SalientResNet50()
# net = SalientResNet18()
# net = ResNet18()
print("Net building finished")
net = net.to(device)
if device == 'cuda':
    print("using GPU for training")
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# if args.resume:
if args.pretrain_model:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    # assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    model_name = "{}.pth".format(args.pretrain_model)
    checkpoint = torch.load(os.path.join(args.ckpt_dir, model_name))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

if not os.path.exists(args.ckpt_dir):
    os.makedirs(args.ckpt_dir)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

LR=0
def adjust_lr(tmp_optimizer, epoch):
	"""Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
	lr = args.lr * (0.1 ** (epoch // 50)); print("learning rate is {}".format(lr)); LR=lr
	for param_group in tmp_optimizer.param_groups: param_group['lr'] = lr


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    adjust_lr(optimizer, epoch)
    iter_num=0
    for batch_idx, (inputs, targets) in enumerate(trainloader): ####################
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        iter_num+=1
        #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        if iter_num %10000==0:
            print("IterNum is: %d, Training Loss: %.3f | Acc: %.3f%% (%d/%d)" % (iter_num, train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print("Training Loss: %.3f | Acc: %.3f%% (%d/%d)" % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print("Testing Loss: %.3f | Acc: %.3f%% (%d/%d)" % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    save_best = True
    if acc > best_acc and save_best:
    # if True:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        # if not os.path.isdir('checkpoint'):
        #     os.mkdir('checkpoint')
        # if save_best:
        torch.save(state, os.path.join(args.ckpt_dir, 'ckpt_{:.4f}.pth'.format(LR)))
        # else:
        #     torch.save(state, os.path.join(args.ckpt_dir, 'ckpt_{}.pth'.format(epoch)))
        best_acc = acc


for epoch in range(start_epoch, start_epoch+500):
    if epoch >=250:
        break
    train(epoch)
    test(epoch)
