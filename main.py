'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR 


import os
import argparse

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', default='resnet18')
parser.add_argument('--variation', default=0, type=int, help='variation on base model')
parser.add_argument('--cuda', default='0', type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--epochs', default=350, type=int)

args = parser.parse_args()

device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='/share/cuvl/pytorch_data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='/share/cuvl/pytorch_data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
args.model = args.model.lower()
net = None
if args.model  == 'vgg19':
    net = VGG('VGG19')
elif args.model  == 'vgg16':
    net = VGG('VGG16')
elif args.model == 'resnet18':
    net = ResNet18(variation=args.variation)
elif args.model == 'preactresnet18':
    net = PreActResNet18()
elif args.model == 'googlenet':
    net = GoogLeNet()
elif args.model == 'densenet121':
    net = DenseNet121()
elif args.model == 'resnext29':
    net = ResNeXt29_2x64d()
elif args.model == 'mobilenet':
    net = MobileNet()
elif args.model == 'mobilenetv2':
    net = MobileNetV2()
elif args.model == 'dpn92':
    net = DPN92()
elif args.model == 'shufflenetg2':
    net = ShuffleNetG2()
elif args.model == 'senet18':
    net = SENet18()

net = net.to(device)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f'./checkpoint/{args.model}.pth')
    net.load_state_dict(checkpoint)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = MultiStepLR(optimizer, milestones=[150,250], gamma=0.1)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
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

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

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

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    saveEpochs = [0,1,2,3,4,5]
    if acc > best_acc or epoch in saveEpochs:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')

        if epoch in saveEpochs:
            torch.save(state, f'./checkpoint/{args.model}_{epoch}.pth')
        else:
            torch.save(state, f'./checkpoint/{args.model}.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+args.epochs):
    scheduler.step()
    train(epoch)
    test(epoch)
