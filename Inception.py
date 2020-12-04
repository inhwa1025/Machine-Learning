'''in Pytorch'''

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
batch_size = 64

# MNIST Dataset
train_dataset = datasets.MNIST(root='./data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./data/',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = nn.Conv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = nn.Conv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = nn.Conv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = nn.Conv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = nn.Conv2d(in_channels, 384, kernel_size=1, stride=2)

        self.branch3x3dbl_1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = nn.Conv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = nn.Conv2d(96, 96, kernel_size=3, stride=2)

        self.mp = nn.MaxPool2d(3, 2)


    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.mp(x)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 192, kernel_size=1)

        self.branch7x7_1 = nn.Conv2d(in_channels, channels_7x7, kernel_size=1)
        self.branch7x7_2 = nn.Conv2d(channels_7x7, channels_7x7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = nn.Conv2d(channels_7x7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7db_1 = nn.Conv2d(in_channels, channels_7x7, kernel_size=1)
        self.branch7x7db_2 = nn.Conv2d(channels_7x7, channels_7x7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7db_3 = nn.Conv2d(channels_7x7, channels_7x7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7db_4 = nn.Conv2d(channels_7x7, channels_7x7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7db_5 = nn.Conv2d(channels_7x7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = nn.Conv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_2(branch7x7)

        branch7x7dbl = self.branch7x7db_1(x)
        branch7x7dbl = self.branch7x7db_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7db_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7db_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7db_5(branch7x7dbl)
       

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()

        self.branch3x3_1 = nn.Conv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = nn.Conv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = nn.Conv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = nn.Conv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = nn.Conv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = nn.MaxPool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):
  
  def __init__(self, in_channels):
    super(InceptionE, self).__init__()
    self.branch1x1 = nn.Conv2d(in_channels, 320, kernel_size=1)

    self.branch3x3_1 = nn.Conv2d(in_channels, 384, kernel_size=1)
    self.branch3x3_2a = nn.Conv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
    self.branch3x3_2b = nn.Conv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

    self.branch3x3db1_1 = nn.Conv2d(in_channels, 448, kernel_size=1)
    self.branch3x3db1_2 = nn.Conv2d(448, 384, kernel_size=3, padding=1)
    self.branch3x3db1_3a = nn.Conv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
    self.branch3x3db1_3b = nn.Conv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

    self.branch_pool = nn.Conv2d(in_channels, 192, kernel_size=1)

  def forward(self, x):
    branch1x1 = self.branch1x1(x)

    branch3x3 = self.branch3x3_1(x)
    branch3x3 = [self.branch3x3_2a(branch3x3), 
                 self.branch3x3_2b(branch3x3), ]
    branch3x3 = torch.cat(branch3x3, 1)

    branch3x3db1 = self.branch3x3db1_1(x)
    branch3x3db1 = self.branch3x3db1_2(branch3x3db1)
    branch3x3db1 = [self.branch3x3db1_3a(branch3x3db1),
                    self.branch3x3db1_3b(branch3x3db1),]
    branch3x3db1 = torch.cat(branch3x3db1, 1)
       

    branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch_pool = self.branch_pool(branch_pool)

    outputs = [branch1x1, branch3x3, branch3x3db1, branch_pool]
    return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

  def __init__(self, in_channels, num_classes):
    super(InceptionAux, self).__init__()
    self.conv0 = nn.Conv2d(in_channels, 128, kernel_size=1)
    self.conv1 = nn.Conv2d(128, 768, kernel_size=5)
    self.conv1.stddev = 0.001

  def forward(self, x):
    x = F.avg_pool2d(x, kernel_size=5, stride=3)
    x = self.conv0(x)
    x = self.conv1(x)
    x = F.adaptive_avg_pool2d(x, (1, 1))
    x = torch.flatten(x, 1)
    x = self.fc(x)
    return x


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1a = nn.Conv2d(1, 32, kernel_size=3, stride=2)
        self.conv2a = nn.Conv2d(32, 32, kernel_size=3)
        self.conv2b = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3b = nn.Conv2d(64, 80, kernel_size=1)
        self.conv4a = nn.Conv2d(80, 192, kernel_size=3)
        self.mp2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.incept1 = InceptionA(in_channels=192, pool_features=32)
        self.incept2 = InceptionA(in_channels=256, pool_features=64)
        self.incept3 = InceptionA(in_channels=288, pool_features=64)
        self.incept4 = InceptionB(in_channels=288)
        self.incept5 = InceptionC(in_channels=768, channels_7x7=128)
        self.incept6 = InceptionC(in_channels=768, channels_7x7=160)
        self.incept7 = InceptionC(in_channels=768, channels_7x7=160)
        self.incept8 = InceptionC(in_channels=768, channels_7x7=192)
        self.inceptAux = InceptionAux(768, 1000)
        self.incept9 = InceptionD(768)
        self.incept10 = InceptionE(1280)
        self.incept11 = InceptionE(2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(2048, 1000)


    def forward(self, x):
        x = self.conv1a(x)
        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.mp1(x)
        x = self.conv3b(x)
        x = self.conv4a(x)
        x = self.mp2(x)
        x = self.incept1(x)
        x = self.incept2(x)
        x = self.incept3(x)
        x = self.incept4(x)
        x = self.incept5(x)
        x = self.incept6(x)
        x = self.incept7(x)
        x = self.incept8(x)
        x = self.inceptAux(x)
        x = self.incept9(x)
        x = self.incept10(x)
        x = self.incept11(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.log_softmax(x)


model = Net()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, 10):
    train(epoch)
    test()
