import torch, time
import torch.nn as nn
import torch.nn.functional as F

from args import *

class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()

        if parser_args.dataset in ['mnist']:
            self.width = 28
            self.height = 28
            self.channel = 1
        elif parser_args.dataset in ['cifar10']:
            self.width = 32
            self.height = 32
            self.channel = 3
        
        self.conv = []

        self.linear = nn.Sequential(
            nn.Linear(self.width * self.height * self.channel, 1000),
            nn.ReLU(),
            nn.Linear(1000, 600),
            nn.ReLU(),
            nn.Linear(600, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )
 
    def forward(self, x):
        out = x.view(x.size(0), self.width * self.height * self.channel)
        out = self.linear(out)

        return out.squeeze()
 
    def initialize_weights(self):
        print('init model FC')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                m.bias.data.zero_()

class FC2(nn.Module):
    def __init__(self):
        super(FC2, self).__init__()

        if parser_args.dataset in ['mnist']:
            self.width = 28
            self.height = 28
            self.channel = 1
        elif parser_args.dataset in ['cifar10']:
            self.width = 32
            self.height = 32
            self.channel = 3
        
        self.conv = []

        self.linear = nn.Sequential(
            nn.Linear(self.width * self.height * self.channel, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )
 
    def forward(self, x):
        out = x.view(x.size(0), self.width * self.height * self.channel)
        out = self.linear(out)

        return out.squeeze()
 
    def initialize_weights(self):
        print('init model FC2')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                m.bias.data.zero_()

class FC12(nn.Module):
    def __init__(self):
        super(FC12, self).__init__()

        if parser_args.dataset in ['mnist']:
            self.width = 28
            self.height = 28
            self.channel = 1
        elif parser_args.dataset in ['cifar10']:
            self.width = 32
            self.height = 32
            self.channel = 3

        self.conv = []

        self.linear = nn.Sequential(
            nn.Linear(self.width * self.height * self.channel, 1000),
            nn.ReLU(),
            nn.Linear(1000,900),#2
			nn.ReLU(),
			nn.Linear(900,800),#3
			nn.ReLU(),
			nn.Linear(800,750),#4
			nn.ReLU(),
			nn.Linear(750,700),#5
			nn.ReLU(),
			nn.Linear(700,650),#6
			nn.ReLU(),
			nn.Linear(650,600),#7
			nn.ReLU(),
			nn.Linear(600,500),#8
			nn.ReLU(),
			nn.Linear(500,400),#9
			nn.ReLU(),
			nn.Linear(400,200),#10
			nn.ReLU(),
			nn.Linear(200,100),#11
			nn.ReLU(),
			nn.Linear(100,10),
        )
 
    def forward(self, x):
        out = x.view(x.size(0), self.width * self.height * self.channel)
        out = self.linear(out)

        return out.squeeze()
 
    def initialize_weights(self):
        print('init model FC12')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if parser_args.init in ['kaiming_normal']:
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                elif parser_args.init in ['kaiming_uniform']:
                    nn.init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                m.bias.data.zero_()

class FC13(nn.Module):
    def __init__(self):
        super(FC13, self).__init__()

        if parser_args.dataset in ['mnist']:
            self.width = 28
            self.height = 28
            self.channel = 1
        elif parser_args.dataset in ['cifar10']:
            self.width = 32
            self.height = 32
            self.channel = 3

        self.conv = []

        self.linear = nn.Sequential(
            nn.Linear(self.width * self.height * self.channel, 1000),
            nn.ReLU(),
            nn.Linear(1000,900),#2
			nn.ReLU(),
			nn.Linear(900,800),#3
			nn.ReLU(),
			nn.Linear(800,750),#4
			nn.ReLU(),
			nn.Linear(750,700),#5
			nn.ReLU(),
			nn.Linear(700,650),#6
			nn.ReLU(),
			nn.Linear(650,600),#7
			nn.ReLU(),
			nn.Linear(600,500),#8
			nn.ReLU(),
			nn.Linear(500,400),#9
			nn.ReLU(),
			nn.Linear(400,250),#10
			nn.ReLU(),
            nn.Linear(250,200),#11
			nn.ReLU(),
			nn.Linear(200,100),#12
			nn.ReLU(),
			nn.Linear(100,10),
        )
 
    def forward(self, x):
        out = x.view(x.size(0), self.width * self.height * self.channel)
        out = self.linear(out)

        return out.squeeze()
 
    def initialize_weights(self):
        print('init model FC13')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if parser_args.init in ['kaiming_normal']:
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                elif parser_args.init in ['kaiming_uniform']:
                    nn.init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if parser_args.init in ['kaiming_normal']:
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                elif parser_args.init in ['kaiming_uniform']:
                    nn.init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                m.bias.data.zero_()

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        if parser_args.dataset in ['mnist']:
            self.width = 28
            self.height = 28
            self.channel = 1
        elif parser_args.dataset in ['cifar10']:
            self.width = 32
            self.height = 32
            self.channel = 3

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.channel, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )

        self.linear = nn.Sequential(
            nn.Linear(800, 120),
            nn.ReLU(),
            nn.Linear(120,84),
			nn.ReLU(),
			nn.Linear(84,10),
        )
 
    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size()[0], -1)
        out = self.linear(out)

        return out.squeeze()
 
    def initialize_weights(self):
        print('init model LeNet')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if parser_args.init in ['kaiming_normal']:
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                elif parser_args.init in ['kaiming_uniform']:
                    nn.init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if parser_args.init in ['kaiming_normal']:
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                elif parser_args.init in ['kaiming_uniform']:
                    nn.init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                m.bias.data.zero_()

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        if parser_args.dataset in ['mnist']:
            self.width = 28
            self.height = 28
            self.channel = 1
        elif parser_args.dataset in ['cifar10']:
            self.width = 32
            self.height = 32
            self.channel = 3

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.channel, out_channels=96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(96,256,5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(256,384,3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384,384,3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384,256,3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
        )

        self.linear = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Linear(4096,4096),
			nn.ReLU(),
			nn.Linear(4096,10),
        )
 
    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size()[0], -1)
        out = self.linear(out)

        return out.squeeze()
 
    def initialize_weights(self):
        print('init model AlexNet')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if parser_args.init in ['kaiming_normal']:
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                elif parser_args.init in ['kaiming_uniform']:
                    nn.init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if parser_args.init in ['kaiming_normal']:
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                elif parser_args.init in ['kaiming_uniform']:
                    nn.init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                m.bias.data.zero_()

class CNet(nn.Module):
    def __init__(self):
        super(CNet, self).__init__()

        if parser_args.dataset in ['mnist']:
            self.width = 28
            self.height = 28
            self.channel = 1
        elif parser_args.dataset in ['cifar10']:
            self.width = 32
            self.height = 32
            self.channel = 3

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.channel, out_channels=64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,32,5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,16,5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )

        self.linear = nn.Sequential(
            nn.Linear(1024, 120),
            nn.ReLU(),
            nn.Linear(120,84),
			nn.ReLU(),
			nn.Linear(84,10),
        )
 
    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size()[0], -1)
        out = self.linear(out)

        return out.squeeze()
 
    def initialize_weights(self):
        print('init model CNet')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if parser_args.init in ['kaiming_normal']:
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                elif parser_args.init in ['kaiming_uniform']:
                    nn.init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if parser_args.init in ['kaiming_normal']:
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                elif parser_args.init in ['kaiming_uniform']:
                    nn.init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                m.bias.data.zero_()

'''VGG11/13/16/19 in Pytorch.'''
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):

        super(VGG, self).__init__()

        if parser_args.dataset in ['mnist']:
            self.width = 28
            self.height = 28
            self.channel = 1
        elif parser_args.dataset in ['cifar10']:
            self.width = 32
            self.height = 32
            self.channel = 3

        self.conv = self._make_layers(cfg[vgg_name])

        self.linear = nn.Sequential(
            nn.Linear(512, 10),
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.channel
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def initialize_weights(self):
        print('init model VGG')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if parser_args.init in ['kaiming_normal']:
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                elif parser_args.init in ['kaiming_uniform']:
                    nn.init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if parser_args.init in ['kaiming_normal']:
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                elif parser_args.init in ['kaiming_uniform']:
                    nn.init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                m.bias.data.zero_()

"""resnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=10):
        super().__init__()

        if parser_args.dataset in ['cifar100']:
            num_classes = 100

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        print(strides)
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output
    
    def initialize_weights(self):
        print('init model ResNet')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if parser_args.init in ['kaiming_normal']:
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                elif parser_args.init in ['kaiming_uniform']:
                    nn.init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if parser_args.init in ['kaiming_normal']:
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                elif parser_args.init in ['kaiming_uniform']:
                    nn.init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                m.bias.data.zero_()

def resnet18():
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34():
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])