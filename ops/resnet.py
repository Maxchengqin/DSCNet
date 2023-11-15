import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import logging
from collections import OrderedDict
from copy import deepcopy

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MULTIPLEXER_1x1(nn.Module):
    # expansion = 1

    def __init__(self, planes, stride=1):
        super(MULTIPLEXER_1x1, self).__init__()
        self.conv1 = conv1x1(planes, planes // 2, stride=1)
        self.bn1 = nn.BatchNorm2d(planes // 2)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv1x1(planes * self.expansion, planes // 2, stride=1)
        # self.bn2 = nn.BatchNorm2d(planes // 2)

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)

        return h


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpoo1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.inplanes = 64
        self.conv02 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn02 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool02 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer05 = self._make_layer(block, 64, layers[0])
        self.layer06 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer07 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer08 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        '''
        ###############  resnet50 and 101    #######################        
        self.multiplexer1 = MULTIPLEXER_1x1(64 * 2, stride=1)
        self.multiplexer2 = MULTIPLEXER_1x1(256 * 2, stride=1)
        self.multiplexer3 = MULTIPLEXER_1x1(512 * 2, stride=1)
        self.multiplexer4 = MULTIPLEXER_1x1(1024 * 2, stride=1)           
        self.multiplexer5 = MULTIPLEXER_1x1(2048 * 2, stride=1)
        '''
        self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        self.fc2 = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')  # , nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            '''
            print("**************************  layers *****************************")
            print(_)
            print(self.inplanes)
            print(planes)
            '''

        return nn.Sequential(*layers)

    def forward(self, x):
        # lamda = 1.0
        # beta = 1.0
        # print('*********************************** Resnet **********************************')
        x1, x2 = x

        # print('************* conv1 *****************')
        x1 = self.conv1(x1)  # (64,112,112)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpoo1(x1)  # (64,56,56)
        x2 = self.conv02(x2)
        x2 = self.bn02(x2)
        x2 = self.relu(x2)
        x2 = self.maxpool02(x2)

        '''
        #print('************* layer-1 *****************')
        x = torch.cat((x1,x2),dim=1)#(64*2,56,56)
        x = self.multiplexer1(x)#(64,56,56)
        x1 = lamda * x1 + (1 - lamda) * x
        x2 = beta * x2 + (1 - beta) * x
        '''
        x1 = self.layer1(x1)  # (64,56,56)
        x2 = self.layer05(x2)  # (64,56,56)
        '''
        #print('************* layer-2 *****************')
        x = torch.cat((x1,x2),dim=1)#(256*2,56,56)
        x = self.multiplexer2(x)
        x1 = lamda * x1 + (1 - lamda) * x
        x2 = beta * x2 + (1 - beta) * x     
        '''
        x1 = self.layer2(x1)
        x2 = self.layer06(x2)
        '''        
        #print('************* layer-3 *****************')
        x = torch.cat((x1,x2),dim=1)#(512*2,28,28)
        x = self.multiplexer3(x)#(512,28,28)
        x1 = lamda * x1 + (1 - lamda) * x
        x2 = beta * x2 + (1 - beta) * x   
        '''
        x1 = self.layer3(x1)  # (1024,14,14)
        x2 = self.layer07(x2)  # (1024,14,14)
        '''
        #print('************* layer-4 *****************')
        x = torch.cat((x1,x2),dim=1)#(1024*2,14,14)
        x = self.multiplexer4(x)#(256,14,14)
        x1 = lamda * x1 + (1 - lamda) * x
        x2 = beta * x2 + (1 - beta) * x      
        '''
        x1 = self.layer4(x1)
        rgb_feature = x1
        x2 = self.layer08(x2)
        depth_feature = x2
        '''
        #print('************* layer-5 *****************')
        x = torch.cat((x1,x2),dim=1)#(2048*2,7,7)
        x = self.multiplexer5(x)#(2048,7,7)
        x1 = lamda * x1 + (1 - lamda) * x
        x2 = beta * x2 + (1 - beta) * x  
        '''
        x1 = self.avgpool(x1)  # (2048,1,1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.fc1(x1)

        x2 = self.avgpool(x2)  # (2048,1,1)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.fc2(x2)

        return [x1, x2, rgb_feature, depth_feature]

        # return [x1, x2]


def xavier(model):
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 and hasattr(m, 'weight'):
            torch.nn.init.xavier_uniform(m.weight.data, gain=1.)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform(m.weight.data, gain=1.)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname in ['Sequential', 'AvgPool', 'MaxPool', \
                           'Dropout', 'ReLU', 'Softmax', 'BnActConv'] \
                or 'Block' in classname:
            pass
        else:
            if classname != classname.upper():
                logging.warning("Initializer:: '{}' is uninitialized.".format(classname))

    model.apply(weights_init)


def init_from_pretrain(model, pretrain, strict=False):
    pretrain_rgb = pretrain.copy()
    pretrain_dep = pretrain_rgb.copy()
    for k, v in list(pretrain_dep.items()):
        if 'conv1.weight' == k:
            pretrain_dep.update({k.replace('conv1.weight', 'conv02.weight'): pretrain_dep.pop(k)})
        elif 'bn1.running_mean' == k:
            pretrain_dep.update({k.replace('bn1.running_mean', 'bn02.running_mean'): pretrain_dep.pop(k)})
        elif 'bn1.running_var' == k:
            pretrain_dep.update({k.replace('bn1.running_var', 'bn02.running_var'): pretrain_dep.pop(k)})
        elif 'bn1.weight' == k:
            pretrain_dep.update({k.replace('bn1.', 'bn02.'): pretrain_dep.pop(k)})
        elif 'bn1.bias' == k:
            pretrain_dep.update({k.replace('bn1.bias', 'bn02.bias'): pretrain_dep.pop(k)})
        elif 'layer1' in k:
            pretrain_dep.update({k.replace('layer1.', 'layer05.'): pretrain_dep.pop(k)})
        elif 'layer2' in k:
            pretrain_dep.update({k.replace('layer2.', 'layer06.'): pretrain_dep.pop(k)})
        elif 'layer3' in k:
            pretrain_dep.update({k.replace('layer3.', 'layer07.'): pretrain_dep.pop(k)})
        elif 'layer4' in k:
            pretrain_dep.update({k.replace('layer4.', 'layer08.'): pretrain_dep.pop(k)})
        elif 'fc' in k:
            pretrain_dep.update({k.replace('fc.', 'fc.'): pretrain_dep.pop(k)})

    pretrain_new = pretrain_rgb.copy()
    del pretrain_new['fc.weight']
    del pretrain_new['fc.bias']
    pretrain_new.update(pretrain_dep)

    model.load_state_dict({k: v for k, v in pretrain_new.items()}, strict=strict)


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        print("resnet_18 RGB-D model parameters loading from pretrain...")
        pretrain = model_zoo.load_url(model_urls['resnet18'])
        init_from_pretrain(model=model, pretrain=pretrain, strict=False)
    else:
        xavier(model)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        print("resnet_34 RGB-D model parameters loading from pretrain...")
        pretrain = model_zoo.load_url(model_urls['resnet34'])
        init_from_pretrain(model=model, pretrain=pretrain, strict=False)
    else:
        xavier(model)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        print("resnet_50 RGB-D model parameters loading from pretrain...")
        # print(model)
        pretrain = model_zoo.load_url(model_urls['resnet50'])
        init_from_pretrain(model=model, pretrain=pretrain, strict=False)
    else:
        xavier(model)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        print("resnet_101 RGB-D model parameters loading from pretrain...")
        pretrain = model_zoo.load_url(model_urls['resnet101'])
        init_from_pretrain(model=model, pretrain=pretrain, strict=False)
    else:
        xavier(model)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        print("resnet_152 RGB-D model parameters loading from pretrain...")
        pretrain = model_zoo.load_url(model_urls['resnet52'])
        init_from_pretrain(model=model, pretrain=pretrain, strict=False)
    else:
        xavier(model)
    return model