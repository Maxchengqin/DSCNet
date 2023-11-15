import torch.nn as nn
import torch

class Exchange(nn.Module):
    def __init__(self):
        super(Exchange, self).__init__()

    def forward(self, x, bn, bn_threshold):
        bn1, bn2 = bn[0].weight.abs(), bn[1].weight.abs()
        x1, x2 = torch.zeros_like(x[0]), torch.zeros_like(x[1])
        x1[:, bn1 >= bn_threshold] = x[0][:, bn1 >= bn_threshold]
        x1[:, bn1 < bn_threshold] = x[1][:, bn1 < bn_threshold]
        x2[:, bn2 >= bn_threshold] = x[1][:, bn2 >= bn_threshold]
        x2[:, bn2 < bn_threshold] = x[0][:, bn2 < bn_threshold]
        return [x1, x2]

class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        # self.module = module
        # self.module1 = module
        # self.module2 = module
        for i in range(2):
            setattr(self, 'module' + str(i), module)

    def forward(self, x_parallel):
        # print('xxxxxxxxxxxxxxxxx',x_parallel[0].size())
        # return [self.module1(x_parallel[0]), self.module2(x_parallel[1])]
        return [getattr(self, 'module' + str(i))(x) for i, x in enumerate(x_parallel)]

class ModuleParallel_conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False):
        super(ModuleParallel_conv, self).__init__()
        for i in range(2):
            setattr(self, 'module' + str(i), nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                                       kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))

    def forward(self, x_parallel):
        return [getattr(self, 'module' + str(i))(x) for i, x in enumerate(x_parallel)]

class ModuleParallel_fc(nn.Module):
    def __init__(self, in_features=2048, out_features=60):
        super(ModuleParallel_fc, self).__init__()
        for i in range(2):
            setattr(self, 'module' + str(i), nn.Linear(in_features=in_features, out_features=out_features))

    def forward(self, x_parallel):
        return [getattr(self, 'module' + str(i))(x) for i, x in enumerate(x_parallel)]

class BatchNorm2dParallel(nn.Module):
    def __init__(self, num_features, num_parallel):
        super(BatchNorm2dParallel, self).__init__()
        for i in range(num_parallel):
            setattr(self, 'bn_' + str(i), nn.BatchNorm2d(num_features))

    def forward(self, x_parallel):
        return [getattr(self, 'bn_' + str(i))(x) for i, x in enumerate(x_parallel)]
