#考虑到时间维度只有8，所以不用太大的卷积核，多尺度模块分两种，到后面的时候不用7以上的卷积核。
#在v2基础上增加通道，即增加参数量
from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F


class msstMoudel(nn.Module):
    def __init__(self, inchannel, outchannel1, outchannel3, outchannel5, outchannel7,stride=(1, 1)):
        super(msstMoudel, self).__init__()
        inplace = True

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel1, kernel_size=(1, 1), stride=stride,
                      padding=(0, 0)),
            nn.BatchNorm2d(outchannel1, affine=True),
            nn.ReLU(inplace))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel3, kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0)),
            nn.BatchNorm2d(outchannel3, affine=True),
            nn.ReLU(inplace),
            nn.Conv2d(in_channels=outchannel3, out_channels=outchannel3, kernel_size=(3, 3), stride=stride,
                      padding=(1, 1)),
            nn.BatchNorm2d(outchannel3, affine=True),
            nn.ReLU(inplace))

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel5, kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0)),
            nn.BatchNorm2d(outchannel5, affine=True),
            nn.ReLU(inplace),
            nn.Conv2d(in_channels=outchannel5, out_channels=outchannel5, kernel_size=(5, 1), stride=stride,
                      padding=(2, 0)),
            nn.BatchNorm2d(outchannel5, affine=True),
            nn.ReLU(inplace),
            nn.Conv2d(in_channels=outchannel5, out_channels=outchannel5, kernel_size=(1, 5), stride=(1, 1),
                      padding=(0, 2)),
            nn.BatchNorm2d(outchannel5, affine=True),
            nn.ReLU(inplace))

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel7, kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0)),
            nn.BatchNorm2d(outchannel7, affine=True),
            nn.ReLU(inplace),
            nn.Conv2d(in_channels=outchannel7, out_channels=outchannel7, kernel_size=(7, 1), stride=stride,
                      padding=(3, 0)),
            nn.BatchNorm2d(outchannel7, affine=True),
            nn.ReLU(inplace),
            nn.Conv2d(in_channels=outchannel7, out_channels=outchannel7, kernel_size=(1, 7), stride=(1, 1),
                      padding=(0, 3)),
            nn.BatchNorm2d(outchannel7, affine=True),
            nn.ReLU(inplace))

    def forward(self, input):
        output1 = self.conv1(input)
        # print('o1',output1.size())
        output3 = self.conv3(input)
        # print('o3', output3.size())
        output5 = self.conv5(input)
        # print('o5', output5.size())
        output7 = self.conv7(input)
        output = torch.cat([output1, output3, output5, output7], 1)
        return output

class MSSTNet(nn.Module):

    def __init__(self, num_class=1000, dropout=0.8):
        super(MSSTNet, self).__init__()
        inplace = True
        self.dropout = dropout

        self.avgpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 2), padding=(1, 1), ceil_mode=False, count_include_pad=True),
            nn.BatchNorm2d(416, affine=True),
            nn.ReLU(inplace))
        self.avgpool2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 2), padding=(1, 1), ceil_mode=False, count_include_pad=True),
            nn.BatchNorm2d(672, affine=True),
            nn.ReLU(inplace))

        self.m1 = msstMoudel(inchannel=3, outchannel1=44, outchannel3=60, outchannel5=60, outchannel7=60, stride=(1, 1))
        self.m2 = msstMoudel(inchannel=224, outchannel1=48, outchannel3=80, outchannel5=80, outchannel7=80, stride=(1, 1))
        self.m3 = msstMoudel(inchannel=288, outchannel1=56, outchannel3=120, outchannel5=120, outchannel7=120, stride=(1, 1))
        self.m4 = msstMoudel(inchannel=416, outchannel1=160, outchannel3=160, outchannel5=160, outchannel7=160, stride=(1, 1))
        self.m5 = msstMoudel(inchannel=640, outchannel1=72, outchannel3=200, outchannel5=200, outchannel7=200, stride=(2, 1))
        self.m6 = msstMoudel(inchannel=672, outchannel1=240, outchannel3=240, outchannel5=240, outchannel7=240, stride=(1, 1))
        self.m7 = msstMoudel(inchannel=960, outchannel1=320, outchannel3=320, outchannel5=320, outchannel7=320, stride=(1, 1))
        self.dropout = nn.Dropout(p=self.dropout)
        self.last_linear = nn.Linear(1280, num_class)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, input):
        m1out = self.m1(input)
        m2out = self.m2(m1out)
        m3out = self.m3(m2out)
        m3poolout = self.avgpool(m3out)
        m4out = self.m4(m3poolout)
        m5out = self.m5(m4out)
        m5poolout = self.avgpool2(m5out)
        m6out = self.m6(m5poolout)
        m7out = self.m7(m6out)
        # print(m7out.size())
        x = self.global_pool(m7out).squeeze()
        x = self.dropout(x)
        x = self.last_linear(x)
        return x


class Net(nn.Module):
    def __init__(self, num_class=60, dropout=0.8):
        super(Net, self).__init__()
        self.dropout = dropout
        self.num_class = num_class
        self.model = MSSTNet(num_class=self.num_class, dropout=self.dropout)

#####################################双图片改bachsize################################
    def forward(self, input):
        B, O, C, H, W = input.size()
        input = input.view(B*O,C,H,W)
        output = self.model(input)
        output = output.view(B,O,-1).mean(dim=1)
        return output
####################################################################################
if __name__=='__main__':
    from ptflops import get_model_complexity_info

    model = Net(num_class=60, dropout=0.8)
    input_sk = torch.rand([4, 2, 3, 8, 17])
    out = model(input_sk)
    print(out.size())
    flops, params = get_model_complexity_info(model, (2, 3, 8, 16), as_strings=True, print_per_layer_stat=False)  # as_strings=True,会用G或M为单位，反之精确到个位。
    print(flops, params) # joint 874.47 MMac 11.12 M   bone 766.02 MMac 11.12 M