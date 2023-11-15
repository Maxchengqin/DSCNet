import torch
from torch import nn
import numpy as np
import time
class STMEM(nn.Module):
    def __init__(self, num_segments, new_length):
        super(STMEM, self).__init__()
        self.num_segments = num_segments
        self.new_length = new_length
        self.sig = torch.nn.Sigmoid()
        # self.max_pool =

        self.m1 = nn.Sequential(
            nn.Conv2d(in_channels=(self.new_length*2-1)*3, out_channels=3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.m2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
    def forward(self, input):
        # print(input)
        # print('11111111111',input.size())#BXseg*length*3X224X224
        # B, S, LC, W, H = input.size()#测试用这句，训练用下一句。。。。
        B, SLC, W, H = input.size()#训练用这句，测试用上一句。。。。
        input = input.view(B * self.num_segments, self.new_length * 3, 224, 224)

        frame_diff = input[:, 3:] - input[:, 0:(self.new_length - 1) * 3]
        input_and_frame_diff = torch.cat((input, frame_diff), 1)
        input_and_frame_diff = self.m1(input_and_frame_diff)

        frame_diff = frame_diff.view(B * self.num_segments, self.new_length-1, 3, 224, 224)
        frame_diff = frame_diff.max(1)[0]#max() 输出是两项，第一项是max后的值，第二项是max值的索引。

        frame_diff = self.m2(frame_diff)
        frame_diff = self.sig(frame_diff)

        output = frame_diff * input_and_frame_diff
        return output#B*segX3X224X224

if __name__ == '__main__':
    a = torch.rand([4,90,224,224])
    model = STMEM(num_segments=5,new_length=6)
    out = model(a)
    print(out.size())