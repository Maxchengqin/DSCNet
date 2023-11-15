from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from ops import msst_seg8_v3
from torch.nn.init import normal_, constant_
weight_path = 'ops/msst_seg8_v3_imagenetpre_16x25_rgb_model_best.pth.tar'#
# weight_path = 'msst_seg8_v3_imagenetpre_16x25_rgb_model_best.pth.tar'#本地


class Net(nn.Module):
    def __init__(self, num_class=60, dropout=0.8):
        super(Net, self).__init__()
        self.dropout = dropout
        self.model = msst_seg8_v3.Net(num_class=1000, dropout=0.9).to(device)
        checkpoint = torch.load(weight_path)
        print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
        base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
        self.model.load_state_dict(base_dict)
        self.model.model.last_linear = nn.Linear(1280, num_class)
        std =0.001
        normal_(self.model.model.last_linear.weight, 0, std)
        constant_(self.model.model.last_linear.bias, 0)

    def forward(self, input):
        output = self.model(input)
        # output = self.newfc(output)
        return output

if __name__ == '__main__':
    net = Net(num_class=60).cuda()
    input = torch.rand((8, 2, 3, 8, 17)).cuda()
    print(net)
    out_put = net(input)
    print(out_put.size())
