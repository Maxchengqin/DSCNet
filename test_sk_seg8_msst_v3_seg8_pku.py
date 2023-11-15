import warnings
from ops import msst_seg8_v3_pre_replace_fc as msst
warnings.filterwarnings("ignore")
import argparse
import time
import numpy as np
from sklearn.metrics import confusion_matrix
from ops.dataset_sk_2d_seg_rand_pku_with_confidence import MSSTdata
import torch
# options
parser = argparse.ArgumentParser(description="Standard video-level testing")
parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics', 'ntu', 'ntu120', 'pku'])
parser.add_argument('modality', type=str, choices=['joint', 'joint_motion', 'bone', 'bone_motion'])
parser.add_argument('test_path', type=str)
parser.add_argument('weights', type=str)
parser.add_argument('--save_scores', type=str, default=None)
parser.add_argument('--dropout', type=float, default=0.7)
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--batchsize', default=1, type=int,)
parser.add_argument('--gpus', nargs='+', type=int, default=None)

args = parser.parse_args()
if args.dataset == 'ucf101':
    num_class = 101
elif args.dataset == 'hmdb51':
    num_class = 51
elif args.dataset == 'kinetics':
    num_class = 400
elif args.dataset == 'ntu':
    num_class = 60
elif args.dataset == 'ntu120':
    num_class = 120
elif args.dataset == 'pku':
    num_class = 51
else:
    raise ValueError('Unknown dataset '+args.dataset)

model = msst.Net(num_class=num_class, dropout=args.dropout)
# net = newmodel3.Net(num_class=num_class, dropout=args.dropout)
checkpoint = torch.load(args.weights)
# print(checkpoint)
print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
# print(base_dict)
print('loading model ......')
model.load_state_dict(base_dict)
print('loading data ......')
data_loader = torch.utils.data.DataLoader(MSSTdata(dataroot=args.test_path, modality=args.modality, test_mode=True, seg=8), batch_size=args.batchsize, shuffle=False, num_workers=args.workers * 2, pin_memory=True)

if args.gpus is not None:
    devices = [args.gpus[i] for i in range(args.workers)]
else:
    devices = list(range(args.workers))

model = torch.nn.DataParallel(model.cuda(devices[0]), device_ids=devices)
model.eval()

# data_gen = enumerate(data_loader)

total_num = len(data_loader.dataset)
print('测试样本总数量', total_num)
output = []

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # print(pred)
    # print(pred.size())
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    answer = pred[0]

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        # print(correct_k)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, target, answer

end = time.time()
batch_time=AverageMeter()
top1 = AverageMeter()
top5 = AverageMeter()
answers = []
targets = []

outputs = []
time_st = time.time()
for i, (input, target) in enumerate(data_loader):
    target = target.cuda()
    input_var = torch.autograd.Variable(input, volatile=True)
    target_var = torch.autograd.Variable(target, volatile=True)
    output = model(input_var)
    # print(output.size())
    outputs.append(output.data.cpu().numpy().copy())

    (prec1, prec5), target1, answer = accuracy(output.data, target, topk=(1, 5))
    answers.append(answer)
    targets.append(target)

    # measure elapsed time
    batch_time.update(time.time() - end)
    top1.update(prec1.item(), input.size(0))
    top5.update(prec5.item(), input.size(0))
    end = time.time()

    if i % 20 == 0:#20个batch打印一次。
        print(('Test: [{0}/{1}]\t'
               'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
               'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
               'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(i, total_num//args.batchsize, batch_time=batch_time, top1=top1, top5=top5)))
time_end = time.time()
print('测试耗时：', time_end-time_st)

answers = torch.cat(answers, 0)
targets = torch.cat(targets, 0)
outputs = np.concatenate(outputs, 0)
np.savez(args.save_scores, scores=outputs, labels=targets.cpu())
print('保存了结果...\n')
cf = confusion_matrix(targets.cpu(), answers.cpu()).astype(float)

cls_cnt = cf.sum(axis=1)  # 得到是每一类各自总评估次数.
all = np.sum(cls_cnt)
print('参与评估的样本总数{}\n'.format(all))
cls_hit = np.diag(cf)  # 每一类总的评估对的次数.
good = np.sum(cls_hit)
print('评估正确的总数{}\n'.format(good))
cls_acc = cls_hit / cls_cnt
print('各类正确率:\n{}'.format(cls_acc))
arracc = np.array(cls_acc)
lowtypes = list(np.argsort(arracc))[:20]
print('各类正确率平均值 {:.04f}%,   累计正确率{:.04f}\n'.format(np.mean(cls_acc) * 100, (good / all) * 100))
print('正确率最低的20种:', lowtypes)

