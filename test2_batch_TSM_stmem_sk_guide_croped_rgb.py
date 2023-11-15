

import argparse
import time

import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
from ops.dataset_sk_croped_rgb_ntu import TSNDataSet
from ops.models2_stmem import TSN
from ops.transforms import *
from ops import dataset_config_video_sk_crop
from torch.nn import functional as F

# options
parser = argparse.ArgumentParser(description="TSM testing on the full validation set")
parser.add_argument('dataset', type=str)

# may contain splits
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--test_segments', type=str, default=25)
parser.add_argument('--dense_sample', default=False, action="store_true", help='use dense sample as I3D')
parser.add_argument('--twice_sample', default=False, action="store_true", help='use twice sample for ensemble')
parser.add_argument('--full_res', default=False, action="store_true",
                    help='use full resolution 256x256 for test as in Non-local I3D')

parser.add_argument('--test_crops', type=int, default=1)
parser.add_argument('--coeff', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')

# for true test
parser.add_argument('--test_list', type=str, default=None)
parser.add_argument('--csv_file', type=str, default=None)

parser.add_argument('--softmax', default=False, action="store_true", help='use softmax')

parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--img_feature_dim',type=int, default=256)
parser.add_argument('--num_set_segments',type=int, default=1,help='TODO: select multiply set of n-frames from a video')
parser.add_argument('--pretrain', type=str, default='imagenet')

parser.add_argument('--result_name', type=str, default='ntu60.npz')#用于命名要保存的结果数据

args = parser.parse_args()

def parse_shift_option_from_log_name(log_name):
    if 'shift' in log_name:
        strings = log_name.split('_')
        for i, s in enumerate(strings):
            if 'shift' in s:
                break
        return True, int(strings[i].replace('shift', '')), strings[i + 1]
    else:
        return False, None, None


weights_list = args.weights.split(',')
test_segments_list = [int(s) for s in args.test_segments.split(',')]
assert len(weights_list) == len(test_segments_list)
if args.coeff is None:
    coeff_list = [1] * len(weights_list)
else:
    coeff_list = [float(c) for c in args.coeff.split(',')]

if args.test_list is not None:
    test_file_list = args.test_list.split(',')
else:
    test_file_list = [None] * len(weights_list)


data_iter_list = []
net_list = []
modality_list = []

total_num = None
for this_weights, this_test_segments, test_file in zip(weights_list, test_segments_list, test_file_list):
    is_shift, shift_div, shift_place = parse_shift_option_from_log_name(this_weights)
    if 'RGB' in this_weights:
        modality = 'RGB'
    elif"depth" in this_weights:
        modality = 'depth'
    elif 'skmap' in this_weights:
        modality = "skmap"
    # this_arch = this_weights.split('TSM_')[1].split('_')[2]#原版
    # this_arch = this_weights.split('TSM_')[1].split('_')[3]#因为ntu_xsub中间有个‘_’。
    this_arch = 'resnet50'
    modality_list.append(modality)
    num_class, args.train_list, val_list, root_path, prefix = dataset_config_video_sk_crop.return_dataset(args.dataset,
                                                                                                          modality)
    print('=> shift: {}, shift_div: {}, shift_place: {}'.format(is_shift, shift_div, shift_place))
    net = TSN(num_class, this_test_segments if is_shift else 1, modality,
              base_model=this_arch,
              consensus_type=args.crop_fusion_type,
              img_feature_dim=args.img_feature_dim,
              pretrain=args.pretrain,
              is_shift=is_shift, shift_div=shift_div, shift_place=shift_place,
              non_local='_nl' in this_weights, data_length=6
              )

    if 'tpool' in this_weights:
        from ops.temporal_shift import make_temporal_pool
        make_temporal_pool(net.base_model, this_test_segments)  # since DataParallel

    checkpoint = torch.load(this_weights)
    best_pre1 = checkpoint['best_prec1']
    print('模型精度：', best_pre1)
    checkpoint = checkpoint['state_dict']

    # base_dict = {('base_model.' + k).replace('base_model.fc', 'new_fc'): v for k, v in list(checkpoint.items())}
    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
    replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
                    'base_model.classifier.bias': 'new_fc.bias',
                    }
    for k, v in replace_dict.items():
        if k in base_dict:
            base_dict[v] = base_dict.pop(k)

    net.load_state_dict(base_dict)

    input_size = net.scale_size if args.full_res else net.input_size
    if args.test_crops == 1:
        cropping = torchvision.transforms.Compose([
            GroupScale(net.scale_size),
            GroupCenterCrop(input_size),
        ])
    elif args.test_crops == 3:  # do not flip, so only 5 crops
        cropping = torchvision.transforms.Compose([
            GroupFullResSample(input_size, net.scale_size, flip=False)
        ])
    elif args.test_crops == 5:  # do not flip, so only 5 crops
        cropping = torchvision.transforms.Compose([
            GroupOverSample(input_size, net.scale_size, flip=False)
        ])
    elif args.test_crops == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(input_size, net.scale_size)
        ])
    else:
        raise ValueError("Only 1, 5, 10 crops are supported while we got {}".format(args.test_crops))

    data_loader = torch.utils.data.DataLoader(
            TSNDataSet(root_path, test_file if test_file is not None else val_list, num_segments=this_test_segments,
                       new_length=6 if modality in['skmap', 'depth', 'RGB'] else 6,
                       modality=modality,
                       image_tmpl=prefix,
                       test_mode=True,
                       remove_missing=len(weights_list) == 1,
                       transform=torchvision.transforms.Compose([
                           cropping,
                           Stack(roll=(this_arch in ['BNInception', 'InceptionV3'])),
                           ToTorchFormatTensor(div=(this_arch not in ['BNInception', 'InceptionV3'])),
                           GroupNormalize(net.input_mean, net.input_std),
                       ]), dense_sample=args.dense_sample, twice_sample=args.twice_sample,),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
    )

    if args.gpus is not None:
        devices = [args.gpus[i] for i in range(args.workers)]
    else:
        devices = list(range(args.workers))

    # net = torch.nn.DataParallel(net.cuda())
    net = torch.nn.DataParallel(net, device_ids=args.gpus).cuda()
    net.eval()

    total_num = len(data_loader.dataset)
    print('测试样本总数量', total_num)
    output = []
    #####################################################################################


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
    batch_time = AverageMeter()
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
        output = net(input_var)
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

        if i % 20 == 0:  # 20个batch打印一次。
            print(('Test: [{0}/{1}]\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(i, total_num // args.batch_size, batch_time=batch_time, top1=top1, top5=top5)))
    time_end = time.time()
    print('测试耗时：', time_end - time_st)

    answers = torch.cat(answers, 0)
    targets = torch.cat(targets, 0)
    outputs = np.concatenate(outputs, 0)
    np.savez(args.result_name, scores=outputs, labels=targets.cpu())
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

###################################################################################################################################

