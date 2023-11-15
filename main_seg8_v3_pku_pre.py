
import warnings
warnings.filterwarnings("ignore")
from ops import msst_seg8_v3_pre_replace_fc as network
# from ops import msst_seg8_v3_pre_new_fc_init as network
import os
import numpy as np
import time
import shutil
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm
from ops.dataset_sk_2d_seg_rand_pku_with_confidence import MSSTdata
from opts_sk import parser
from sklearn.metrics import confusion_matrix
best_prec1 = 0

modelstarttime = str(time.strftime("%Y%m%d%H"))
file_handle = open('logs/'+'log' + str(time.strftime("%Y-%m-%d-%H-%M")) + '_seg_' + str(parser.parse_args().seg) + '_V3_pre_' + parser.parse_args().model_name+ '_' + parser.parse_args().modality + '.txt', mode='w+')
def main():
    global args, best_prec1
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
    file_handle.write('开始时间:' + str(time.strftime("%Y-%m-%d-%H-%M")) + '\n')
    file_handle.write('dataset: ' + args.dataset + '\n')
    file_handle.write('初始学习率: ' + str(args.lr) + '\n')
    # file_handle.write('basemodel: ' + args.arch + '\n')
    file_handle.write('batchsize: ' + str(args.batch_size) + '\n')
    file_handle.write('学习率节点: ' + str(args.lr_steps) + '\n')
    file_handle.write('模型名字: ' + args.model_name + '\n')
    print('loading model...')
    model = network.Net(num_class=num_class, dropout=args.dropout)

    # policies = model.get_optim_policies()
    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()#Pytorch 的多 GPU 处理接口也可以指定gpu的id
    # model = torch.nn.DataParallel(model, device_ids=[1]).to(device)  # Pytorch 的多 GPU 处理接口也可以指定gpu的id
    # 例如model = torch.nn.DataParallel(model, device_ids=[0,1])
    print('model done!')
    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])#load_state_dict 是model或optimizer之后pytorch自动具备的函数,可以直接调用
            #pytorch 中的 state_dict 是一个简单的python的字典对象,将每一层与它的对应参数建立映射关系.(如model的每一层的weights及偏置等等)
            #(注意,只有那些参数可以训练的layer才会被保存到模型的state_dict中,如卷积层,线性层等等)
            #model.load_state_dict(checkpoint[‘state_dict’])是完成导入模型的参数初始化model这个网络的过程
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))#.format参考https://blog.csdn.net/jpch89/article/details/84099277
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    cudnn.benchmark = True#增加程序的运行效率
    print('loading val_data...')
    valdata = MSSTdata(dataroot=args.data_root, modality=args.modality,  test_mode=True, seg=8)
    val_loader=torch.utils.data.DataLoader(valdata, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    print('val_data done')
    print('loading train_data...')
    traindata = MSSTdata(dataroot=args.data_root, modality=args.modality, test_mode=False, seg=8)
    train_loader = torch.utils.data.DataLoader(traindata, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    print('train_data done')
    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,weight_decay=args.weight_decay)  # momentum动量。weight_decay权重衰减

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    for epoch in range(args.start_epoch, args.epochs):
        file_handle.write('epoch:' + str(epoch) + '\n')
        adjust_learning_rate(optimizer, epoch, args.lr_steps)#依据epoch次数更新学习率。

        # train(train_loader, model, criterion,criterion1, optimizer, epoch)#用centerloss的时候用
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            with torch.no_grad():
                # prec1, needenhance = validate(val_loader, model, criterion, criterion1, (epoch + 1) * len(train_loader))#用centerloss的时候用
                prec1, needenhance = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader))
                # remember best prec@1 and save checkpoint
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                }, is_best)
            file_handle.write('best_pre：' + str(round(best_prec1, 4)) + '\n\n')
            print('best_prec:', best_prec1, '\n')
    file_handle.write('结束时间:' + str(time.strftime("%Y-%m-%d-%H-%M")) + '\n')

# def train(train_loader, model, criterion, criterion1, optimizer, epoch):
def train(train_loader, model, criterion, optimizer, epoch):
    timest = time.time()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # lossescen = AverageMeter()
    # lossescro = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    file_handle.write('当前学习率：' + str(optimizer.param_groups[-1]['lr']) + '\n')  #
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        output = model(input_var)
        # print('trainoutput',output)
        loss = criterion(output, target_var)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))[0]
        losses.update(loss.item(), input.size(0))

        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()#只有用了optimizer.step()，模型才会更新

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.8f}  '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})  '
                  'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})  '
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))
            file_handle.write('Epoch:'+str(epoch)+'\t'+str(i)+'/'+str(len(train_loader))+'\t'+ 'top1: ' + str(round(top1.avg, 4)) + '\t' + 'top5: ' + str(round(top5.avg, 4)) + '\t' + 'trainloss: ' + str(round(losses.avg, 4)) + '\n')
            file_handle.flush()
        # time.sleep(0.3)
    timeend = time.time()
    print('本轮耗时：', timeend - timest, '\n')
    file_handle.write('本轮耗时：' + str(timeend - timest) + '\n\n')

def validate(val_loader, model, criterion, iter, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    # model.apply(apply_dropout)
    end = time.time()

    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        output = model(input_var)
        loss = criterion(output, target_var)
        (prec1, prec5), target1, right = accuracy(output.data, target, topk=(1, 5))
        if i == 0:
            rights = right
            targets = target1
        rights = torch.cat([rights, right])
        targets = torch.cat([targets, target1])

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5)))
    # print(targets)
    # print(rights)
    cf = confusion_matrix(targets.cpu(), rights.cpu()).astype(float)

    cls_cnt = cf.sum(axis=1)  # 得到是每一类各自总评估次数.
    all = np.sum(cls_cnt)
    print('参与评估的样本总数{}'.format(all))
    cls_hit = np.diag(cf)  # 每一类总的评估对的次数.
    good = np.sum(cls_hit)
    print('评估正确的总数{}'.format(good))
    cls_acc = cls_hit / cls_cnt
    print('各类正确率:\n{}'.format(cls_acc))
    arracc = np.array(cls_acc)
    lowtypes=list(np.argsort(arracc))[:20]
    print('累计总数算出的正确率{}'.format(good / all))
    acc_avg = np.mean(cls_acc)*100
    print('各类正确率平均值{}'.format(acc_avg))
    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'.format(top1=top1, top5=top5, loss=losses)))
    file_handle.write('val_result\n')
    file_handle.write('本次测试结果：'+'top1: ' + str(round(top1.avg, 4)) + '\t' + 'top5: ' + str(round(top5.avg, 4)) + '\t' + 'val_loss: ' + str(round(losses.avg, 4)) + '\n\n')
    file_handle.flush()

    return acc_avg, lowtypes


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = 'checkpoint/'+'_'.join((modelstarttime, args.model_name, 'seg', str(args.seg), args.modality.lower(), filename))
    torch.save(state, filename)
    if is_best:
        best_name = 'checkpoint/'+'_'.join((modelstarttime, args.model_name, 'seg', str(args.seg), args.modality.lower(), 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)


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


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))#lr_steps是[30, 60] epoch大于几个就返回几
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = decay


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # print(pred)
    # print(pred.size())
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    right = pred[0]

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        # print(correct_k)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, target, right


if __name__ == '__main__':
    main()
