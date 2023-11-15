import numpy as np


def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]


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
    # print('pppppppppppppppppppppppppp',pred)
    # print('ppppppppppppppppppppppppppsssssssssssssssss',pred.size())#5xbatchsize,是预测概率前五的标签。
    correct = pred.eq(target.view(1, -1).expand_as(pred))#对的就是Ture，错的是False，
    # print('ccccccccccccccccccccc', correct.size())#5xbatchsize，是ture和FALSE，
    # print('ccccccccooooooooooooooooooooo', correct)
    pred_top1 = pred[0]#1xbatchsize，预测概率最高的标签
    res = []
    for k in topk:
        # correct_k = correct[:k].view(-1).float().sum(0)#原版
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, pred_top1