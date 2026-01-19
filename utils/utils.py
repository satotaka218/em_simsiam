import math
import numpy as np
import torch

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr

def accuracy(output, target, topk = (1,)) :
    " Compute the accuracy over the k top predictions for the specified values of k "

    with torch.no_grad() :
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def make_confusion_matrix(label_list: list, predict_list: list) :
    cifer10_class_list = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    result_confusion_matrix = confusion_matrix(predict_list, label_list)
    result_confusion_matrix = result_confusion_matrix / np.sum(result_confusion_matrix, axis = 0)

    fig = plt.figure(figsize=(20, 10))
    graph_confusion_matrix = sns.heatmap(result_confusion_matrix, cmap = 'Blues', annot= True)

    graph_confusion_matrix.set( xlabel = "label", ylabel = "predict",xticklabels=cifer10_class_list, yticklabels=cifer10_class_list)

    fig.savefig('./result_figure/simple_pretrained_resnet_confusion_matrix.png')
    plt.show()

