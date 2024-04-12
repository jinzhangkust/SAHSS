from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms

import matplotlib.pyplot as plt


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]
    
    
class KDCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform_feature, transform_im):
        self.transform_im = transform_im
        self.transform_feature = transform_feature

    def __call__(self, x):
        hfn = transforms.functional.crop(x, i=0,j=0,h=300,w=1)
        x_a = transforms.functional.crop(x, i=0,j=1,h=300,w=300)
        '''plt.subplot(1,2,1)
        plt.imshow(x_a)
        #plt.show()
        plt.subplot(1,2,2)
        plt.imshow(x_b)
        plt.show()'''
        return [self.transform_feature(hfn), self.transform_im(x_a)]
    
    
class KDTwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform_feature, transform_A, transform_B):
        self.transform_feature = transform_feature
        self.transform_A = transform_A
        self.transform_B = transform_B

    def __call__(self, x):
        hfn = transforms.functional.crop(x, i=0,j=0,h=300,w=1)
        x_a = transforms.functional.crop(x, i=0,j=1,h=300,w=300)
        x_b = transforms.functional.crop(x, i=0,j=301,h=300,w=300)
        '''plt.subplot(1,2,1)
        plt.imshow(x_a)
        #plt.show()
        plt.subplot(1,2,2)
        plt.imshow(x_b)
        plt.show()'''
        return [self.transform_feature(hfn), self.transform_A(x_a), self.transform_B(x_b)]


class TwoCropTransform_:
    """Create two crops of the same image"""
    def __init__(self, transform_A, transform_B):
        self.transform_A = transform_A
        self.transform_B = transform_B

    def __call__(self, x):
        x_a = transforms.functional.crop(x, i=0,j=1,h=300,w=300)
        x_b = transforms.functional.crop(x, i=0,j=301,h=300,w=300)
        '''plt.subplot(1,2,1)
        plt.imshow(x_a)
        #plt.show()
        plt.subplot(1,2,2)
        plt.imshow(x_b)
        plt.show()'''
        return [self.transform_A(x_a), self.transform_B(x_b)]


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
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def compute_confusion_matrix(pred, truth):
    cm = torch.zeros([6,6])
    with torch.no_grad():
        for i in range(truth.size(0)):
            _, pred_label = pred[i].topk(1)
            truth_label = truth[i]#.topk(1)
            #print(pred_label)
            pred_label = int(pred_label)
            truth_label = int(truth_label)
            cm[truth_label, pred_label] = cm[truth_label, pred_label] + 1
            #print(cm)
    return cm
    
    
def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state
