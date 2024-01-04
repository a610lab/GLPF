import torch.nn.functional as F
import torch
from torch import Tensor


def Accuracy(input: Tensor, target: Tensor):
    input = F.softmax(input, dim=1)
    pred = torch.argmax(input, dim=1)
    correct_num = torch.eq(pred, target).sum().float().item()
    acc = correct_num/len(target)
    return acc


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

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







