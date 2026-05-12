import argparse
import os
import random
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch.nn.functional as F
import torch
from pytorch_lightning import seed_everything
from torch import optim, nn
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from model.resnest.torch import resnest50 as create_model
from tqdm import tqdm

from utils.misc import Accuracy
from dataset.pneumoniamnist import get_breastmnist


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    if args.seed is not None:
        seed_everything(args.seed)
        set_seed(args.seed)

    labeled_dataset, unlabeled_dataset, test_dataset = get_breastmnist(args)

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=RandomSampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=RandomSampler(unlabeled_dataset),
        batch_size=args.batch_size * args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    model = create_model(num_classes=args.num_classes).to(args.device)
    parameter = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(parameter, lr=args.lr)
    model.zero_grad()

    if args.use_ema:
        from model.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)
    if args.usewarmup:
        print('----------------------stage one: warm up------------------------------------------')
        model.load_state_dict(torch.load('./warmupmodel/warmup_' + str(args.seed) + '_' + str(args.num_labeled) + '_.pth',
                                         map_location=device))
        print('----------------------stage one: Finish------------------------------------------')

    train(args, labeled_trainloader, unlabeled_trainloader, test_loader, model, optimizer, ema_model)


def disable_batchnorm_tracking(model):
    def fn(module):
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.track_running_stats = False

    model.apply(fn)


def enable_batchnorm_tracking(model):
    def fn(module):
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.track_running_stats = True

    model.apply(fn)


def compute_source_loss(logits_weak, logits_strong, labels):
    """
    Receives logits as input (dense layer outputs with no activation function)
    """
    loss_function = nn.CrossEntropyLoss()  # default: `reduction="mean"`
    weak_loss = loss_function(logits_weak, labels)
    strong_loss = loss_function(logits_strong, labels)

    # return weak_loss + strong_loss
    return (weak_loss + strong_loss) / 2


def compute_target_loss(pseudolabels, logits_strong, mask):
    """
    Receives logits as input (dense layer outputs with no activation function).
    `pseudolabels` are treated as ground truth (standard SSL practice).
    """
    loss_function = nn.CrossEntropyLoss(reduction="none")
    pseudolabels = pseudolabels.detach()  # remove from backpropagation

    loss = loss_function(logits_strong, pseudolabels)

    return (loss * mask).mean()


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader, model, optimizer, ema_model):
    # iters = max(len(labeled_trainloader), len(unlabeled_trainloader))
    iters = args.eval_step
    steps_per_epoch = iters
    total_steps = args.epochs * steps_per_epoch
    current_step = 0
    best_acc = 0
    for epoch in range(0, args.epochs):
        running_loss = 0.0
        model.train()
        p_bar = tqdm(range(iters), file=sys.stdout)
        for batch_idx in p_bar:
            labeled_iter = iter(labeled_trainloader)
            _, (source_weak, source_strong), source_labels = next(labeled_iter)
            unlabeled_iter = iter(unlabeled_trainloader)
            _, (target_weak, target_strong), _ = next(unlabeled_iter)
            data_combined = torch.cat([source_weak, source_strong, target_weak, target_strong], 0).to(args.device)
            source_combined = torch.cat([source_weak, source_strong], 0).to(args.device)

            source_total = source_combined.size(0)

            logits_combined = model(data_combined)
            logits_combined = logits_combined[0] if isinstance(logits_combined, (tuple, list)) else logits_combined

            # source weak + source strong 对应的 logits
            logits_source_p = logits_combined[:source_total]

            disable_batchnorm_tracking(model)  # 关闭 BN 统计
            logits_source_pp = model(source_combined)
            logits_source_pp = logits_source_pp[0] if isinstance(logits_source_pp, (tuple, list)) else logits_source_pp
            enable_batchnorm_tracking(model)  # 恢复 BN 统计

            lambd = torch.rand_like(logits_source_p).to(args.device)
            final_logits_source = (lambd * logits_source_p) + ((1 - lambd) * logits_source_pp)

            # todo 将有标签数据结果（weak） 变为 概率
            logits_source_weak = final_logits_source[:source_weak.size(0)]
            pseudolabels_source = F.softmax(logits_source_weak, 1)

            ## softmax for logits of weakly augmented target images
            # todo 将无标签数据结果（weak） 变为 概率
            logits_target = logits_combined[source_total:]
            logits_target_weak = logits_target[:target_weak.size(0)]
            pseudolabels_target = F.softmax(logits_target_weak, 1)

            ## allign target label distribtion to source label distribution
            expectation_ratio = (1e-6 + torch.mean(pseudolabels_source)) / (1e-6 + torch.mean(pseudolabels_target))
            final_pseudolabels = F.normalize((pseudolabels_target * expectation_ratio), p=2,
                                             dim=1)  # L2 normalization # todo 按行进行标准化

            # perform relative confidence thresholding
            # todo 有标签数据 每个batch 最大 概率 均值
            row_wise_max, _ = torch.max(pseudolabels_source, dim=1)
            final_sum = torch.mean(row_wise_max, 0)

            ## define relative confidence threshold
            # todo  乘以 超参 default 0.9
            c_tau = args.tau * final_sum
            # todo 将 c_tau 作为阈值
            max_values, _ = torch.max(final_pseudolabels, dim=1)
            mask = (max_values >= c_tau).float()
            source_loss = compute_source_loss(logits_source_weak,
                                              final_logits_source[source_weak.size(0):], source_labels.type(torch.LongTensor).to(args.device))

            # todo 与 FixMatch 类似
            final_pseudolabels = torch.max(final_pseudolabels, 1)[1]  # argmax   todo 伪标签
            target_loss = compute_target_loss(final_pseudolabels, logits_target[target_weak.size(0):],
                                              mask)

            ## compute target loss weight (mu)  todo 动态变化无标签数据loss超参数
            pi = torch.tensor(np.pi, dtype=torch.float).to(args.device)
            step = torch.tensor(current_step, dtype=torch.float).to(args.device)
            mu = 0.5 - torch.cos(torch.minimum(pi, (2 * pi * step) / total_steps)) / 2

            ## get total loss
            loss = source_loss + (mu * target_loss)
            current_step += 1

            model.zero_grad()
            optimizer.zero_grad()
            # backpropagate and update weights
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            p_bar.desc = "[train epoch {}] loss: {:.3f}".format(epoch + 1, running_loss / (batch_idx + 1))

            if args.use_ema:
                ema_model.update(model)

        p_bar.close()
        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        test_loss, test_acc = acc(args, test_loader, test_model)

        if best_acc < test_acc:
            best_acc = test_acc
            print('best acc is:{:.4f}'.format(best_acc))
            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            filepath = os.path.join(args.save_name + '.pth')
            torch.save(
                ema_to_save.state_dict() if args.use_ema else model_to_save.state_dict()
                , filepath)


def acc(args, test_loader, model):
    running_loss = 0
    acces = 0
    with torch.no_grad():
        test_loader = tqdm(test_loader, file=sys.stdout)
        for batch_idx, (_, inputs, targets) in enumerate(test_loader):
            model.eval()
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            outputs = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            loss = F.cross_entropy(outputs, targets)
            acc = Accuracy(outputs, targets)
            running_loss += loss.item()
            acces += acc
            test_loader.desc = 'val loss is:{:.4f} val acc is:{:.4f}'.format(running_loss / (batch_idx + 1), acces / (batch_idx + 1))
    # print()
    return running_loss / (batch_idx + 1), acces / (batch_idx + 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--num_labeled', default=75, type=int)
    parser.add_argument('--eval-step', default=50, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--batch-size', default=8, type=int,
                        help='train batchsize')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=2, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='number of workers')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tau', type=float, default=0.90, help='unsupervised s')
    parser.add_argument('--save_name', default='AdaMatch')
    parser.add_argument('--train_path', default='../../data_csv/train_data.csv')
    parser.add_argument('--val_path', default='../../data_csv/validate_data.csv')
    parser.add_argument('--usewarmup', default=True, type=bool, help='use pretraining')
    parser.add_argument('--data_path', default='pneumoniamnist_128.npz', type=str)
    args = parser.parse_args()
    for j in [0]:
        args.seed = j
        for i in [470, 1412, 2354]:
            args.num_labeled = i
            args.save_name = './adamatch_pth/AdaMatch_' + str(args.seed) + '_' + str(args.num_labeled)
            main(args)
