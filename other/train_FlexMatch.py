'''
 1.数据改写csv  √
 2.加课程伪标签  √
'''

import sys
import argparse
import csv
import math
import os
import random
import shutil
import time
from collections import Counter
from pytorch_lightning import seed_everything
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dataset.pneumoniamnist import get_breastmnist
from model.resnest.torch import resnest50 as create_model
from utils.misc import AverageMeter, Accuracy

best_acc = 0





def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# def get_cosine_schedule_with_warmup(optimizer,
#                                     num_warmup_steps,
#                                     num_training_steps,
#                                     num_cycles=7./16.,
#                                     last_epoch=-1):
#     def _lr_lambda(current_step):
#         if current_step < num_warmup_steps:
#             return float(current_step) / float(max(1, num_warmup_steps))
#         no_progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
#         return max(0., math.cos(math.pi * num_cycles * no_progress))

    # return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    if args.seed is not None:
        seed_everything(args.seed)
        set_seed(args.seed)

    labeled_dataset, unlabeled_dataset, test_dataset = get_breastmnist(args)
    train_sampler = RandomSampler
    args.ulb_dset = len(unlabeled_dataset)
    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    model = create_model(num_classes=args.num_classes).to(args.device)

    parameter = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.SGD(parameter, lr=args.lr,
    #                       momentum=0.9, nesterov=args.nesterov)
    optimizer = optim.Adam(parameter, lr=args.lr)
    args.epochs = math.ceil(args.total_steps / args.eval_step)

    if args.use_ema:
        from model.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    model.zero_grad()
    if args.usewarmup:
        print('----------------------stage one: warm up------------------------------------------')
        model.load_state_dict(
            torch.load('./warmupmodel/warmup_' + str(args.seed) + '_' + str(args.num_labeled) + '_.pth',
                       map_location=device))
        print('----------------------stage one: Finish------------------------------------------')
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model)




def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model):
    fid_csv = open(args.save_name+'.csv', 'w', encoding='utf-8')
    csv_writer = csv.writer(fid_csv)
    csv_writer.writerow(["parameters", "batch_size", "learning_rate", "epochs"])
    csv_writer.writerow([sum(p.numel() for p in model.parameters())/1000000.0, args.batch_size, args.lr, args.epochs])
    csv_writer.writerow(["epoch", "train_loss", "train_loss_x", "train_loss_u", "mask_probs", "val_acc", "val_loss", ])
    best_acc = 0
    end = time.time()

    model.train()

    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()

        # 改课程伪标签
        # 1.每个epoch将所有弱增强数据都标记为未选择
        selectd_labels = torch.ones((args.ulb_dset,), dtype=torch.long, ) * -1
        selectd_labels = selectd_labels.to(args.device)

        # 2.用于统计被标记类别数量
        class_acc = torch.zeros((args.num_classes,)).to(args.device)

        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step))
        for batch_idx in range(args.eval_step):
            labeled_iter = iter(labeled_trainloader)
            _, inputs_x, targets_x = next(labeled_iter)

            unlabeled_iter = iter(unlabeled_trainloader)
            inputs_u_w_idx, (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
            inputs_u_w_idx = inputs_u_w_idx.to(args.device)
            ##3.根据 class_acc 统计各个类别数量
            pseudo_counter = Counter(selectd_labels.tolist())

            for i in range(args.num_classes):
                class_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())

            batch_size = inputs_x.shape[0]
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.device)
            targets_x = targets_x.to(args.device)
            outputs = model(inputs)
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            logits = de_interleave(logits, 2 * args.mu + 1)
            logits_x = logits[:batch_size]#有标签数据
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)#无标签数据 （弱增强 + 强增强）
            del logits

            Lx = F.cross_entropy(logits_x, targets_x.type(torch.LongTensor).to(args.device), reduction='mean')#有标签数据

            pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)#弱增强数据 one-hot
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            #课程伪标签标记mask
            mask = max_probs.ge(args.p_cutoff * (class_acc[targets_u] / (2. - class_acc[targets_u]))).float()
            select = max_probs.ge(args.p_cutoff).long()#伪标签大于0.95才会被标记

            if inputs_u_w_idx[select == 1].nelement() != 0:
                selectd_labels[inputs_u_w_idx[select == 1]] = targets_u[select == 1]


            #mask = max_probs.ge(args.threshold).float()#筛选 ont-hot > threshold数据
            '''
            2.将 mask 中的 threshold 改为 课程伪标签
            '''
            Lu = (F.cross_entropy(logits_u_s, targets_u,
                                  reduction='none') * mask).mean()

            loss = Lx + args.lambda_u * Lu
            loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            optimizer.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    lr=args.lr,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    mask=mask_probs.avg))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        test_loss, test_acc = test(args, test_loader, test_model)

        csv_writer.writerow([epoch, losses.avg, losses_x.avg, losses_u.avg, mask_probs.avg,test_acc,test_loss])

        if best_acc < test_acc:
            best_acc = test_acc
            print('best acc is:{:.4f}'.format(best_acc))
            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            filepath = os.path.join(args.save_name + '.pth')
            torch.save({
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
            }, filepath)

    fid_csv.close()
    print('best acc is:{:.4f}'.format(best_acc))



def test(args, test_loader, model):
    losses = AverageMeter()
    acces = AverageMeter()

    if not args.no_progress:
        test_loader = tqdm(test_loader)

    with torch.no_grad():
        for batch_idx, (_, inputs, targets) in enumerate(test_loader):
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            outputs = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            loss = F.cross_entropy(outputs, targets)
            acc = Accuracy(outputs, targets)

            losses.update(loss.item(), inputs.shape[0])
            acces.update(acc, inputs.shape[0])

            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Loss: {loss:.4f}. acc: {acc:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    loss=losses.avg,
                    acc=acces.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    return losses.avg,acces.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MyMatch Training')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='number of workers')
    parser.add_argument('--num-labeled', type=int, default=75,
                        help='number of labeled data')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--total-steps', default=150 * 50, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=50, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=8, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=2, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    # parser.add_argument('--csv_save_path', default='Flex_data_resnest50_144_8_1.csv')
    parser.add_argument('--p_cutoff', default=0.95, type=float)
    parser.add_argument('--save_name', default='flexmatch', type=str)
    parser.add_argument('--ave_class', default=False, type=bool)
    parser.add_argument('--train_path', default='../data_csv/train_data.csv')
    parser.add_argument('--val_path', default='../data_csv/validate_data.csv')
    parser.add_argument('--usewarmup', default=True, type=bool, help='use pretraining')
    parser.add_argument('--data_path', default='pneumoniamnist_128.npz', type=str)
    args = parser.parse_args()
    for i in [0]:
        args.seed = i
        for j in [470, 1412, 2354]:
            args.num_labeled = j
            args.save_name = './flexmatch_pth/FlexMatch_' + str(args.seed) + '_' + str(args.num_labeled)
            main(args)
