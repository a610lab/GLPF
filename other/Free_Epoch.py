import argparse
import sys
import math
import os
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from tqdm import tqdm
from pytorch_lightning import seed_everything
from dataset.pneumoniamnist import get_breastmnist
from utils.loss import ce_loss
from model.resnest.torch import resnest50 as create_model


from utils.misc import AverageMeter, Accuracy

from utils.losses import SelfAdaptiveFairnessLoss, ConsistencyLoss

best_acc = 0


class SelfAdaptiveThresholdLoss:

    def __init__(self, sat_ema):
        self.sat_ema = sat_ema
        self.criterion = ConsistencyLoss()

    @torch.no_grad()
    def __update__params__(self, logits_ulb_w, tau_t, p_t, label_hist):
        # Updating the histogram for the SAF loss here so that I dont have to call the torch.no_grad() function again.
        # You can do it in the SAF loss also, but without accumulating the gradient through the weak augmented logits

        probs_ulb_w = torch.softmax(logits_ulb_w, dim=-1)
        max_probs_w, max_idx_w = torch.max(probs_ulb_w, dim=-1)
        tau_t = tau_t * self.sat_ema + (1. - self.sat_ema) * max_probs_w.mean()
        p_t = p_t * self.sat_ema + (1. - self.sat_ema) * probs_ulb_w.mean(dim=0)
        histogram = torch.bincount(max_idx_w, minlength=p_t.shape[0]).to(p_t.dtype)
        label_hist = label_hist * self.sat_ema + (1. - self.sat_ema) * (histogram / histogram.sum())
        return tau_t, p_t, label_hist

    def __call__(self, targets_u_w, logits_ulb_w, logits_ulb_s, tau_t, p_t, label_hist):
        tau_t, p_t, label_hist = self.__update__params__(logits_ulb_w, tau_t, p_t, label_hist)

        logits_ulb_w = logits_ulb_w.detach()
        probs_ulb_w = torch.softmax(logits_ulb_w, dim=-1)

        x = torch.zeros(len(logits_ulb_w), ).to(logits_ulb_w.device)
        for step, i in enumerate(targets_u_w):
            x[step] = probs_ulb_w[step][i]

        max_probs_w, max_idx_w = torch.max(probs_ulb_w, dim=-1)
        tau_t_c = (p_t / torch.max(p_t, dim=-1)[0])
        mask = x.ge(tau_t * tau_t_c[max_idx_w]).to(logits_ulb_w.dtype)

        loss = self.criterion(logits_ulb_s, targets_u_w, mask=mask)

        return loss, mask, tau_t, p_t, label_hist


def cosine_similarity(feats, feat_u):
    num = torch.mm(feats, feat_u.T)
    denom = torch.norm(feats) * torch.norm(feat_u)
    return (num / denom).T


def split_data(args, data):
    data = de_interleave(data, 2 * args.mu + 1)
    data_x_w = data[:args.batch_size]  # 有标签数据
    data_u_w, data_u_s = data[args.batch_size:].chunk(2)  # 无标签数据 （弱增强 + 强增强）
    return data_x_w, data_u_w, data_u_s


def interleave(x, size):
    s = list(x.shape)
    p = x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])
    return p


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def set_seed(seed):
    seed_everything(seed)
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
    print(device)
    if args.seed is not None:
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
    args.epochs = math.ceil(args.total_steps / args.eval_step)

    if args.use_ema:
        from model.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    model.zero_grad()
    if args.usewarmup:
        print('----------------------stage one: warm up------------------------------------------')
        model.load_state_dict(torch.load('./warmupmodel/warmup_' + str(args.seed) + '_' + str(args.num_labeled) + '_.pth',
                                         map_location=device))
        print('----------------------stage one: Finish------------------------------------------')
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model)


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model):
    device = args.device

    sat_criterion = SelfAdaptiveThresholdLoss(args.ema_decay)
    saf_criterion = SelfAdaptiveFairnessLoss()
    p_t = (torch.ones(args.num_classes) / args.num_classes).to(device)
    label_hist = (torch.ones(args.num_classes) / args.num_classes).to(device)
    tau_t = p_t.mean()

    best_acc = 0
    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()

        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step))
        for batch_idx in range(args.eval_step):
            labeled_iter = iter(labeled_trainloader)
            _, inputs_x, targets_x = next(labeled_iter)

            unlabeled_iter = iter(unlabeled_trainloader)
            _, (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)

            batch_size = inputs_x.shape[0]
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1).to(args.device)
            targets_x = targets_x.to(args.device)
            logits, _ = model(inputs)
            logits_x_w, logits_u_w, logits_u_s = split_data(args, logits)
            del logits, _

            Lx = ce_loss(
                logits_x_w,
                targets_x.type(torch.LongTensor).to(args.device),
                reduction='mean'
            )

            pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)
            _, targets_u_w = torch.max(pseudo_label, dim=-1)

            loss_sat, mask, tau_t, p_t, label_hist = sat_criterion(
                targets_u_w, logits_u_w, logits_u_s, tau_t, p_t, label_hist
            )

            loss_saf, hist_p_ulb_s = saf_criterion(mask, logits_u_s, p_t, label_hist)
            loss = Lx + args.ulb_loss_ratio * loss_sat + args.ent_loss_ratio * loss_saf
            loss.backward()
            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update((args.ulb_loss_ratio * loss_sat + args.ent_loss_ratio * loss_saf).item())
            optimizer.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description(
                    "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. ".format(
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
        if best_acc < test_acc:
            best_acc = test_acc
            print('best acc is:{:.4f}'.format(best_acc))
            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            filepath = os.path.join(args.save_name + '.pth')
            torch.save(
                ema_to_save.state_dict() if args.use_ema else model_to_save.state_dict(), filepath)
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
            outputs, _ = model(inputs)
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

    return losses.avg, acces.avg


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
    parser.add_argument('--ulb_loss_ratio', default=1, type=float)
    parser.add_argument('--ent_loss_ratio', default=0.1, type=float)
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=0, type=int,
                        help="random seed")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    parser.add_argument('--save_name', default='FreeMatch')
    parser.add_argument('--ave_class', default=False, type=bool)
    parser.add_argument('--threshold', default=0.95, type=float)
    parser.add_argument('--train_path', default='../data_csv/train_data.csv')
    parser.add_argument('--val_path', default='../data_csv/validate_data.csv')
    parser.add_argument('--usewarmup', default=True, type=bool, help='use pretraining')
    parser.add_argument('--data_path', default='pneumoniamnist_128.npz', type=str)
    args = parser.parse_args()
    for j in [0]:
        args.seed = j
        for i in [470, 1412, 2354]:
            args.num_labeled = i
            args.save_name = './freematch_pth/FreeMatch_' + str(j) + '_' + str(i)
            main(args)
