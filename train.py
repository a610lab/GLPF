import argparse
import math
import os
import sys
from operator import truediv
import random
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from tqdm import tqdm
import torch.nn.functional as F
from GLPF.utils.loss import ce_loss, consistency_loss, BarlowTwinsLoss
from GLPF.utils.misc import AverageMeter, Accuracy
from GLPF.utils.GLPF import GlobalFeature, LocalFeature, PseudoBalance
from utils.ema import ModelEMA
from model.resnet18 import ResNet18_Model as create_model
from dataset.BUSI_read import get_plaque


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_data(args, data):
    s = list(data.shape)
    data = data.reshape([2 * args.mu + 1, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])
    data_x_w = data[:args.batch_size]
    data_u_w, data_u_s = data[args.batch_size:].chunk(2)
    return data_x_w, data_u_w, data_u_s


def cosine_similarity(feats, feat_u):
    num = torch.mm(feats, feat_u.T)
    denom = torch.norm(feats) * torch.norm(feat_u)
    return (num / denom).T


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    if args.seed is not None:
        set_seed(args.seed)
    labeled_dataset, unlabeled_dataset, test_dataset = get_plaque(args)

    labeled_loader = DataLoader(
        labeled_dataset,
        sampler=RandomSampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_loader = DataLoader(
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

    model = create_model(args=args, num_classes=args.num_classes, feature=args.feature_dim).to(args.device)
    parameter = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(parameter, lr=args.lr)
    args.epochs = math.ceil(args.total_steps / args.eval_step)
    model.zero_grad()
    if args.use_ema:
        ema_model = ModelEMA(args, model, args.ema_decay)
    train(args, labeled_loader, unlabeled_loader, test_loader, model, optimizer, ema_model)


def train(args, labeled_loader, unlabeled_loader, test_loader, model, optimizer, ema_model):
    device = args.device
    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_select = AverageMeter()
        GF = GlobalFeature(args)
        PB = PseudoBalance(args)
        loss_u = BarlowTwinsLoss()

        p_bar = tqdm(range(args.eval_step), file=sys.stdout)
        for batch_idx in p_bar:
            labeled_iter = iter(labeled_loader)
            index_x, inputs_x, targets_x = next(labeled_iter)
            targets_x = targets_x.to(device)
            unlabeled_iter = iter(unlabeled_loader)
            index_u, (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
            logits, feats = model(torch.cat((inputs_x, inputs_u_w, inputs_u_s)))
            logits_x_w, logits_u_w, logits_u_s = split_data(args, logits)
            feats_x_w, feats_u_w, feats_u_s = split_data(args, feats)
            del logits, feats
            GF.update(index_x, feats_x_w, targets_x)
            per_feats = GF.centerFeat()
            sim_u_feat = cosine_similarity(per_feats, feats_u_w)
            pseudo_label_w = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
            pseudo_label_w_adjust = truediv(sim_u_feat * pseudo_label_w, (sim_u_feat * pseudo_label_w).sum(dim=1).reshape(-1, 1))
            max_probs_w_a, targets_u_w_a = torch.max(pseudo_label_w_adjust, dim=-1)
            max_probs_w, targets_u_w = torch.max(pseudo_label_w, dim=-1)
            mask_a = max_probs_w_a.ge(args.threshold)
            no_Pso = torch.logical_not(mask_a.bool())
            mask_l = LocalFeature(GF.feats_all, GF.labels_all, feats_u_w, targets_u_w, args)
            mask_b = no_Pso * mask_l
            mask = mask_a + mask_b
            targets_u = mask_a * targets_u_w_a + mask_b * targets_u_w
            PB.update(mask_a, mask_b, targets_u_w_a, targets_u_w)
            Lx = ce_loss(logits_x_w, targets_x.type(torch.LongTensor).to(args.device), reduction='mean')
            Lp = consistency_loss(logits_u_s, targets_u.type(torch.LongTensor).to(args.device), name='ce',
                                  mask=mask, weight=None, alpha=PB.alpha)
            Lf = loss_u(feats_u_w, feats_u_s)
            Lu = args.lambda_u * Lp + args.lambda_f * Lf
            loss = Lx + Lu
            loss.backward()
            optimizer.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()
            optimizer.zero_grad()
            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            mask_select.update(mask.float().mean().item())
            p_bar.set_description(
                "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. mask {mask:.4f}.".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    mask=mask_select.avg
                ))
            p_bar.update()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        test_loss, test_acc = acc(args, test_loader, test_model, epoch)

        if best_acc < test_acc:
            best_acc = test_acc
            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            filepath = os.path.join(args.save_name + '.pth')
            torch.save(ema_to_save.state_dict() if args.use_ema else model_to_save.state_dict(), filepath)
    print('best val acc is:{:.4f}'.format(best_acc))


def acc(args, test_loader, model, epoch):
    losses = AverageMeter()
    acces = AverageMeter()
    test_loader = tqdm(test_loader, file=sys.stdout)
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
            test_loader.set_description(
                "Test Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. Loss: {loss:.4f}. acc {acc:.4f}.  ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    loss=losses.avg,
                    acc=acces.avg
                ))
            test_loader.update()
    return losses.avg, acces.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GLPF Training')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='number of workers')
    parser.add_argument('--num-labeled', type=int, default=282,
                        help='number of labeled data')
    parser.add_argument('--feature-dim', type=int, default=128,
                        help='number of feature dim')
    parser.add_argument('--total-steps', default=50 * 50, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=50, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--num_classes', default=3, type=int,
                        help='number of classes')
    parser.add_argument('--batch-size', default=8, type=int,
                        help='train batchsize')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--lr', '--learning-rate', default=5e-6, type=float,
                        help='initial learning rate')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--mu', default=2, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--lambda-f', default=5e-3, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--seed', default=700, type=int,
                        help="random seed")
    parser.add_argument('--threshold', default=0.95, type=float)
    parser.add_argument('--save_name', default='GLPF')
    parser.add_argument('--ave_class', default=True, type=bool)
    parser.add_argument('--k', default=7, type=int)
    parser.add_argument('--train_path', default='D:/BUSI/classification_BUSI_T3/train')
    parser.add_argument('--val_path', default='D:/BUSI/classification_BUSI_T3/validation')
    args = parser.parse_args()
    main(args)