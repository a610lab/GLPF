import argparse
import math
import os
import random
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from pytorch_lightning import seed_everything
from operator import truediv
import torch.nn.functional as F

import torch
from torch import optim
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from model.resnest.torch import resnest50 as create_model

from tqdm import tqdm

# from Dataset.plaque import get_plaque
from dataset.pneumoniamnist import get_breastmnist
from utils.loss import ce_loss, consistency_loss, BarlowTwinsLoss
from utils.misc import AverageMeter, Accuracy
from model.ema import ModelEMA


def interleave(x, size):
    s = list(x.shape)
    p = x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])
    return p


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def split_data(args, data):
    data = de_interleave(data, 2 * args.mu + 1)
    data_x_w = data[:args.batch_size]  # 有标签数据
    data_u_w, data_u_s = data[args.batch_size:].chunk(2)  # 无标签数据 （弱增强 + 强增强）
    return data_x_w, data_u_w, data_u_s


def cosine_similarity(feats, feat_u):
    num = torch.mm(feats, feat_u.T)
    denom = torch.norm(feats) * torch.norm(feat_u)
    return (num / denom).T


def probs_adjust(sim, probs):
    all = sim * probs
    return truediv(all, all.sum(dim=1).reshape(-1, 1))


def tensor_knn(x_train, y_train, x_test, k):
    # 计算训练样本和测试样本之间的余弦相似度
    cos_sim = F.cosine_similarity(x_test.unsqueeze(1), x_train.unsqueeze(0), dim=2)
    # 获取与测试样本最相似的k个训练样本的索引
    cos_sim, indices = torch.topk(cos_sim, k=k, dim=1)
    # 根据索引获取最相似的k个训练样本的标签
    knn_labels = y_train[indices]
    # 对k个训练样本的标签进行投票，得到测试样本的预测标签
    pred_labels, _ = torch.mode(knn_labels, dim=1)
    return cos_sim, knn_labels, pred_labels


def adjust_function(sim_u_feat, mean_sim):
    adjust_vlaue = torch.exp(sim_u_feat * mean_sim)
    adjust_vlaue_norm = torch.nn.functional.normalize(adjust_vlaue, p=1, dim=1)
    return adjust_vlaue_norm


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader, model, optimizer, ema_model):
    device = args.device
    best_acc = 0
    model.train()
    alpha_one = torch.ones((args.num_classes,), dtype=torch.long).to(device)
    if args.usewarmup:
        print('----------------------stage one: warm up------------------------------------------')
        model.load_state_dict(torch.load(
            './warmupmodel/warmup_' + str(args.seed) + '_' + str(args.num_labeled) + '_.pth',
            map_location=device
        ))
        '''
        try:
            model.load_state_dict(torch.load('../../FullSupervised/model_resnet18_144_' + str(args.num_labeled) + '.pth',
                                             map_location=device))
        except:
            loss_num = torch.nan
            for epoch in range(args.warmup_epoch):
                sample_num = 0
                accu_loss = torch.zeros(1).to(device)  # 累计损失
                accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
                labeled_trainloader = tqdm(labeled_trainloader)
                for step, labeled_train in enumerate(labeled_trainloader):
                    _, inputs_x, targets_x = labeled_train
                    sample_num += inputs_x.shape[0]
                    logits, feats = model(inputs_x.to(device))
                    los_ce = (F.cross_entropy(logits, targets_x.type(torch.LongTensor).to(args.device),reduction='none')).mean()
                    pred_classes = torch.max(logits, dim=1)[1]
                    accu_num += torch.eq(pred_classes, targets_x.to(device)).sum()
                    accu_loss += los_ce.detach()
                    optimizer.zero_grad()
                    los_ce.backward()
                    optimizer.step()
                    labeled_trainloader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch + 1,
                                                                                           accu_loss.item() / (step + 1),
                                                                                           accu_num.item() / sample_num)
                if loss_num < los_ce:
                    los_ce = loss_num
                    save_mode_path = os.path.join('warmup.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    '''
        print('----------------------stage one: Finish------------------------------------------')
    print('----------------------stage two: semi-supervised------------------------------------------')
    for epoch in range(args.epochs):
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_u = AverageMeter()
        mask_w = AverageMeter()
        loss_u = BarlowTwinsLoss()
        features_dict = {}
        label_dict = {}
        p_bar = tqdm(range(args.eval_step), file=sys.stdout)
        num_classes = torch.zeros((args.num_classes,), dtype=torch.long).to(device)

        labeled_iter = iter(labeled_trainloader)
        unlabeled_iter = iter(unlabeled_trainloader)

        for batch_idx in range(args.eval_step):
            try:
                index_x, inputs_x, targets_x = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_trainloader)
                index_x, inputs_x, targets_x = next(labeled_iter)

            try:
                index_u, (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_trainloader)
                index_u, (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)

            batch_size = inputs_x.shape[0]
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1).to(device)
            targets_x = targets_x.to(args.device)
            logits, feats = model(inputs)
            logits_x_w, logits_u_w, logits_u_s = split_data(args, logits)
            feats_x_w, feats_u_w, feats_u_s = split_data(args, feats)

            del logits, feats
            # TODO 获取EMA均值
            # 根据字典存储 更新 特征 用于knn
            for i in range(batch_size):
                features_dict[index_x[i]] = feats_x_w[i]
                label_dict[index_x[i]] = targets_x[i]
            feature_tensors = [item.cpu().detach() for item in features_dict.values()]
            feats_all = torch.stack(feature_tensors).to(device)
            labels_all = torch.tensor(list(label_dict.values())).to(device)
            #中心特征修改
            all_feats = torch.zeros((args.num_classes, 128)).to(device)
            all_nums = torch.zeros((args.num_classes,)).to(device)
            for i in range(len(labels_all)):
                all_feats[labels_all[i]] = all_feats[labels_all[i]] + feats_all[i]
                all_nums[labels_all[i]] = all_nums[labels_all[i]] + 1

            per_feats = torch.nan_to_num(all_feats / all_nums.reshape(-1, 1))  # TODO 获取聚类中心

            sim_u_feat = cosine_similarity(per_feats, feats_u_w)  # TODO 计算样本特征与聚类中心相似度

            Lx = ce_loss(logits_x_w, targets_x.type(torch.LongTensor).to(args.device), reduction='mean')  # 有标签数据交叉熵

            pseudo_label_w = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
            pseudo_label_w_adjust = probs_adjust(sim_u_feat, pseudo_label_w)

            max_probs_w_m, targets_u_w_m = torch.max(pseudo_label_w_adjust, dim=-1)
            max_probs_w, targets_u_w = torch.max(pseudo_label_w, dim=-1)

            # todo 大于阈值的伪标签
            mask1 = max_probs_w_m.ge(args.threshold)
            #低于阈值的数据
            no_Pso = torch.logical_not(mask1.bool())

            # TODO 计算最近n个点相似度,以及n个点的标签
            # addd KNN  3 &  5   #(args.num_labeled // 75)
            cos_sim, knn_labels, pred_labels1 = tensor_knn(feats_all, labels_all, feats_u_w, args.k)#args.num_labeled//args.num_classes
            knn_labels = knn_labels.type(torch.LongTensor)
            same_P = pred_labels1.eq(targets_u_w).float()  # 模型预测与knn预测一致性
            cos_sim_count = torch.zeros_like(logits_u_w, dtype=torch.float32)  # 统计相似度累加和
            count = torch.zeros_like(logits_u_w, dtype=torch.int32)  # 统计类别累加和
            for step, i in enumerate(knn_labels):
                for step2, j in enumerate(i):
                    cos_sim_count[step][j] = cos_sim[step][step2].item() + cos_sim_count[step][j]
                    count[step][j] = count[step][j] + 1

            knn_prb = F.softmax(cos_sim_count, dim=-1)
            max_probs_knn, targets_u_knn = torch.max(knn_prb, dim=-1)
            same_K = max_probs_knn.ge(args.threshold).float()
            mask_end = no_Pso * same_P * same_K.to(device)
            # mask_end = same_P * no_Pso
            # 计算各个点平均值
            # mean_sim = cos_sim_count / (count + 1e-6)
            if args.use_focal:
                # todo 统计特征中心矫正 伪标签
                for step, i in enumerate(targets_u_w_m):
                    if max_probs_w_m[step] > args.threshold:
                        num_classes[i] = num_classes[i] + 1
                # todo 统计knn矫正 伪标签
                for step, i in enumerate(mask_end):
                    if i:
                        num_classes[targets_u_w[step]] = num_classes[targets_u_w[step]] + 1
                alpha = torch.tensor([
                    (1 - num_classes[0] / (num_classes.sum() + 1e-6)),
                    (1 - num_classes[1] / (num_classes.sum() + 1e-6))
                ]).to(device)
                # alpha_one = alpha_one * args.ema_decay + F.normalize(torch.exp(alpha), p=1, dim=0) * (1 - args.ema_decay)
                alpha_one = alpha_one * args.ema_decay + alpha * (1 - args.ema_decay)

                Lu = consistency_loss(logits_u_s, targets_u_w.type(torch.LongTensor).to(args.device), name='ce',
                                       mask=mask_end.float() + mask1.float(), weight=None, alpha=alpha_one.detach())
            else:
                Lu = consistency_loss(logits_u_s, targets_u_w.type(torch.LongTensor).to(args.device), name='ce',
                                       mask=mask_end.float() + mask1.float(), weight=None, alpha=None)

            Lu2 = loss_u(feats_u_w, feats_u_s)
            loss = Lx + Lu + 0.005 * Lu2
            loss.backward()
            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            mask_u.update(mask1.float().mean().item())
            mask_w.update(mask_end.float().mean().item())
            optimizer.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()
            optimizer.zero_grad()
            p_bar.set_description(
                "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. mask1 {mask:.4f}. mask2 {mask_w1:.4f} ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    mask=mask_u.avg,
                    mask_w1=mask_w.avg
                ))
            p_bar.update()
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
            torch.save(ema_to_save.state_dict() if args.use_ema else model_to_save.state_dict(), filepath)
    print('best acc is:{:.4f}'.format(best_acc))


def acc(args, test_loader, model):
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
    print('val loss is:{:.4f} val acc is:{:.4f}'.format(losses.avg, acces.avg))
    return losses.avg, acces.avg


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
    # 使用CUDA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    # 是否随机
    if args.seed is not None:
        set_seed(args.seed)

    # 获取数据集
    labeled_dataset, unlabeled_dataset, test_dataset = get_breastmnist(args)# 获取颈动脉数据集    #获取BUSI数据集

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
    # 计算总epoch
    args.epochs = math.ceil(args.total_steps / args.eval_step)

    model.zero_grad()

    if args.use_ema:
        ema_model = ModelEMA(args, model, args.ema_decay)

    train(args, labeled_trainloader, unlabeled_trainloader, test_loader, model, optimizer, ema_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='number of workers')
    parser.add_argument('--num-labeled', type=int, default=75,
                        help='number of labeled data')
    parser.add_argument('--total-steps', default=100 * 50, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=50, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--batch-size', default=8, type=int,
                        help='train batchsize')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--warmup_epoch', default=100, type=int,
                        help='warmup_epoch')
    parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float,
                        help='initial learning rate')  # 0.00005  0.0001
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--mu', default=2, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--seed', default=0, type=int,
                        help="random seed")
    parser.add_argument('--threshold', default=0.95, type=float)
    parser.add_argument('--save_name', default='K_Mean_Fea')
    parser.add_argument('--ave_class', default=True, type=bool)
    parser.add_argument('--k', default=3, type=int)
    parser.add_argument('--add_feats_u', default=False, type=bool)
    parser.add_argument('--usewarmup', default=True, type=bool, help='use pretraining')
    parser.add_argument('--use-focal', default=True, type=bool, help='use focal-loss')
    parser.add_argument('--data_path', default='pneumoniamnist_128.npz', type=str)
    args = parser.parse_args()
    for j in [0, 300, 500]:
        args.seed = j
        for i in [470, 1412, 2354]:
            args.num_labeled = i
            args.save_name = './glpf_pth/GLPF_' + str(args.seed) + '_' + str(args.num_labeled)
            main(args)
