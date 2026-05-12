import argparse
import csv
import random
import sys
import time
import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import os
from model.resnest.torch import resnest50 as create_model
from tqdm import tqdm
from pytorch_lightning import seed_everything
from loss import SemiLoss

from dataset.pneumoniamnist_mixmatch import get_breastmnist
from utils.misc import AverageMeter, Accuracy
import numpy as np
'''add ema'''

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

@torch.no_grad()
def validate(model, labeled_dataloader, criterion, epoch):
    losses = AverageMeter()
    acces = AverageMeter()

    #labeled_dataloader = tqdm(labeled_dataloader, file=sys.stdout)
    with torch.no_grad():
        for step, (_, inputs, targets) in enumerate(labeled_dataloader):
            model.eval()
            outputs = model(inputs.to(args.device))
            outputs = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            loss = criterion(outputs, targets.to(args.device))
            p1 = Accuracy(outputs, targets.to(args.device))
            losses.update(loss.item(), inputs.size(0))
            acces.update(p1, inputs.size(0))
            #labeled_dataloader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch+1, losses.avg, acces.avg)
    return losses.avg, acces.avg


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


def train(model, labeled_dataloader, unlabeled_dataloader,  optimizer, val_dataloader, criterion, ema_model,
           T, lambda_u, epoch, num_steps):
    # fid_csv = open(args.fid_path+'.csv' , 'w', encoding='utf-8')
    # csv_writer = csv.writer(fid_csv)
    # csv_writer.writerow(["epoch", "train_loss", "train_loss_x", "train_loss_u", "val_acc", "val_loss", ])
    best_acc = 0
    criterion_u = SemiLoss()
    model.train()
    for e in range(epoch):
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        ws = AverageMeter()

        p_bar = tqdm(range(num_steps), file=sys.stdout)

        for s in range(num_steps):
            # 加载数据，已经是数据增强完了的
            lbl_iter = iter(labeled_dataloader)
            _, inputs_x, targets_x = next(lbl_iter)
            inputs_x = inputs_x.to(args.device)
            targets_x = targets_x.type(torch.LongTensor).to(args.device)

            ulbl_iter = iter(unlabeled_dataloader)
            _, (inputs_us1, inputs_us2), _ = next(ulbl_iter)

            inputs_us1 = inputs_us1.to(args.device)
            inputs_us2 = inputs_us2.to(args.device)
            batch_size = inputs_x.size(0)
            # Transform label to one-hot
            targets_x = torch.zeros(batch_size, args.num_classes).to(args.device).scatter_(1, targets_x.view(-1, 1), 1)

            with torch.no_grad():
                outputs_u = model(inputs_us1)
                outputs_u = outputs_u[0] if isinstance(outputs_u, (tuple, list)) else outputs_u

                outputs_u2 = model(inputs_us2)
                outputs_u2 = outputs_u2[0] if isinstance(outputs_u2, (tuple, list)) else outputs_u2

                p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2

                #sharpen
                pt = p ** (1 / T)
                targets_u = pt / pt.sum(dim=1, keepdim=True)
                targets_u = targets_u.detach()

            #mix up
            all_inputs = torch.cat([inputs_x, inputs_us1, inputs_us2], dim=0)
            all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)
            l = np.random.beta(args.alpha, args.alpha)
            l = max(l, 1 - l)
            idx = torch.randperm(all_inputs.size(0))
            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]
            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b

            mixed_input = list(torch.split(mixed_input, batch_size))
            mixed_input = interleave(mixed_input, batch_size)

            first_out = model(mixed_input[0])
            first_out = first_out[0] if isinstance(first_out, (tuple, list)) else first_out
            logits = [first_out]

            for input in mixed_input[1:]:
                out = model(input)
                out = out[0] if isinstance(out, (tuple, list)) else out
                logits.append(out)

            logits = interleave(logits, batch_size)
            logits_x = logits[0]
            logits_u = torch.cat(logits[1:], dim=0)

            lx, lu, w = criterion_u(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:],
                                  e + s / num_steps, lambda_u, args.epoch)

            loss = lx + lu * w

            # record loss
            losses.update(loss.item(), batch_size)
            losses_x.update(lx.item(), batch_size)
            losses_u.update(lu.item(), batch_size)
            ws.update(w, batch_size)
            # compute gradient and do SGD step
            model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.use_ema:
                ema_model.update(model)
            p_bar.desc = "[train epoch {}/{}] Iter: {}/{} lr: {:.4f} ,loss: {:.3f},loss_x: {:.3f},loss_u: {:.3f}".format(e + 1, args.epoch, s + 1, num_steps, args.lr, losses.avg, losses_x.avg, losses_u.avg)
            p_bar.update()
        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model
        val_loss, val_acc = validate(test_model, val_dataloader, criterion, e)
        p_bar.close()
        print('val_loss is{:.3f},val_acc is{:.3f}'.format(val_loss,val_acc))
        # csv_writer.writerow([e, losses.avg, losses_x.avg, losses_u.avg, val_acc, val_loss])
        if val_acc > best_acc:
            best_acc = val_acc
            print('best acc is :{:.4f}'.format(best_acc))
            save_path = os.path.join( args.fid_path+'.pth')
            torch.save(test_model.state_dict(), save_path)

    # fid_csv.close()


def main(args):
    #device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    if args.seed is not None:
        # seed_everything(args.seed)
        set_seed(args.seed)
    # datasets
    labeled_dataset, unlabeled_dataset, test_dataset = get_breastmnist(args)
    train_sampler = RandomSampler
    args.ulb_dset = len(unlabeled_dataset)
    args.num_workers = 0
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
    if args.use_ema:
        from model.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.usewarmup:
        print('----------------------stage one: warm up------------------------------------------')
        model.load_state_dict(torch.load('./warmupmodel/warmup_' + str(args.seed) + '_' + str(args.num_labeled) + '_.pth',
                                         map_location=device))
        print('----------------------stage one: Finish------------------------------------------')
    train(model, labeled_trainloader, unlabeled_trainloader, optimizer, test_loader, criterion, ema_model,
          T=args.T, lambda_u=args.lambda_u, epoch=args.epoch, num_steps=args.num_steps )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--dp', default=0.0, type=float, help='dropout')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num-labeled', type=int, default=225,
                        help='number of labeled data')
    parser.add_argument('--mu', default=2, type=int)
    parser.add_argument('--alpha', default=0.75, type=float)
    parser.add_argument('--lambda-u', default=75, type=float)
    parser.add_argument('--T', default=0.5, type=float)
    parser.add_argument('--k', default=2, type=int)
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--ave_class', default=False, type=bool)
    parser.add_argument('--epoch', default=150, type=int, help='epoch')
    parser.add_argument('--num_steps', default=50, type=int, help='num_steps')
    parser.add_argument('--train_path', default='../../data_csv/train_data.csv')
    parser.add_argument('--val_path', default='../../data_csv/validate_data.csv')
    parser.add_argument('--fid-path', default='MixMatch_1', type=str)
    parser.add_argument('--data_path', default='pneumoniamnist_128.npz', type=str)
    parser.add_argument('--seed', default=900, type=int,
                        help="random seed")
    parser.add_argument('--usewarmup', default=True, type=bool, help='use pretraining')
    args = parser.parse_args()
    # time.sleep(13*6*260)
    for j in [0]:
        args.seed = j
        for i in [470, 1412, 2354]:
            args.num_labeled =i
            args.fid_path = './mixmatch_pth/MixMatch_' + str(args.seed) + '_' + str(args.num_labeled)
            main(args)
