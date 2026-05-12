import os
import sys
import random
import argparse

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from pytorch_lightning import seed_everything
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dataset.pneumoniamnist import get_breastmnist
from model.resnest.torch import resnest50 as create_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0.0


def evaluate(args, loader, model):
    losses = AverageMeter()
    all_targets = []
    all_preds = []

    model.eval()
    eval_loader = tqdm(loader, file=sys.stdout, colour='cyan') if not args.no_progress else loader

    with torch.no_grad():
        for _, inputs, targets in eval_loader:
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)

            outputs = model(inputs)
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

            loss = F.cross_entropy(logits, targets)
            preds = torch.argmax(logits, dim=1)

            losses.update(loss.item(), inputs.size(0))
            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    acc = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='binary', zero_division=0)
    kappa = cohen_kappa_score(all_targets, all_preds)

    return {
        "loss": losses.avg,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "kappa": kappa,
    }


def run_one_setting(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    print(f"\n===== Warmup | seed={args.seed}, num_labeled={args.num_labeled} =====")
    print("Using device:", device)

    seed_everything(args.seed)
    set_seed(args.seed)

    labeled_dataset, _, val_dataset = get_breastmnist(args)

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=RandomSampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False
    )

    model = create_model(num_classes=args.num_classes).to(device)
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    save_dir = "./warmupmodel"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"warmup_{args.seed}_{args.num_labeled}_.pth")

    best_acc = -1.0

    for epoch in range(args.epochs):
        model.train()
        train_losses = AverageMeter()

        train_loader = tqdm(labeled_trainloader, file=sys.stdout, colour='green') if not args.no_progress else labeled_trainloader

        for _, inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

            loss = F.cross_entropy(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.update(loss.item(), inputs.size(0))

            if not args.no_progress:
                train_loader.set_description(
                    f"Warmup Epoch: {epoch + 1:03d}/{args.epochs:03d} | Loss: {train_losses.avg:.4f}"
                )

        val_metrics = evaluate(args, val_loader, model)

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs:03d} | "
            f"Train Loss: {train_losses.avg:.4f} | "
            f"Val Acc: {val_metrics['acc']:.4f} | "
            f"Precision: {val_metrics['precision']:.4f} | "
            f"Recall: {val_metrics['recall']:.4f} | "
            f"F1: {val_metrics['f1']:.4f} | "
            f"Kappa: {val_metrics['kappa']:.4f}"
        )

        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            torch.save(model.state_dict(), save_path)
            print(f"Saved best warmup model to: {save_path}")

    print(f"Finished | best_acc={best_acc:.4f} | saved={save_path}")


def main(args):
    for j in [0]:
        args.seed = j
        for i in [470, 1412, 2354]:
            args.num_labeled = i
            run_one_setting(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Warmup pretraining for FixMatch')

    parser.add_argument('--num-workers', type=int, default=0, help='number of workers')
    parser.add_argument('--num-labeled', type=int, default=470, help='number of labeled data')
    parser.add_argument('--expand-labels', action='store_true', help='expand labels')
    parser.add_argument('--eval_step', default=50, type=int, help='kept for dataset split compatibility')
    parser.add_argument('--num_classes', default=2, type=int, help='number of classes')
    parser.add_argument('--batch-size', default=8, type=int, help='train batch size')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='warmup epochs')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--no-progress', action='store_true', help="don't use progress bar")
    parser.add_argument('--ave_class', default=False, type=bool)
    parser.add_argument('--data_path', default='pneumoniamnist_128.npz', type=str)

    args = parser.parse_args()
    main(args)