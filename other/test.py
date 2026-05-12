import os
import sys
import csv
import glob
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model.resnest.torch import resnest50 as create_model


breast_mean = (0.5, 0.5, 0.5)
breast_std = (0.5, 0.5, 0.5)
size = [128, 128]


class BreastMNISTTestDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels.reshape(-1)
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=breast_mean, std=breast_std)
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        label = int(self.labels[index])

        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)

        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        img = self.transform(img)

        return index, img, label


def load_test_loader(data_path, batch_size, num_workers):
    data = np.load(data_path)
    test_images = data["test_images"]
    test_labels = data["test_labels"]

    test_dataset = BreastMNISTTestDataset(test_images, test_labels)
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False
    )
    return test_loader


def evaluate(model, test_loader, device):
    model.eval()

    all_targets = []
    all_preds = []
    total_loss = 0.0

    with torch.no_grad():
        for _, inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

            loss = F.cross_entropy(logits, targets)
            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item() * inputs.size(0)
            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    avg_loss = total_loss / len(test_loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average="binary", zero_division=0)
    recall = recall_score(all_targets, all_preds, average="binary", zero_division=0)
    f1 = f1_score(all_targets, all_preds, average="binary", zero_division=0)
    kappa = cohen_kappa_score(all_targets, all_preds)

    return {
        "loss": avg_loss,
        "acc": acc,
        "recall": recall,
        "precision": precision,
        "kappa": kappa,
        "f1": f1,
    }


def parse_ckpt_name(path):
    """
    例子:
    FixMatch_0_55.pth
    FixMatch_50_164.pth
    """
    name = os.path.basename(path).replace(".pth", "")
    parts = name.split("_")

    result = {
        "file": os.path.basename(path),
        "method": parts[0] if len(parts) > 0 else "",
        "seed": "",
        "num_labeled": ""
    }

    if len(parts) >= 3:
        result["seed"] = parts[1]
        result["num_labeled"] = parts[2]

    return result


def load_model_from_ckpt(ckpt_path, num_classes, device):
    model = create_model(num_classes=num_classes).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "ema_state_dict" in ckpt and ckpt["ema_state_dict"] is not None:
        print(f"[{os.path.basename(ckpt_path)}] Loading ema_state_dict")
        model.load_state_dict(ckpt["ema_state_dict"])
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        print(f"[{os.path.basename(ckpt_path)}] Loading state_dict")
        model.load_state_dict(ckpt["state_dict"])
    else:
        print(f"[{os.path.basename(ckpt_path)}] Loading raw state_dict")
        model.load_state_dict(ckpt)

    return model


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    test_loader = load_test_loader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    ckpt_paths = sorted(glob.glob(os.path.join(args.ckpt_dir, "*.pth")))
    if len(ckpt_paths) == 0:
        raise FileNotFoundError(f"No .pth files found in: {args.ckpt_dir}")

    results = []

    for ckpt_path in ckpt_paths:
        print("\n========================================")
        print("Testing checkpoint:", ckpt_path)

        meta = parse_ckpt_name(ckpt_path)
        model = load_model_from_ckpt(ckpt_path, args.num_classes, device)
        metrics = evaluate(model, test_loader, device)

        row = {
            "file": meta["file"],
            "method": meta["method"],
            "seed": meta["seed"],
            "num_labeled": meta["num_labeled"],
            "acc": metrics["acc"],
            "recall": metrics["recall"],
            "precision": metrics["precision"],
            "kappa": metrics["kappa"],
            "f1": metrics["f1"],
            "loss": metrics["loss"],
        }
        results.append(row)

        print(f"Acc      : {row['acc']:.4f}")
        print(f"Recall   : {row['recall']:.4f}")
        print(f"Precision: {row['precision']:.4f}")
        print(f"Kappa    : {row['kappa']:.4f}")
        print(f"F1       : {row['f1']:.4f}")
        print(f"Loss     : {row['loss']:.4f}")

    os.makedirs(os.path.dirname(args.csv_path) if os.path.dirname(args.csv_path) else ".", exist_ok=True)

    with open(args.csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "method", "seed", "num_labeled", "acc", "recall", "precision", "kappa", "f1", "loss"]
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n===== All Results Saved =====")
    print("CSV path:", args.csv_path)

    print("\n===== Summary =====")
    for row in results:
        print(
            f"{row['file']} | "
            f"Acc={row['acc']:.4f}, "
            f"Recall={row['recall']:.4f}, "
            f"Precision={row['precision']:.4f}, "
            f"Kappa={row['kappa']:.4f}, "
            f"F1={row['f1']:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch test FixMatch checkpoints on BreastMNIST test set")
    parser.add_argument("--data_path", type=str, default="pneumoniamnist_128.npz")
    parser.add_argument("--ckpt_dir", type=str, default="glpf_pth")
    parser.add_argument("--csv_path", type=str, default="glpf_pth/glpf_test_results.csv")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    main(args)