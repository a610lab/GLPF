import random
from collections import Counter
from model.HS_ResNet.hs_resnet import hs_resnet50

import math
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    precision_score,
    recall_score,
)
import timm


# =========================
# 基本配置
# =========================
DATA_PATH = "pneumoniamnist_128.npz"   # 如果数据文件不在当前目录，改成完整路径
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOCAL_CONVNEXTV2_CKPT = "pytorch_model.bin"
LOCAL_RESNEST50_CKPT = "pytorch_model_resnext50.bin"
LOCAL_VIT_CKPT = "pytorch_model_vit.bin"

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
NUM_WORKERS = 0   # Windows 建议先用 0
SEED = 42
NUM_CLASSES = 2

BACKBONES = [
    "resnet50",
    "convnextv2_tiny",
    "resnest50d",
    "vit_b_16",
    "hs_resnet50",
]


# =========================
# 固定随机种子
# =========================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# 数据集
# =========================
class NpzDataset(Dataset):
    def __init__(self, images, labels, train=False):
        self.images = images
        self.labels = labels.reshape(-1).astype(np.int64)

        if train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]

        # 如果是单通道灰度图，扩成 3 通道
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)

        # 如果是 (H, W, 1)，也扩成 3 通道
        if img.ndim == 3 and img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)

        img = img.astype(np.uint8)
        label = self.labels[idx]
        img = self.transform(img)

        return img, label


# =========================
# 读取数据
# =========================
def load_data(data_path):
    data = np.load(data_path)

    train_set = NpzDataset(data["train_images"], data["train_labels"], train=True)
    val_set = NpzDataset(data["val_images"], data["val_labels"], train=False)
    test_set = NpzDataset(data["test_images"], data["test_labels"], train=False)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return train_loader, val_loader, test_loader

def resize_vit_pos_embed(state_dict, model):
    if "pos_embed" not in state_dict:
        return state_dict

    pos_embed_checkpoint = state_dict["pos_embed"]         # [1, old_num_tokens, C]
    pos_embed_model = model.pos_embed                      # [1, new_num_tokens, C]

    if pos_embed_checkpoint.shape == pos_embed_model.shape:
        return state_dict

    embed_dim = pos_embed_checkpoint.shape[-1]

    # cls token
    cls_pos_embed = pos_embed_checkpoint[:, :1, :]         # [1, 1, C]
    patch_pos_embed = pos_embed_checkpoint[:, 1:, :]       # [1, N, C]

    old_grid_size = int(math.sqrt(patch_pos_embed.shape[1]))
    new_grid_size = int(math.sqrt(pos_embed_model.shape[1] - 1))

    patch_pos_embed = patch_pos_embed.reshape(
        1, old_grid_size, old_grid_size, embed_dim
    ).permute(0, 3, 1, 2)                                  # [1, C, H, W]

    patch_pos_embed = F.interpolate(
        patch_pos_embed,
        size=(new_grid_size, new_grid_size),
        mode="bicubic",
        align_corners=False
    )

    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(
        1, new_grid_size * new_grid_size, embed_dim
    )

    new_pos_embed = torch.cat((cls_pos_embed, patch_pos_embed), dim=1)
    state_dict["pos_embed"] = new_pos_embed
    return state_dict

# =========================
# 构建模型
# =========================
def build_model(name, num_classes=2):
    name = name.lower()

    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "convnextv2_tiny":
        model = timm.create_model(
            "convnextv2_tiny.fcmae_ft_in22k_in1k",
            pretrained=True,
            pretrained_cfg_overlay={"file": LOCAL_CONVNEXTV2_CKPT},
            num_classes=num_classes,
        )
        return model

    if name == "resnest50d":
        model = timm.create_model(
            "resnest50d",
            pretrained=False,
            checkpoint_path=LOCAL_RESNEST50_CKPT
        )
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "vit_b_16":
        model = timm.create_model(
            "vit_base_patch16_224.augreg2_in21k_ft_in1k",
            pretrained=False,
            img_size=128,
        )

        checkpoint = torch.load(LOCAL_VIT_CKPT, map_location="cpu")

        # 有些权重文件外面包了一层 state_dict / model
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model" in checkpoint:
            checkpoint = checkpoint["model"]

        checkpoint = resize_vit_pos_embed(checkpoint, model)

        # 去掉分类头，后面自己换成2类
        checkpoint.pop("head.weight", None)
        checkpoint.pop("head.bias", None)

        model.load_state_dict(checkpoint, strict=False)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model

    if name == "hs_resnet50":
        model = hs_resnet50(
            small_input=True,
            num_classes=num_classes
        )
        return model

    raise ValueError(f"Unsupported backbone: {name}")

# =========================
# 验证 / 测试
# =========================
def evaluate(model, loader, criterion):
    model.eval()

    all_labels = []
    all_preds = []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item() * images.size(0)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")
    kappa = cohen_kappa_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="binary", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="binary", zero_division=0)

    return {
        "loss": avg_loss,
        "acc": acc,
        "f1": f1,
        "kappa": kappa,
        "precision": precision,
        "recall": recall,
    }


# =========================
# 单个模型训练
# =========================
def train_one_model(backbone_name):
    print(f"\n===== Training {backbone_name} =====")

    train_loader, val_loader, test_loader = load_data(DATA_PATH)
    model = build_model(backbone_name, NUM_CLASSES).to(DEVICE)

    data = np.load(DATA_PATH)
    train_labels = data["train_labels"].reshape(-1)
    counts = Counter(train_labels.tolist())
    total = sum(counts.values())

    class_weights = []
    for i in range(NUM_CLASSES):
        class_weights.append(total / counts[i])

    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = -1.0
    best_model_path = f"best_{backbone_name}.pth"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, criterion)

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Acc: {val_metrics['acc']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val Kappa: {val_metrics['kappa']:.4f}"
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save(model.state_dict(), best_model_path)

    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    test_metrics = evaluate(model, test_loader, criterion)

    print(f"Best checkpoint saved at: {best_model_path}")
    print("Test metrics:")
    print(test_metrics)

    return test_metrics


# =========================
# 主函数
# =========================
def main():
    seed_everything(SEED)

    print(f"Using device: {DEVICE}")
    data = np.load(DATA_PATH)

    for split in ["train", "val", "test"]:
        labels = data[f"{split}_labels"].reshape(-1)
        cnt = Counter(labels.tolist())
        print(f"{split}: {len(labels)} samples | class distribution = {dict(cnt)}")

    results = []

    for backbone in BACKBONES:
        metrics = train_one_model(backbone)
        row = {"backbone": backbone}
        row.update(metrics)
        results.append(row)

    print("\n===== Final Comparison =====")
    for row in results:
        print(row)


if __name__ == "__main__":
    main()