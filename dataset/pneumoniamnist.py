import math
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


breast_mean = (0.5, 0.5, 0.5)
breast_std = (0.5, 0.5, 0.5)
size = [128, 128]


def get_breastmnist(args):
    data = np.load(args.data_path)

    train_images = data["train_images"]
    train_labels = data["train_labels"].reshape(-1)

    val_images = data["val_images"]
    val_labels = data["val_labels"].reshape(-1)

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=breast_mean, std=breast_std)
    ])

    transform_val = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=breast_mean, std=breast_std)
    ])

    if args.ave_class:
        train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, train_labels)
    else:
        train_labeled_idxs, train_unlabeled_idxs = x_u_split_random(args, train_labels)

    train_labeled_dataset = BREASTMNISTSSL(
        train_images, train_labels, train_labeled_idxs,
        train=True, transform=transform_labeled
    )

    train_unlabeled_dataset = BREASTMNISTSSL(
        train_images, train_labels, train_unlabeled_idxs,
        train=True, transform=TransformMatch(mean=breast_mean, std=breast_std)
    )

    val_dataset = BREASTMNISTSSL(
        val_images, val_labels, indexs=None,
        train=False, transform=transform_val
    )

    print("========== BreastMNIST Split ==========")
    print(f"train total      : {len(train_images)}")
    print(f"labeled samples  : {len(train_labeled_dataset)}")
    print(f"unlabeled samples: {len(train_unlabeled_dataset)}")
    print(f"val samples      : {len(val_dataset)}")
    print("=======================================")

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset


def x_u_split_random(args, labels):
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = np.array(range(len(labels)))

    idx = np.random.choice(unlabeled_idx, args.num_labeled, replace=False)
    labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)

    return labeled_idx, unlabeled_idx


def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = np.array(range(len(labels)))

    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, replace=False)
        labeled_idx.extend(idx)

    labeled_idx = np.array(labeled_idx)

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])

    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


class BREASTMNISTSSL(Dataset):
    def __init__(self, images, labels, indexs, train=True, transform=None):
        self.images = images
        self.labels = labels
        self.train = train
        self.transform = transform

        if indexs is not None:
            self.images = np.array(self.images)[indexs]
            self.labels = np.array(self.labels)[indexs]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        target = int(self.labels[index])

        img = self._to_pil(img)

        if self.transform is not None:
            img = self.transform(img)

        return index, img, target

    @staticmethod
    def _to_pil(img):
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)

        img = img.astype(np.uint8)
        return Image.fromarray(img)


class TransformMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(size),
        ])

        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(size),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05)
            ),
        ])

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)