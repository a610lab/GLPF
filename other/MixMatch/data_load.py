import math

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from All_Project_plaqueClassification.SSL_Plaque.dataset.myload_data import load_per_data, read_datalist


plaque_mean = (0.246890, 0.257212, 0.279224)
plaque_std = (0.228331, 0.236681, 0.252307)


class RandomFlip(object):
    """Flip randomly the image.
    """
    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1]

        return x.copy()


def get_plaque(args):
    images_paths, labels = read_datalist(args.train_path)
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize([144, 96]),
        transforms.ToTensor(),
        transforms.Normalize(mean=plaque_mean, std=plaque_std)
    ])

    transform_val = transforms.Compose([
        transforms.Resize([144, 96]),
        transforms.ToTensor(),
        transforms.Normalize(mean=plaque_mean, std=plaque_std)
    ])
    if args.ave_class is True:
        train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, labels)
    else:
        train_labeled_idxs, train_unlabeled_idxs = x_u_split_Random(args, labels)
    # train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, labels)

    train_labeled_dataset = PLAQUESSL(
        images_paths, labels, train_labeled_idxs,
        train=True, transform=transform_labeled
    )

    train_unlabeled_dataset = PLAQUESSL(
        images_paths, labels, train_unlabeled_idxs,
        train=True, transform=TransformMatch(mean=plaque_mean, std=plaque_std)
    )
    val_images_paths, val_labels = read_datalist(args.val_path)
    val_dataset = PLAQUESSL(
        val_images_paths, val_labels, indexs=None,
        train=False, transform=transform_val
    )

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset


def x_u_split_Random(args, labels):
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = np.array(range(len(labels)))
    idx = np.random.choice(unlabeled_idx, args.num_labeled, False)
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
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


class PLAQUESSL(Dataset):
    def __init__(self, images_paths, labels, indexs,
                 train=True, transform=None):
        self. images_paths = images_paths
        self.labels = labels
        self.train = train
        self.transform = transform

        if indexs is not None:
            self.images_paths = np.array(self.images_paths)[indexs]
            self.labels = np.array(self.labels)[indexs]

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, index):
        img = load_per_data(self.images_paths[index])
        target = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)

        return index, img, target


class TransformMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize([144,96])])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak1 = self.weak(x)
        weak2 = self.weak(x)
        return self.normalize(weak1), self.normalize(weak2)




