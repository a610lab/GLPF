import torch.nn.functional as F
import torch


class GlobalFeature(object):
    def __init__(self, args):
        self.reset()
        self.args = args

    def reset(self):
        self.features_dict = {}
        self.label_dict = {}

    def update(self, index, feats, targets):
        self.device = feats.device
        for i in range(self.args.batch_size):
            self.features_dict[index[i]] = feats[i]
            self.label_dict[index[i]] = targets[i]
        self.feature_tensors = [item.cpu().detach() for item in self.features_dict.values()]
        self.feats_all = torch.stack(self.feature_tensors).to(self.device)
        self.labels_all = torch.tensor(list(self.label_dict.values())).to(self.device)

    def centerFeat(self):
        all_feats = torch.zeros((self.args.num_classes, self.args.feature_dim)).to(self.device)
        all_nums = torch.zeros((self.args.num_classes,)).to(self.device)
        for i in range(len(self.labels_all)):
            all_feats[self.labels_all[i]] = all_feats[self.labels_all[i]] + self.feats_all[i]
            all_nums[self.labels_all[i]] = all_nums[self.labels_all[i]] + 1
        per_feats = torch.nan_to_num(all_feats / all_nums.reshape(-1, 1))
        return per_feats


def LocalFeature(feats_all, labels_all, feats_u_w, targets_u_w, args):
    cos_sim = F.cosine_similarity(feats_u_w.unsqueeze(1), feats_all.unsqueeze(0), dim=2)
    cos_sim, indices = torch.topk(cos_sim, k=args.k, dim=1)
    knn_labels = labels_all[indices]
    pred_labels, _ = torch.mode(knn_labels, dim=1)
    knn_labels = knn_labels.type(torch.LongTensor)
    same_P = pred_labels.eq(targets_u_w).float()
    cos_sim_count = torch.zeros([args.mu*args.batch_size, args.num_classes], dtype=torch.float32)
    count = torch.zeros([args.mu*args.batch_size, args.num_classes], dtype=torch.float32)
    for step, i in enumerate(knn_labels):
        for step2, j in enumerate(i):
            cos_sim_count[step][j] = cos_sim[step][step2].item() + cos_sim_count[step][j]
            count[step][j] = count[step][j] + 1
    knn_prb = F.softmax(cos_sim_count, dim=-1)
    max_probs_knn, targets_u_knn = torch.max(knn_prb, dim=-1)
    same_K = max_probs_knn.ge(args.threshold).float().to(args.device)
    return same_P * same_K


class PseudoBalance(object):
    def __init__(self, args):
        self.args = args
        self.alpha = torch.ones(args.num_classes).to(args.device)
        self.num_classes = torch.zeros(args.num_classes).to(args.device)
    def update(self, mask_a, mask_b, targets_u_w_a, targets_u_w):
        for i in mask_a:
            if i:
                self.num_classes[targets_u_w_a] = self.num_classes[targets_u_w_a] + 1
        for i in mask_b:
            if i:
                self.num_classes[targets_u_w] = self.num_classes[targets_u_w] + 1
        temp = 1 - self.num_classes / (torch.sum(self.num_classes)+1e-6)
        self.alpha = self.alpha * self.args.ema_decay + (1-self.args.ema_decay) * temp