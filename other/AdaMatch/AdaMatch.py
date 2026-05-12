import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import sklearn.metrics
import seaborn as sns


class Adamatch():
    """
    Paper: AdaMatch: A Unified Approach to Semi-Supervised Learning and Domain Adaptation
    Authors: David Berthelot, Rebecca Roelofs, Kihyuk Sohn, Nicholas Carlini, Alex Kurakin
    """

    def __init__(self, encoder, classifier):
        """
        NOTE: the actual AdaMatch paper doesn't separate between encoder and classifier,
        but I find it more practical for the purposes of setting up the networks.

        Arguments:
        ----------
        encoder: PyTorch neural network
            Neural network that receives images and encodes them into an array of size X.

        classifier: PyTorch neural network
            Neural network that receives an array of size X and classifies it into N classes.
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder.to(self.device)
        self.classifier = classifier.to(self.device)

    def train(self, source_dataloader_weak, source_dataloader_strong,
              target_dataloader_weak, target_dataloader_strong, target_dataloader_test,
              epochs, hyperparams, save_path):
        """
        Trains the model (encoder + classifier).

        Arguments:
        ----------
        source_dataloader_weak: PyTorch DataLoader
            DataLoader with source domain training data with weak augmentations.

        source_dataloader_strong: PyTorch DataLoader
            DataLoader with source domain training data with strong augmentations.

        target_dataloader_weak: PyTorch DataLoader
            DataLoader with target domain training data with weak augmentations.
            THIS DATALOADER'S BATCH SIZE MUST BE 3 * SOURCE_DATALOADER_BATCH_SIZE.

        target_dataloader_strong: PyTorch DataLoader
            DataLoader with target domain training data with strong augmentations.
            THIS DATALOADER'S BATCH SIZE MUST BE 3 * SOURCE_DATALOADER_BATCH_SIZE.

        target_dataloader_test: PyTorch DataLoader
            DataLoader with target domain validation data, used for early stopping.

        epochs: int
            Amount of epochs to train the model for.

        hyperparams: dict
            Dictionary containing hyperparameters for this algorithm. Check `data/hyperparams.py`.

        save_path: str
            Path to store model weights.

        Returns:
        --------
        encoder: PyTorch neural network
            Neural network that receives images and encodes them into an array of size X.

        classifier: PyTorch neural network
            Neural network that receives an array of size X and classifies it into N classes.
        """
        # TODO 各种超参
        # configure hyperparameters
        lr = hyperparams['learning_rate']
        wd = hyperparams['weight_decay']
        step_scheduler = hyperparams['step_scheduler']
        tau = hyperparams['tau']
        # TODO 确定每个epoch运行数量
        iters = max(len(source_dataloader_weak), len(source_dataloader_strong), len(target_dataloader_weak),
                    len(target_dataloader_strong))

        # mu related stuff
        steps_per_epoch = iters
        total_steps = epochs * steps_per_epoch  # TODO 总运行数
        current_step = 0

        # TODO 优化器
        # configure optimizer and scheduler
        optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.classifier.parameters()), lr=lr,
                               weight_decay=wd)
        if step_scheduler:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20)

        # early stopping variables
        start_epoch = 0
        best_acc = 0.0
        patience = 20
        bad_epochs = 0

        self.history = {'epoch_loss': [], 'accuracy_source': [], 'accuracy_target': []}

        # training loop
        for epoch in range(start_epoch, epochs):
            running_loss = 0.0

            # set network to training mode
            self.encoder.train()
            self.classifier.train()

            dataset = zip(source_dataloader_weak, source_dataloader_strong, target_dataloader_weak,
                          target_dataloader_strong)

            # this is where the unsupervised learning comes in, as such, we're not interested in labels
            # TODO 相当于 dataloader循环
            for (data_source_weak, labels_source), (data_source_strong, _), (data_target_weak, _), (
            data_target_strong, _) in dataset:
                # todo 加载入GPU
                data_source_weak = data_source_weak.to(self.device)
                labels_source = labels_source.to(self.device)

                data_source_strong = data_source_strong.to(self.device)
                data_target_weak = data_target_weak.to(self.device)
                data_target_strong = data_target_strong.to(self.device)

                # concatenate data (in case of low GPU power this could be done after classifying with the model)
                # todo 拼接有标签数据(source) weak & strong 无标签数据(target) weak & strong
                # todo 作者认为 source 与 target 分布不同
                data_combined = torch.cat([data_source_weak, data_source_strong, data_target_weak, data_target_strong],
                                          0)
                source_combined = torch.cat([data_source_weak, data_source_strong], 0)

                # get source data limit (useful for slicing later)
                source_total = source_combined.size(0)

                # zero gradients
                optimizer.zero_grad()

                # forward pass: calls the model once for both source and target and once for source only
                logits_combined = self.classifier(self.encoder(data_combined))  # todo 将所有数据带入模型
                logits_source_p = logits_combined[:source_total]  # todo 确定有标签模型结果

                # from https://github.com/yizhe-ang/AdaMatch-PyTorch/blob/main/trainers/adamatch.py
                self._disable_batchnorm_tracking(self.encoder)  # todo 关闭均值与方差的统计
                self._disable_batchnorm_tracking(self.classifier)  # todo 关闭均值与方差的统计
                logits_source_pp = self.classifier(self.encoder(source_combined))  # todo 重新计算有标签结果

                self._enable_batchnorm_tracking(self.encoder)  # todo 打开均值与方差的统计
                self._enable_batchnorm_tracking(self.classifier)  # todo 打开均值与方差的统计

                # perform random logit interpolation
                lambd = torch.rand_like(logits_source_p).to(
                    self.device)  # todo 生成一个与 logits_source_p 大小相似的随机tensor（0 -1 之间）
                final_logits_source = (lambd * logits_source_p) + (
                            (1 - lambd) * logits_source_pp)  # todo Random Logit Interpolation

                # distribution allignment
                ## softmax for logits of weakly augmented source images
                # todo 将有标签数据结果（weak） 变为 概率
                logits_source_weak = final_logits_source[:data_source_weak.size(0)]
                pseudolabels_source = F.softmax(logits_source_weak, 1)

                ## softmax for logits of weakly augmented target images
                # todo 将无标签数据结果（weak） 变为 概率
                logits_target = logits_combined[source_total:]
                logits_target_weak = logits_target[:data_target_weak.size(0)]
                pseudolabels_target = F.softmax(logits_target_weak, 1)

                ## allign target label distribtion to source label distribution
                expectation_ratio = (1e-6 + torch.mean(pseudolabels_source)) / (1e-6 + torch.mean(pseudolabels_target))
                final_pseudolabels = F.normalize((pseudolabels_target * expectation_ratio), p=2,
                                                 dim=1)  # L2 normalization # todo 按行进行标准化

                # perform relative confidence thresholding
                # todo 有标签数据 每个batch 最大 概率 均值
                row_wise_max, _ = torch.max(pseudolabels_source, dim=1)
                final_sum = torch.mean(row_wise_max, 0)

                ## define relative confidence threshold
                # todo  乘以 超参 default 0.9
                c_tau = tau * final_sum
                # todo 将 c_tau 作为阈值
                max_values, _ = torch.max(final_pseudolabels, dim=1)
                mask = (max_values >= c_tau).float()

                # compute loss
                # todo 有标签 1.weak 与 label 2.strong 与 label 二者最终 loss 平均
                source_loss = self._compute_source_loss(logits_source_weak,
                                                        final_logits_source[data_source_weak.size(0):], labels_source)

                # todo 与 FixMatch 类似
                final_pseudolabels = torch.max(final_pseudolabels, 1)[1]  # argmax   todo 伪标签
                target_loss = self._compute_target_loss(final_pseudolabels, logits_target[data_target_weak.size(0):],
                                                        mask)

                ## compute target loss weight (mu)  todo 动态变化无标签数据loss超参数
                pi = torch.tensor(np.pi, dtype=torch.float).to(self.device)
                step = torch.tensor(current_step, dtype=torch.float).to(self.device)
                mu = 0.5 - torch.cos(torch.minimum(pi, (2 * pi * step) / total_steps)) / 2

                ## get total loss
                loss = source_loss + (mu * target_loss)
                current_step += 1

                # backpropagate and update weights
                loss.backward()
                optimizer.step()

                # metrics
                running_loss += loss.item()

            # get losses
            epoch_loss = running_loss / iters
            self.history['epoch_loss'].append(epoch_loss)

            # self.evaluate on testing data (target domain)
            epoch_accuracy_source = self.evaluate(source_dataloader_weak)
            epoch_accuracy_target = self.evaluate(target_dataloader_weak)
            test_epoch_accuracy = self.evaluate(target_dataloader_test)

            self.history['accuracy_source'].append(epoch_accuracy_source)
            self.history['accuracy_target'].append(epoch_accuracy_target)

            # save checkpoint
            if test_epoch_accuracy > best_acc:
                torch.save({'encoder_weights': self.encoder.state_dict(),
                            'classifier_weights': self.classifier.state_dict()
                            }, save_path)
                best_acc = test_epoch_accuracy
                bad_epochs = 0

            else:
                bad_epochs += 1

            print(
                '[Epoch {}/{}] loss: {:.6f}; accuracy source: {:.6f}; accuracy target: {:.6f}; val accuracy: {:.6f};'.format(
                    epoch + 1, epochs, epoch_loss, epoch_accuracy_source, epoch_accuracy_target, test_epoch_accuracy))

            if bad_epochs >= patience:
                print(f"reached {bad_epochs} bad epochs, stopping training with best val accuracy of {best_acc}!")
                break

            # scheduler step
            if step_scheduler:
                scheduler.step()

        best = torch.load(save_path)
        self.encoder.load_state_dict(best['encoder_weights'])
        self.classifier.load_state_dict(best['classifier_weights'])

        return self.encoder, self.classifier

    def evaluate(self, dataloader, return_lists_roc=False):
        """
        Evaluates model on `dataloader`.

        Arguments:
        ----------
        dataloader: PyTorch DataLoader
            DataLoader with test data.

        return_lists_roc: bool
            If True returns also list of labels, a list of outputs and a list of predictions.
            Useful for some metrics.

        Returns:
        --------
        accuracy: float
            Accuracy achieved over `dataloader`.
        """

        # set network to evaluation mode
        self.encoder.eval()
        self.classifier.eval()

        labels_list = []
        outputs_list = []
        preds_list = []

        with torch.no_grad():
            for data, labels in dataloader:
                data = data.to(self.device)
                labels = labels.to(self.device)

                # predict
                outputs = F.softmax(self.classifier(self.encoder(data)), dim=1)

                # numpify
                labels_numpy = labels.detach().cpu().numpy()
                outputs_numpy = outputs.detach().cpu().numpy()  # probs (AUROC)

                preds = np.argmax(outputs_numpy, axis=1)  # accuracy

                # append
                labels_list.append(labels_numpy)
                outputs_list.append(outputs_numpy)
                preds_list.append(preds)

            labels_list = np.concatenate(labels_list)
            outputs_list = np.concatenate(outputs_list)
            preds_list = np.concatenate(preds_list)

        # metrics
        # auc = sklearn.metrics.roc_auc_score(labels_list, outputs_list, multi_class='ovr')
        accuracy = sklearn.metrics.accuracy_score(labels_list, preds_list)

        if return_lists_roc:
            return accuracy, labels_list, outputs_list, preds_list

        return accuracy

    def plot_metrics(self):
        """
        Plots the training metrics (only usable after calling .train()).
        """

        # plot metrics for losses n stuff
        fig, axs = plt.subplots(1, 3, figsize=(18, 5), dpi=200)

        epochs = len(self.history['epoch_loss'])

        axs[0].plot(range(1, epochs + 1), self.history['epoch_loss'])
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].set_title('Entropy loss')

        axs[1].plot(range(1, epochs + 1), self.history['accuracy_source'])
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Accuracy')
        axs[1].set_title('Accuracy on weakly augmented source')

        axs[2].plot(range(1, epochs + 1), self.history['accuracy_target'])
        axs[2].set_xlabel('Epochs')
        axs[2].set_ylabel('Accuracy')
        axs[2].set_title('Accuracy on weakly augmented target')

        plt.show()

    def plot_cm_roc(self, dataloader, n_classes=10):
        """
        Plots the confusion matrix and ROC curves of the model on `dataloader`.

        Arguments:
        ----------
        dataloader: PyTorch DataLoader
            DataLoader with test data.

        n_classes: int
            Number of classes.
        """

        cmap = sns.diverging_palette(220, 20, as_cmap=True)

        self.encoder.eval()
        self.classifier.eval()

        accuracy, labels_list, outputs_list, preds_list = self.evaluate(dataloader, return_lists_roc=True)

        # plot confusion matrix
        cm = sklearn.metrics.confusion_matrix(labels_list, preds_list)
        group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
        group_percentages = ['({0:.2%})'.format(value) for value in cm.flatten() / np.sum(cm)]
        labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(n_classes, n_classes)
        # tn, fp, fn, tp = cm.ravel()

        plt.figure(figsize=(10, 10), dpi=200)
        sns.heatmap(cm, annot=labels, cmap=cmap, fmt="")
        plt.title("Confusion matrix")
        plt.ylabel("Actual label")
        plt.xlabel("Predicted label")
        plt.show()

        # plot roc
        ## one hot encode data
        onehot = np.zeros((labels_list.size, labels_list.max() + 1))
        onehot[np.arange(labels_list.size), labels_list] = 1
        onehot = onehot.astype('int')

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        ## get roc curve and auroc for each class
        for i in range(n_classes):
            fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(onehot[:, i], outputs_list[:, i])
            roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

        ## get macro average auroc
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = sklearn.metrics.auc(fpr["macro"], tpr["macro"])

        plt.figure(figsize=(9, 9), dpi=200)

        plt.plot([0, 1], [0, 1], color='black', linestyle='--')

        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label=f"AUC class {i} = {roc_auc[i]:.4f}")

        plt.plot(fpr["macro"], tpr["macro"], label=f"macro-average AUC = {roc_auc['macro']:.4f}", color='deeppink',
                 linewidth=2)

        plt.title('Receiver Operating Characteristic (ROC)')
        plt.xlabel('False Positives')
        plt.ylabel('True Positives')
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.legend(loc='lower right')
        plt.show()

    @staticmethod
    def _disable_batchnorm_tracking(model):
        def fn(module):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.track_running_stats = False

        model.apply(fn)

    @staticmethod
    def _enable_batchnorm_tracking(model):
        def fn(module):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.track_running_stats = True

        model.apply(fn)

    @staticmethod
    def _compute_source_loss(logits_weak, logits_strong, labels):
        """
        Receives logits as input (dense layer outputs with no activation function)
        """
        loss_function = nn.CrossEntropyLoss()  # default: `reduction="mean"`
        weak_loss = loss_function(logits_weak, labels)
        strong_loss = loss_function(logits_strong, labels)

        # return weak_loss + strong_loss
        return (weak_loss + strong_loss) / 2

    @staticmethod
    def _compute_target_loss(pseudolabels, logits_strong, mask):
        """
        Receives logits as input (dense layer outputs with no activation function).
        `pseudolabels` are treated as ground truth (standard SSL practice).
        """
        loss_function = nn.CrossEntropyLoss(reduction="none")
        pseudolabels = pseudolabels.detach()  # remove from backpropagation

        loss = loss_function(logits_strong, pseudolabels)

        return (loss * mask).mean()