import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

plt.rcParams['font.sans-serif'] = ['SimSun']
# 解决坐标轴负号显示问题
plt.rcParams['axes.unicode_minus'] = False

class evaluate():
    def __init__(self, y_true, y_pred , classes = ['high', 'low', 'mix']):
        self.matrix = confusion_matrix(y_true, y_pred, normalize=None)
        self.classes = classes
        self.y_true = y_true
        self.y_pred = y_pred

    def Accuracy(self):
        t = np.trace(self.matrix)
        return t/(self.matrix.sum())

    def Precision(self):#某一类别预测x个  有y个预测对了
        t = np.sum(self.matrix, axis=0)
        x = np.diag(self.matrix)
        return x / t

    def Macro_Average(self):
        macro_P = self.Precision().mean()
        macro_R = self.Recall().mean()
        macro_F = self.F1_score().mean()
        return macro_P, macro_R, macro_F

    def Micro_Average(self):
        micro_P = self.Precision().sum() / (self.Precision().sum() + self.Recall().sum())
        micro_R =self.Recall().sum() / (self.Precision().sum() + self.Recall().sum())
        micro_F = 2 * micro_P * micro_R /(micro_P + micro_R)
        return micro_P, micro_R, micro_F

    def Recall(self):#某一类别拥有x个  有y个预测对了
        t = np.sum(self.matrix, axis=1)
        x = np.diag(self.matrix)
        return x / t

    def F1_score(self):
        pre = self.Precision()
        rec = self.Recall()
        return (2 * pre * rec)/(pre + rec)

    def Show_ConM(self):

        classes = ['high', 'low', 'mix']
        C = confusion_matrix(self.y_true, self.y_pred, normalize='true')

        # 绘制混淆矩阵图像
        plt.imshow(C, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('SSTVC',fontdict={'size': 14})   # TODO
        plt.colorbar().ax.tick_params(labelsize=16)
        tick_marks = np.arange(3)
        plt.xticks(tick_marks, classes, fontdict={'size': 14})
        plt.yticks(tick_marks, classes, fontdict={'size': 14})
        plt.xlabel('预测',fontdict={'size': 14})
        plt.ylabel('实际',fontdict={'size': 14})

        # 添加文本标记
        thresh = C.max() / 2.
        for i, j in np.ndindex(C.shape):
            plt.text(j, i, format(C[i, j], '.3f'), horizontalalignment="center",
                     verticalalignment='center' ,color="white" if C[i, j] > thresh else "black",
                     fontdict={'size':16}

                     )
        plt.tight_layout()
        plt.show()
        '''
        plt.matshow(C, cmap=plt.cm.Reds)
        xlocations = np.array(range(len(classes)))
        plt.xticks(xlocations, classes)
        plt.yticks(xlocations, classes)
        for i in range(len(C)):
            for j in range(len(C)):
                plt.annotate(C[j, i]/np.sum(C, axis=0)[j], xy=(i, j), horizontalalignment='center', verticalalignment='center')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('confusion matrix')
        plt.show()
        '''

    def Kappa(self):
        line = np.sum(self.matrix, axis=1)
        arrange = np.sum(self.matrix, axis=0)
        x = line * arrange
        p0 = self.Accuracy()
        pe = x.sum()/self.matrix.sum()**2
        k = (p0 - pe) / (1 - pe)
        return k

    def show_All(self):
        # print('ACC is :{:.4f}'.format(self.Accuracy()))
        # print('Precision is '+str(self.Precision()))
        # print('Recall is '+str(self.Recall()))
        # print('F1_score is '+str(self.F1_score()))
        # print('Macro_Average is '+str(self.Macro_Average()))
        # print('Micro_Average is '+str(self.Micro_Average()))
        # print('*'*50)
        print('ACC is :{:.4f}'.format(self.Accuracy()))
        print('Kappa is :{:.4f}'.format(self.Kappa()))
        print('Precision is ' + str(self.Precision().mean()))
        print('F1_Mean is :{:.4f}'.format(self.F1_score().mean()))
        print('Recall is :{:.4f}'.format(self.Recall().mean()))

        # x = 2*self.Precision().mean() * self.Recall().mean()/(self.Precision().mean() + self.Recall().mean())
        # print('MEAN F1_score is ' + str(x))
        self.Show_ConM()
    def returnData(self):
        return self.Accuracy(),self.Kappa(),self.Precision().mean(),self.F1_score().mean(),self.Recall().mean()
if __name__ == '__main__':
    y_true = [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2]
    y_pred = [0,0,0,0,1,2,0,0,0,0,0,0,1,1,2,2,0,0,0,2,2,2,2,2,2]
    eva = evaluate(y_true, y_pred)
    # print(eva.Accuracy())
    # print(eva.Precision())
    # print(eva.Recall())
    # print(eva.F1_score())
    # print(eva.Macro_Average())
    # print(eva.Micro_Average())
    eva.show_All()
    # eva.Kappa()