import torch
import numpy as np

class IoU:
    """ Class to find the mean IoU using confusion matrix approach """    
    def __init__(self, num_classes):
        self.iou_metric = 0.0
        self.num_classes = num_classes
        # placeholder for confusion matrix on entire dataset | IoU 需要混淆矩阵？？
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        
    def _fast_hist(self, label_true, label_pred):
        """ Function to calculate confusion matrix on single batch """
        # mask only valid labels (this step should be irrelevant usually)
        mask = (label_true >= 0) & (label_true < self.num_classes)
        # calculate correctness of segementation by assigning numbers and count them
        # e.g. for 6 classes [0:5], 
            # 7 is a class 2 pixel segemented correctly (6*1+1)
            # 16 is a class 3 pixel segmented as class 5 (6*2+4)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def update(self, y_preds, labels):
        """ 
            Function finds the IoU for the input batch
            and add batch metrics to overall metrics 
        """
        predicted_labels = torch.argmax(y_preds, dim=1)
        batch_confusion_matrix = self._fast_hist(labels.numpy().flatten(), \
                                                 predicted_labels.numpy().flatten())
        self.confusion_matrix += batch_confusion_matrix
    
    def compute(self, matrix = None):
        """ Computes overall meanIoU metric from confusion matrix data """ 
        hist = self.confusion_matrix
        # if a matrix is given as argument to the function, compute the metrices based on that matrix 
        if matrix:
            hist = matrix
        # divide number of pixels segmented correctly (area of overlap) 
        # by number of pixels that were segmented in this class and that should have been segmented in this class (hist.sum(axis=1) + hist.sum(axis=0))
        # minus 1 time the pixels segmented correctly in the denominator as they are in both sums
        # IoU = TP / (TP + FP + FN)
        # TP = np.diag(hist); FP = hist.sum(axis=0) - np.diag(hist); FN = hist.sum(axis=1) - np.diag(hist) ?
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)) 
        # calculate mean of IoU per class
        mean_iu = np.nanmean(iu)
        # calculate accuracy
        accuracy = np.diag(hist).sum() / hist.sum().sum()
        # class_accuracy = (np.diag(hist) + (hist.sum().sum() - hist.sum(axis=1) - hist.sum(axis=0) + np.diag(hist))) / (hist.sum().sum())
        # calculate dice coefficient / f1 score
        f1 = 2*np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0))
        meanf1 = np.nanmean(f1)
        # return {'hist' : hist, 'accuracy' : accuracy, 'classwise_accuracy' : class_accuracy, 'miou' : mean_iu, 'classwise_iou' : iu}
        return {'accuracy' : accuracy, \
                'miou' : mean_iu, \
                'classwise_iou' : iu, \
                'classwise_f1': f1, \
                'f1_mean': meanf1, \
                'matrix': hist}

    def reset(self):
        self.iou_metric = 0.0
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.eps = 1e-8
        # self.eps = 0

        # print("confusion_matrix", self.confusion_matrix)

    def get_tp_fp_tn_fn(self):
        '''
        这是一种通用的方法，可以计算任意维度的混淆矩阵的 TP、FP、FN、TN。
        '''
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)
        fn = self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix)
        tn = np.diag(self.confusion_matrix).sum() - np.diag(self.confusion_matrix)
        # print("tp, fp, tn, fn", tp, fp, fn, tn)
        return tp, fp, tn, fn

    def Pixel_Accuracy_Class(self):
        #         TP          TP+FP
        Acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=0) + self.eps)
        return Acc
    
    def class_precision(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        precision_per_class  = tp / (tp + fp + self.eps)
        macro_precision = np.mean(precision_per_class)
        micro_precision = tp.sum() / (tp.sum() + fp.sum() + self.eps)
        # print("precision_per_class", precision_per_class, macro_precision, micro_precision)
        return precision_per_class, macro_precision, micro_precision


    def Precision(self):
        '''方法一得到的是每个类的 Precision 值， 微观'''
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        # print("tp, fp, tn, fn", tp, fp, tn, fn)
        precision = tp / (tp + fp + self.eps)
        # precision = tp.sum() / (tp.sum() + fp.sum() + self.eps)
        return precision

    def Recall(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        recall = tp / (tp + fn + self.eps)
        # recall = tp.sum() / (tp.sum() + fn)
        return recall

    def F1(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Precision = tp / (tp + fp + self.eps)
        Recall = tp / (tp + fn + self.eps)
        F1 = (2.0 * Precision * Recall) / (Precision + Recall + self.eps)

        # precision = tp.sum() / (tp.sum() + fp.sum() + self.eps)
        # recall = tp.sum() / (tp.sum() + fn)
        # # F1 = (2.0 * precision * recall) / (precision + recall + self.eps)
        return F1


    def OA(self):
        OA = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + self.eps)
        return OA

    def Intersection_over_Union(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        IoU = tp / (tp + fn + fp + self.eps)
        return IoU

    def Dice(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Dice = 2 * tp / ((tp + fp) + (tp + fn) + self.eps)
        return Dice

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / (np.sum(self.confusion_matrix) + self.eps)
        iou = self.Intersection_over_Union()
        FWIoU = (freq[freq > 0] * iou[freq > 0]).sum()
        return FWIoU

    def Kappa(self):
        # kappa metric 
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        po = (tp + tn) / (tp + tn + fp + fn + self.eps)
        pe = ((tp + fp) * (tp + fn) + (fp + tn) * (fn + tn)) / (tp + tn + fp + fn + self.eps) ** 2
        kappa = (po - pe) / (1 - pe + self.eps)
        return kappa
    
    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape, 'pre_image shape {}, gt_image shape {}'.format(pre_image.shape,
                                                                                                 gt_image.shape)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
        # print("confusion_matrix", self.confusion_matrix)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


if __name__ == '__main__':

    # gt = np.array([[0, 2, 1],
    #                [1, 2, 1],
    #                [1, 0, 1]])

    # pre = np.array([[0, 1, 1],
    #                [2, 0, 1],
    #                [1, 1, 1]])
    

    gt = np.array([0, 0, 1, 1, 1, 1, 1, 0])
    pre = np.array([0, 1, 0, 1, 1, 1, 1, 0])
    gt = np.array([0, 0, 1, 1, 1, 1])
    pre = np.array([0, 1, 0, 1, 1, 1])
    # print(gt.shape, pre.shape)

    eval = Evaluator(num_class=2)
    eval.add_batch(gt, pre)
    # print("confusion_matrix", eval.confusion_matrix)
    # print("OA", eval.OA())
    # print("Precision", eval.Precision())
    # print("Recall", eval.Recall())1
    # print("F1", eval.F1())
    # print("Kappa", eval.Kappa())
    # print("IOU", eval.Intersection_over_Union())

    # print("class accuracy", eval.Pixel_Accuracy_Class())
    print("classAcc 0: ", round(eval.class_precision()[0][0]*100, 2))
    print("classAcc 1: ", round(eval.class_precision()[0][1]*100, 2))
    # print("Macro-acc", eval.class_precision()[1].round(4)*100) # 宏观平均更适合于平衡的数据集
    # print("Micro-acc", eval.class_precision()[2].round(4)*100) # 微观平均更适合于不平衡的数据集
    print("OA        : ", round(np.nanmean(eval.OA())*100, 2))   # # 微观平均更适合于不平衡的数据集
    print("Precision : ", round(np.nanmean(eval.Precision())*100, 2))
    print("Recall    : ", round(np.nanmean(eval.Recall())*100, 2))
    print("F1        : ", round(np.nanmean(eval.F1())*100, 2))
    print("Kappa     : ", round(np.nanmean(eval.Kappa())*100, 2))
    # print("IOU      ", str(eval.Intersection_over_Union()))
    print("mIOU      : ", round(np.nanmean(eval.Intersection_over_Union())*100, 2))


    # print("get_tp_fp_tn_fn", eval.get_tp_fp_tn_fn())
    # print("Frequency_Weighted_IOU", eval.Frequency_Weighted_Intersection_over_Union())


