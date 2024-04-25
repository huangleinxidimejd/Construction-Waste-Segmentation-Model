from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score


# 计算多分类问题准确率及其他指标
def multi_category_accuracy(pred, label, num_classes):
    """
    具体来说，align_dims函数可能是用来处理维度不匹配的情况，将维度调整为二维。
    pred和label变量经过这个函数后，变成了类似于(batch_size, num_classes)的形状。
    """
    valid = (label > 0)
    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)

    precisions = []
    for cls_pre in range(1, num_classes):
        TP = float(((pred == cls_pre) * (label == cls_pre)).sum())
        FP = float(((pred == cls_pre) * (label != cls_pre)).sum())
        precision = TP / (TP + FP + 1e-10)
        precisions.append(precision)

    recalls = []
    for cls_rec in range(1, num_classes):
        TP = float(((pred == cls_rec) * (label == cls_rec)).sum())
        FN = float(((pred != cls_rec) * (label == cls_rec)).sum())
        recall = TP / (TP + FN + 1e-10)
        recalls.append(recall)

    F1_scores = [2 * p * r / (p + r + 1e-10) for p, r in zip(precisions, recalls)]

    IoU = []
    for cls_iou in range(1, num_classes):
        TP = float(((pred == cls_iou) * (label == cls_iou)).sum())
        FP = float(((pred == cls_iou) * (label != cls_iou)).sum())
        FN = float(((pred != cls_iou) * (label == cls_iou)).sum())
        iou = TP / (TP + FP + FN + 1e-10)
        IoU.append(iou)

    return acc, precisions, recalls, F1_scores, IoU


true_lable = 'E:\\学习\\研究生-北京工业大学信息学部软件学院\\高分遥感图像识别\\建筑垃圾消纳场\\WasteSeg\\utils\\19-1.png'
prediction = 'E:\学习\研究生-北京工业大学信息学部软件学院\高分遥感图像识别\建筑垃圾消纳场\WasteSeg\utils\19-.png'
num_classes = ['0', '1', '2']

measure_result = classification_report(true_lable, prediction)
print('measure_result = \n', measure_result)


acc, precisions, recalls, F1_scores, IoU = multi_category_accuracy(prediction, true_lable, 3)
print(acc, precisions, recalls, F1_scores, IoU)


