import numpy as np


class runningScore(object):
    """
        n_classes: database的类别,包括背景
        ignore_index: 需要忽略的类别id,一般为未标注id, eg. CamVid.id_unlabel
    """

    def __init__(self, n_classes, ignore_index=None):
        self.n_classes = n_classes
        # 混淆矩阵被初始化为全零矩阵
        self.confusion_matrix = np.zeros((n_classes, n_classes))

        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            # 判断是否为整型，将其转化为元组
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

    # 用于计算混淆矩阵
    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2  # 类别数的平方
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            # 展平多维数组
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - pixel_acc:
            - class_acc: class mean acc
            - mIou :     mean intersection over union
            - fwIou:     frequency weighted intersection union
        """

        hist = self.confusion_matrix

        # ignore unlabel
        if self.ignore_index is not None:
            for index in self.ignore_index:
                hist = np.delete(hist, index, axis=0)
                hist = np.delete(hist, index, axis=1)

        acc = np.diag(hist).sum() / hist.sum()
        # 按行进行求和
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        # 求均值 acc_cls是一个数组
        acc_cls = np.nanmean(acc_cls)
        # 每个类别的交并比
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        # 平均交并比
        mean_iou = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fw_iou = (freq[freq > 0] * iu[freq > 0]).sum()

        # set unlabel as nan
        if self.ignore_index is not None:
            for index in self.ignore_index:
                iu = np.insert(iu, index, np.nan)

        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "pixel_acc: ": acc,
                "class_acc: ": acc_cls,
                "mIou: ": mean_iou,
                "fwIou: ": fw_iou,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count
