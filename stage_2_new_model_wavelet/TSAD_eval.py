# --- NEW IMPORTS: Added for correct data preprocessing ---
from sklearn.preprocessing import StandardScaler
import warnings
import os
import numpy as np

# --- PHẦN 0: CÀI ĐẶT VÀ IMPORT ---
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow INFO messages

# --- METRIC CLASSES IMPLEMENTATION (No changes needed) ---
def pointwise_to_segmentwise(pointwise):
    segmentwise = []
    prev = -10
    for point in pointwise:
        if point > prev + 1:
            segmentwise.append([point, point])
        else:
            segmentwise[-1][-1] += 1
        prev = point
    return np.array(segmentwise)

def segmentwise_to_pointwise(segmentwise):
    pointwise = []
    for start, end in segmentwise:
        for point in range(start, end + 1):
            pointwise.append(point)
    return np.array(pointwise)

def pointwise_to_full_series(pointwise, length):
    anomalies_full_series = np.zeros(length)
    if len(pointwise) > 0:
        assert pointwise[-1] < length
        anomalies_full_series[pointwise] = 1
    return anomalies_full_series

def f1_from_pr(p, r, beta=1):
    if r == 0 and p == 0:
        return 0
    return ((1 + beta**2) * r * p) / (beta**2 * p + r)

def recall(*args, tp, fn):
    return 0 if tp + fn == 0 else tp / (tp + fn)

def precision(*args, tp, fp):
    return 0 if tp + fp == 0 else tp / (tp + fp)

class Binary_anomalies:
    def __init__(self, length, anomalies):
        self._length = length
        self._set_anomalies(anomalies)
    def _set_anomalies(self, anomalies):
        anomalies = np.array(anomalies, dtype=int)
        if self._is_pointwise(anomalies):
            anomalies_ptwise = anomalies
        elif self._is_segmentwise(anomalies):
            anomalies_ptwise = segmentwise_to_pointwise(anomalies)
        else:
            raise ValueError(f"Illegal shape of anomalies:\n{anomalies}")
        self.anomalies_ptwise = anomalies_ptwise
    def _is_pointwise(self, anomalies): return len(anomalies.shape) == 1
    def _is_segmentwise(self, anomalies): return len(anomalies.shape) == 2

class Binary_detection:
    def __init__(self, length, gt_anomalies, predicted_anomalies):
        self._length = length
        self._gt = Binary_anomalies(length, gt_anomalies)
        self._prediction = Binary_anomalies(length, predicted_anomalies)
    def get_gt_anomalies_ptwise(self): return self._gt.anomalies_ptwise
    def get_predicted_anomalies_ptwise(self): return self._prediction.anomalies_ptwise
    def get_predicted_anomalies_full_series(self): return pointwise_to_full_series(self._prediction.anomalies_ptwise, self._length)
    def get_gt_anomalies_full_series(self): return pointwise_to_full_series(self._gt.anomalies_ptwise, self._length)

class Pointwise_metrics(Binary_detection):
    def __init__(self, *args):
        super().__init__(*args)
        gt = self.get_gt_anomalies_full_series()
        pred = self.get_predicted_anomalies_full_series()
        self.tp = np.sum(pred * gt)
        self.fp = np.sum(pred * (1 - gt))
        self.fn = np.sum((1 - pred) * gt)

class Segmentwise_metrics(Pointwise_metrics):
    def __init__(self, *args):
        super().__init__(*args)
        gt_seg = pointwise_to_segmentwise(self.get_gt_anomalies_ptwise())
        pred_seg = pointwise_to_segmentwise(self.get_predicted_anomalies_ptwise())
        tp = 0
        for gt_anomaly in gt_seg:
            for pred_anomaly in pred_seg:
                if self._overlap(gt_anomaly, pred_anomaly):
                    tp += 1
                    break
        self.tp = tp
        self.fp = len(pred_seg) - tp
        self.fn = len(gt_seg) - tp
    def _overlap(self, anomaly1, anomaly2):
        return not (anomaly1[1] < anomaly2[0] or anomaly2[1] < anomaly1[0])

class Composite_f(Binary_detection):
    def get_score(self):
        pointwise = Pointwise_metrics(self._length, self.get_gt_anomalies_ptwise(), self.get_predicted_anomalies_ptwise())
        segmentwise = Segmentwise_metrics(self._length, self.get_gt_anomalies_ptwise(), self.get_predicted_anomalies_ptwise())
        r = recall(tp=segmentwise.tp, fn=segmentwise.fn)
        p = precision(tp=pointwise.tp, fp=pointwise.fp)
        return f1_from_pr(p, r)

class Affiliation(Binary_detection):
    def get_score(self):
        gt_segments = pointwise_to_segmentwise(self.get_gt_anomalies_ptwise())
        pred_segments = pointwise_to_segmentwise(self.get_predicted_anomalies_ptwise())
        if not len(gt_segments) and not len(pred_segments): return 1.0
        if not len(gt_segments) or not len(pred_segments): return 0.0

        matches = np.zeros((len(gt_segments), len(pred_segments)))
        for i, gt in enumerate(gt_segments):
            for j, pred in enumerate(pred_segments):
                matches[i,j] = self._overlap(gt, pred)

        recall_cardinality = np.sum(np.sum(matches, axis=1) > 0) / len(gt_segments)
        precision_cardinality = np.sum(np.sum(matches, axis=0) > 0) / len(pred_segments)

        return f1_from_pr(precision_cardinality, recall_cardinality)

    def _overlap(self, seg1, seg2):
        return not (seg1[1] < seg2[0] or seg2[1] < seg1[0])

class Temporal_distance(Binary_detection):
    def get_score(self):
        a = self.get_gt_anomalies_ptwise()
        b = self.get_predicted_anomalies_ptwise()
        return self._dist(a, b) + self._dist(b, a)
    def _dist(self, a, b):
        return np.sum([np.min(np.abs(b - pt)) if len(b) else self._length for pt in a])