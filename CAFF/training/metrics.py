from typing import List
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, average_precision_score


def compute_masked_metrics(scores: List[np.ndarray], labels: List[np.ndarray], masks: List[np.ndarray], threshold: float = 0.5):
    all_scores = []
    all_labels = []

    for score_arr, label_arr, mask_arr in zip(scores, labels, masks):
        valid = mask_arr.flatten().astype(bool)
        all_scores.append(score_arr.flatten()[valid])
        all_labels.append(label_arr.flatten()[valid])

    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    pred = (all_scores >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, pred, average='binary', zero_division=0)
    map_score = average_precision_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0.0
    return {
        'precision': p,
        'recall': r,
        'f1': f1,
        'map': map_score,
    }


def hop_stratified_precision(scores_by_hop, labels_by_hop, masks_by_hop, threshold=0.5):
    precisions = []
    for scores, labels, masks in zip(scores_by_hop, labels_by_hop, masks_by_hop):
        valid = masks.flatten().astype(bool)
        if valid.sum() == 0:
            precisions.append(0.0)
            continue
        preds = (scores.flatten()[valid] >= threshold).astype(int)
        labels_flat = labels.flatten()[valid].astype(int)
        tp = int((preds & labels_flat).sum())
        p = tp / preds.sum() if preds.sum() > 0 else 0.0
        precisions.append(p)
    return precisions
