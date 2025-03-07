import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def get_tnr(label, conf, fixed_fnr):
    fpr, tpr, thresholds = roc_curve(label, conf)
    target_tpr = 1 - fixed_fnr
    idx = np.where(tpr >= target_tpr)[0][0]

    # Get the corresponding threshold and tnr
    threshold = thresholds[idx]
    tnr_at_fixed_fnr = 1 - fpr[idx]  # tnr = 1 - fpr

    return threshold, tnr_at_fixed_fnr
