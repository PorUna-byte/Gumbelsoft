from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
def auroc_fpr_fnr(labels, scores):
    # Calculate the AUROC
    auroc = roc_auc_score(labels, scores)
    # Calculate the ROC curve which includes FPR, TPR, and thresholds
    fprs, tprs, thresholds = roc_curve(labels, scores)
    fnr_at_1_fpr = 1-tprs[np.where(fprs <= 0.01)[0][-1]]
    fpr_at_1_fnr = fprs[np.where(tprs >= 0.99)[0][0]]
    return auroc, fpr_at_1_fnr.item(), fnr_at_1_fpr.item()


