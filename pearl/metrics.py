import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, roc_auc_score)


def edge_metrics(y, q):
    y, q = np.asarray(y), np.asarray(q)
    if not len(y):
        return {"n": 0}
    calibration = []
    ece = 0.0
    for i in range(10):
        mask = (q >= i / 10) & (q < (i + 1) / 10 if i < 9 else q <= 1)
        if mask.any():
            error = abs(float(y[mask].mean() - q[mask].mean()))
            ece += mask.mean() * error
            calibration.append(dict(bin=i, n=int(mask.sum()), confidence=float(q[mask].mean()),
                                    observed=float(y[mask].mean())))
    two = len(np.unique(y)) == 2
    return dict(n=len(y), positive_rate=float(y.mean()), auroc=float(roc_auc_score(y, q)) if two else None,
                auprc=float(average_precision_score(y, q)) if two else None,
                brier=float(brier_score_loss(y, q)), ece=float(ece),
                log_loss=float(log_loss(y, np.clip(q, 1e-7, 1 - 1e-7), labels=[0, 1])),
                q_mean=float(q.mean()), q_std=float(q.std()), calibration=calibration)


def distribution(q):
    q = np.asarray(q)
    return dict(n=len(q), mean=float(q.mean()), std=float(q.std()),
                p05=float(np.quantile(q, .05)), p95=float(np.quantile(q, .95)),
                near_zero=float((q < .05).mean()), near_one=float((q > .95).mean())) if len(q) else {"n": 0}


def classification_metrics(y, p):
    pred = np.asarray(p) >= .5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(n=len(y), accuracy=float(accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
                human_f1=float(f1_score(y, pred, zero_division=0)),
                auroc=float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
                human_recall=float(tp / (tp + fn)) if tp + fn else None,
                ai_recall=float(tn / (tn + fp)) if tn + fp else None,
                confusion_ai_human=[[int(tn), int(fp)], [int(fn), int(tp)]],
                bce=float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])))
