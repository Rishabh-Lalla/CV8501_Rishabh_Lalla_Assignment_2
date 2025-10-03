from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix)
from ..constants import LABELS, LABEL2ID, ID2LABEL

def multiclass_metrics(y_true, y_prob):
    # y_prob: [N, C] softmax probabilities
    y_pred = np.argmax(y_prob, axis=1)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    # For AUC, need probability matrix; handle classes with single label by adjustment
    try:
        macro_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except Exception:
        macro_auc = float('nan')
    report = classification_report(y_true, y_pred, target_names=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'macro_auc': float(macro_auc),
        'report': report,
        'confusion_matrix': cm.tolist(),
    }
