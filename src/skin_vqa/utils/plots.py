import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from ..constants import LABELS

def save_confusion_matrix(cm, out_png: str, title: str = 'Confusion Matrix'):
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(LABELS))
    plt.xticks(tick_marks, LABELS, rotation=45, ha='right')
    plt.yticks(tick_marks, LABELS)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()

def save_roc_curves(y_true, y_prob, out_png: str):
    plt.figure()
    n_classes = y_prob.shape[1]
    for i in range(n_classes):
        try:
            fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{LABELS[i]} (AUC={roc_auc:.2f})')
        except Exception:
            continue
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend(loc='lower right', fontsize='small')
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()

def save_pr_curves(y_true, y_prob, out_png: str):
    plt.figure()
    n_classes = y_prob.shape[1]
    for i in range(n_classes):
        try:
            precision, recall, _ = precision_recall_curve((y_true == i).astype(int), y_prob[:, i])
            plt.plot(recall, precision, label=f'{LABELS[i]}')
        except Exception:
            continue
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves (One-vs-Rest)')
    plt.legend(loc='lower left', fontsize='small')
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()
