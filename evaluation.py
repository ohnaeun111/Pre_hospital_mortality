import numpy as np
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score
)

def get_clf_eval(y_true, y_pred, y_proba):
    """
    Compute classification metrics including accuracy, precision, recall,
    specificity, F1 score, AUC, AUPRC, and balanced accuracy.
    """
    # Confusion matrix
    confusion = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = confusion.ravel()

    # Basic metrics
    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall    = recall_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred)

    # Curve metrics
    roc_auc   = roc_auc_score(y_true, y_proba)
    pr_auc    = average_precision_score(y_true, y_proba)

    # Derived metrics
    specificity   = tn / (tn + fp)
    balanced_acc  = (recall + specificity) / 2

    # Print results
    print('Confusion Matrix:')
    print(confusion)
    print(
        'Accuracy: {:.4f}, Precision: {:.4f}, Sensitivity: {:.4f}, '
        'Specificity: {:.4f}, F1 Score: {:.4f}, AUC: {:.4f}, '
        'AUPRC: {:.4f}, Balanced Accuracy: {:.4f}'
        .format(accuracy, precision, recall, specificity, f1, roc_auc, pr_auc, balanced_acc)
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'balanced_accuracy': balanced_acc
    }
