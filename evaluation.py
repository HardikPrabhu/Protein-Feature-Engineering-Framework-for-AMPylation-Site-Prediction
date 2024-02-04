""" Method to generate evaluation report"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef


def classification_report(y_true, y_pred, y_prob):
    """
    Evaluate binary classification metrics.

    Parameters
    -----------
     y_true: list or array, true labels.
     y_pred: list or array, predicted labels.
     y_prob: list or array, predicted probabilities for the positive class.

    Returns
    ---------
     A dictionary with accuracy, precision, recall, f1-score, AUC-ROC, and MCC.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "auc_roc": roc_auc_score(y_true, y_prob),
        "mcc": matthews_corrcoef(y_true, y_pred)
    }
    return metrics


if __name__ == "__main__":
    true = [0, 1, 1, 1, 0]
    predicted = [0, 0, 0, 1, 0]
    prob = [0.1, 0, 0.2, 0.8, 0.1]
    print(classification_report(true, predicted, prob))
