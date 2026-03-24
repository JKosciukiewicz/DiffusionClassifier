from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
import numpy as np


def calculate_val_metrics(y_true, y_pred, threshold=0.5):
    sure_y_true = y_true  # np.concatenate(y_true)
    sure_y_pred = y_true  # np.concatenate(y_pred)

    mask = (sure_y_pred != -1) & (sure_y_true != -1)
    sure_y_true = sure_y_true[mask]
    sure_y_pred = sure_y_pred[mask]

    # Convert probabilities to binary predictions
    binary_pred = (sure_y_pred > threshold).astype(int)

    try:
        roc_auc = roc_auc_score(sure_y_true, sure_y_pred)
    except:
        roc_auc = 0

    return {
        "accuracy": accuracy_score(sure_y_true, binary_pred),
        "recall": recall_score(sure_y_true, binary_pred),
        "f1_score": f1_score(sure_y_true, binary_pred),
        "roc_auc": roc_auc,
    }


def calculate_metrics(
    y_true, y_pred, y_true_bt, y_pred_bt, ex_rejected, cls_rejected, alpha
):
    """
    Calculates accuracy, precision, recall, and F1 score for the sure predictions.

    Parameters:
        y_true (np.array): True labels.
        y_pred (np.array): Predicted labels (output from apply_multiclass_thresholds).
        y_false (np.array): (Unused in this implementation; can be used for additional analysis).
        y_pred_false (np.array): (Unused in this implementation; can be used for additional analysis).

    Returns:
        dict: A dictionary with keys "accuracy", "precision", "recall", and "f1_score".
    """
    # Filter out predictions where the model is not sure (-1)
    sure_y_true = np.concatenate(y_true)
    sure_y_pred = np.concatenate(y_pred)
    sure_y_true_bt = np.concatenate(y_true_bt)
    sure_y_pred_bt = np.concatenate(y_pred_bt)

    # Filter out -1s for main predictions
    mask = (sure_y_pred != -1) & (sure_y_true != -1)
    sure_y_true = sure_y_true[mask]
    sure_y_pred = sure_y_pred[mask]

    # Filter out -1s for backup predictions
    mask_bt = (sure_y_pred_bt != -1) & (sure_y_true_bt != -1)
    sure_y_true_bt = sure_y_true_bt[mask_bt]
    sure_y_pred_bt = sure_y_pred_bt[mask_bt]

    # Accuracy: fraction of sure predictions that are correct
    accuracy = np.mean(sure_y_true == sure_y_pred)
    accuracy_bt = np.mean(sure_y_true_bt == sure_y_pred_bt)
    # Compute true positives, false positives, and false negatives for label 1 as the positive class
    TP = np.sum((sure_y_true == 1) & (sure_y_pred == 1))
    FP = np.sum((sure_y_true == 0) & (sure_y_pred == 1))
    FN = np.sum((sure_y_true == 1) & (sure_y_pred == 0))

    TP_bt = np.sum((sure_y_true_bt == 1) & (sure_y_pred_bt == 1))
    FP_bt = np.sum((sure_y_true_bt == 0) & (sure_y_pred_bt == 1))
    FN_bt = np.sum((sure_y_true_bt == 1) & (sure_y_pred_bt == 0))

    # Precision: TP / (TP + FP)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    precision_bt = TP_bt / (TP_bt + FP_bt) if (TP_bt + FP_bt) > 0 else 0.0

    # Recall: TP / (TP + FN)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    recall_bt = TP_bt / (TP_bt + FN_bt) if (TP_bt + FN_bt) > 0 else 0.0
    # F1 score: harmonic mean of precision and recall
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    f1_bt = (
        2 * precision_bt * recall_bt / (precision_bt + recall_bt)
        if (precision_bt + recall_bt) > 0
        else 0.0
    )

    # ROC AUC score
    try:
        roc_auc = roc_auc_score(sure_y_true, sure_y_pred)
    except:
        roc_auc = 0

    test_results = {
        "alpha": alpha,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "ROC_AUC": roc_auc,
        "unthrustworthy_ex": ex_rejected,
        "unthrustworthy_cls": cls_rejected,
        "accuracy_bt": accuracy_bt,
        "precision_bt": precision_bt,
        "recall_bt": recall_bt,
        "f1_score_bt": f1_bt,
    }

    return test_results


# def plot_ambiguous_class_distribution(y_pred_conf_bt):
#     """
#     Plots the distribution of positions where '1' appears in the prediction vectors.
#
#     Parameters:
#     y_pred_conf_bt (numpy.ndarray): 2D array of shape (10, batch_size) with binary vectors of size 10.
#     """
#     # Ensure y_pred_conf_bt is a numpy array
#     y_pred_conf_bt = np.array(y_pred_conf_bt)
#     print(y_pred_conf_bt.shape)
#     # Check if the input array has the expected shape
#     if y_pred_conf_bt.shape[1] != 10:
#         raise ValueError("Expected input array with shape (10, batch_size)")
#
#     # Sum across the batch dimension to count occurrences of '1' in each position
#     position_counts = np.sum(y_pred_conf_bt, axis=0)
#
#     # Plotting
#     plt.figure(figsize=(8, 4))
#     plt.bar(range(0, 10), position_counts, color='skyblue')
#     plt.xlabel("Position in Prediction Vector")
#     plt.ylabel("Frequency of '1's")
#     plt.title("Distribution of '1's Across Prediction Vector Positions")
#     plt.xticks(range(1, 11))
#     plt.show()
