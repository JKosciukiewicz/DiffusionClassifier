import numpy as np


def multiclass_non_conformity_score(y_pred_prob, y, mask=None):
    """
    Computes per-class nonconformity scores for a single sample.

    Both y_pred_prob and y are expected to be 1D PyTorch tensors of length n_classes.
    mask (optional) is a boolean array/tensor of the same length, where:
        True  -> class should be ignored (unknown target)
        False -> class should be scored

    For each class:
      If the true label is 1, score = 1 - predicted probability.
      If the true label is 0, score = predicted probability.
      If masked, score = np.nan (ignored in later aggregation).

    Returns:
      scores: numpy array of shape (n_classes,)
    """
    # Convert to numpy
    y_pred_prob = (
        np.array(y_pred_prob.cpu())
        if hasattr(y_pred_prob, "cpu")
        else np.array(y_pred_prob)
    )
    y = np.array(y)

    scores = np.where(y == 1, 1 - y_pred_prob, y_pred_prob)

    if mask is not None:
        mask = np.array(mask)
        scores = np.where(mask, scores, np.nan)  # NaN out unknown targets

    return scores


def multiclass_conformal_thresholds(alpha, calibration_scores):
    """
    Compute per-class thresholds based on calibration scores.

    Parameters:
      alpha: significance level such that thresholds are set at the (1 - alpha) quantile.
      calibration_scores: list (or array) of calibration nonconformity score vectors,
                          one per calibration example (shape: [n_samples, n_classes]).

    Returns:
      thresholds: numpy array of shape (n_classes,)
    """
    calib_scores = np.array(calibration_scores)  # shape: [n_samples, n_classes]
    # thresholds = np.quantile(calib_scores, 1 - alpha, axis=0)
    thresholds = np.nanquantile(calib_scores, 1 - alpha, axis=0)

    return thresholds


def apply_multiclass_thresholds(y_pred, y_true, thresholds):
    """
    Apply conformal thresholds to one sample's predictions.

    NOW HANDLES BOTH SCALAR AND VECTOR INPUTS!

    Parameters:
      y_pred: numpy array, tensor, or scalar of predicted probabilities for a single sample.
              If scalar, treats as single binary classification.
              If vector, treats as multiclass with one prediction per class.
      y_true: numpy array, tensor, or scalar of true labels for that sample.
      thresholds: numpy array or scalar of thresholds (should match dimensionality of y_pred).

    Returns a tuple:
      y_pred_out: conformal predictions (1, 0, or -1 if ambiguous)
      y_true_out: corresponding true labels (or -1 if not available)
      ex_rejected: 1 if prediction is ambiguous, else 0.
      cls_rejected: count of ambiguous predictions.
      y_pred_bt, y_true_bt: alternative predictions for ambiguous cases.
    """
    # Convert inputs to numpy arrays
    if hasattr(y_pred, 'cpu'):
        y_pred = y_pred.cpu().numpy()
    else:
        y_pred = np.array(y_pred)

    if hasattr(y_true, 'cpu'):
        y_true = y_true.cpu().numpy()
    else:
        y_true = np.array(y_true)

    thresholds = np.array(thresholds)

    # Handle scalar inputs by converting to single-element arrays
    if y_pred.ndim == 0:  # 0-dimensional (scalar)
        y_pred = np.array([y_pred])
    if y_true.ndim == 0:  # 0-dimensional (scalar)
        y_true = np.array([y_true])
    if thresholds.ndim == 0:  # 0-dimensional (scalar)
        thresholds = np.array([thresholds])

    y_pred_out = []
    y_true_out = []
    y_pred_bt = []
    y_true_bt = []
    ex_rejected = 0
    cls_rejected = 0
    ex_unknown = 0
    cls_unknown = 0
    thresholds_crossed = False

    for idx, pred in enumerate(y_pred):
        # Handle case where thresholds might be shorter than y_pred
        threshold = thresholds[min(idx, len(thresholds) - 1)]

        # Compute candidate nonconformity scores:
        score_pos = pred  # if candidate label is 1
        score_neg = pred  # if candidate label is 0
        # threshold above 0.5
        accept_pos = pred >= threshold
        accept_neg = pred <= 1 - threshold

        if threshold < 0.5:
            accept_pos = False
            accept_neg = False
            thresholds_crossed = True
        else:
            thresholds_crossed = False

        # Get corresponding true label
        true_label = y_true[min(idx, len(y_true) - 1)]

        # Decide prediction:
        if accept_pos and not accept_neg:
            y_pred_out.append(1)
            y_true_out.append(
                true_label.item() if hasattr(true_label, "item") else true_label
            )
            y_pred_bt.append(-1)
            y_true_bt.append(-1)
        elif accept_neg and not accept_pos:
            y_pred_out.append(0)
            y_true_out.append(
                true_label.item() if hasattr(true_label, "item") else true_label
            )
            y_pred_bt.append(-1)
            y_true_bt.append(-1)
        elif not thresholds_crossed:
            # Ambiguous case: either both or neither candidate is acceptable.
            y_pred_out.append(-1)
            y_true_out.append(-1)
            # Optionally, break the tie using a fixed decision threshold (0.5 here)
            decision_threshold = 0.5
            y_pred_bt.append(1 if pred >= decision_threshold else 0)
            y_true_bt.append(
                true_label.item() if hasattr(true_label, "item") else true_label
            )
            ex_rejected = 1
            cls_rejected += 1
        else:
            ex_unknown += 1
            cls_unknown += 1

    return (
        np.array(y_pred_out, dtype="float32"),
        np.array(y_true_out, dtype="float32"),
        ex_rejected,
        cls_rejected,
        np.array(y_pred_bt, dtype="float32"),
        np.array(y_true_bt, dtype="float32"),
    )