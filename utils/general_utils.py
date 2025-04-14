

def pad_output_to_target(prediction, target):
    """
    Pads the prediction tensor's spatial dimensions to match the target tensor's.
    Assumes prediction is potentially smaller than target.
    """
    pred_h, pred_w = prediction.shape[-2], prediction.shape[-1]
    tgt_h, tgt_w = target.shape[-2], target.shape[-1]

    if pred_h == tgt_h and pred_w == tgt_w:
        # No padding needed
        return prediction

    # Calculate padding amounts (assuming center alignment)
    pad_h_total = tgt_h - pred_h
    pad_w_total = tgt_w - pred_w

    # F.pad format: (pad_left, pad_right, pad_top, pad_bottom)
    # Ensure non-negative padding
    pad_top = max(0, pad_h_total // 2)
    pad_bottom = max(0, pad_h_total - pad_top)
    pad_left = max(0, pad_w_total // 2)
    pad_right = max(0, pad_w_total - pad_left)

    # Apply padding (using 'constant' mode with value 0 by default)
    padded_prediction = F.pad(prediction, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)

    return padded_prediction