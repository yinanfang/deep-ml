import numpy as np


def logit_margin(z, t):
    """Margin m = z_t - log sum_{j != t} exp(z_j), computed stably; returns a Python float."""
    other_logis = np.delete(z, t)
    max_logit = np.max(other_logis)
    lse = max_logit + np.log(np.sum(np.exp(other_logis - max_logit)))
    return float(z[t] - lse)


def precision_cliff(dtype):
    """Smallest m >= 0 at which 1/(1 + exp(-m)) == 1 when evaluated in dtype; bisection on a float64 m.

    Returns the margin rounded to 4 decimals.
    """
    low, high = 0.0, 100.0
    while (high - low) > 1e-9:
        mid = (low + high) / 2.0
        exp_neg_m = dtype(np.exp(-mid))
        one = dtype(1)
        prob = dtype(one / (one + exp_neg_m))
        if prob == one:
            high = mid
        else:
            low = mid
    return round((low + high) / 2.0, 4)
