import numpy as np
import math

def calculate_itr(classification_acc, num_targets, target_selection_time_seconds):
    p = classification_acc
    logp = np.log2(p)
    ip = 1.0 - p

    a = np.log2(num_targets)
    b = p * logp
    c = ip * np.log2( ip/(num_targets-1) )
    result = a + b + c
    return result * (60 / target_selection_time_seconds)