import numpy as np
import scipy as sp
from scipy.special import logsumexp, log_softmax

def log_mat_vec_mul(mat, vec):
    """
    Log of dot product of a mat and a vec.

    Args:
        mat: (num_site, num_state, num_state)
        vec: (num_site, 1, num_state)

    Returns:
        (num_site, num_state)
    """
    # assert mat.shape[1] == vec.shape[0], "shape is not match."
    v = mat + vec # broadcasting
    # print(v[0].round(2), logsumexp(v, axis=-1).round(2)[0])
    return logsumexp(v, axis=-1)


def log_mat_vec_max(mat, vec):
    """
    Log of maximum of pointwise multiplication. 
    Consider all sites when picking the maximal one.

    Args:
        mat: (num_site, num_state, num_state)
        vec: (num_site, 1, num_state)
    
    Returns:
        (num_site, num_state)
    """
    # assert mat.shape[1] == vec.shape[0], "shape is not match."
    v = mat + vec # broadcasting
    # sum_v = v.sum(axis=0) # (num_state, num_state)
    # arg = sum_v.argmax(axis=-1) # (num_state)
    # arg = arg.reshape(1, -1, 1) # (1, num_state, 1)
    # val = np.take_along_axis(v, arg, axis=-1) # (num_site, num_state)
    val = v.max(axis=-1)
    arg = v.argmax(axis=-1)
    return val, arg


def log_matmul(logA, logB):
    """
    Compute matrix multiplication in log space: log(A @ B)   
    logA: (N, p, p) array 
    logB: (N, p, p) array 
    Returns:
    logC: (N, p, p) array, where logC = log(A @ B)
    """
    # logA has shape (N, p, p)
    # logB has shape (N, p, p)
    # Compute log(A @ B) using broadcasting
    product = logsumexp(logA[:, :, :, np.newaxis] + logB[:, np.newaxis, :, :], axis=2)
    return product




