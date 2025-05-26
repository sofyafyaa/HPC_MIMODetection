import numpy as np
from numpy.linalg import inv, pinv


def zf_detector(H, y):
    '''
    Zero-Forcing detection:
    x_hat = pinv(H) @ y
    H: (n_symb, n_subcarries, n_rx, n_tx)
    y: (n_symb, n_subcarries, n_rx)
    returns x_hat: (n_symb, n_subcarries, n_tx)
    '''
    H_pinv = np.linalg.pinv(H)  
    x_hat = np.matmul(H_pinv, y[..., np.newaxis])
    return x_hat[..., 0]


def mmse_detector(H, y, noise_var):
    '''
    MMSE detector without einsum.
    H: (n_symb, n_subcarriers, n_rx, n_tx)
    y: (n_symb, n_subcarriers, n_rx)
    noise_var: scalar noise variance
    returns: x_hat (n_symb, n_subcarriers, n_tx)
    '''
    n_symb, n_subc, n_rx, n_tx = H.shape
    eye = np.eye(n_tx)
    H_herm = np.conj(np.transpose(H, (0,1,3,2)))
    gram = np.matmul(H_herm, H)
    regularized = gram + noise_var * eye[np.newaxis, np.newaxis, :, :]
    inv = np.linalg.inv(regularized)
    W = np.matmul(inv, H_herm)
    y_expanded = y[..., np.newaxis]
    x_hat = np.matmul(W, y_expanded)
    return x_hat[..., 0]
