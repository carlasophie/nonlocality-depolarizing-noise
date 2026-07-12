import numpy as np
import pandas as pd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def Mermin_violation(value):
    """
    Checks if Mermin inequality is violated

    Args:
        value (_type_): measured value of the Mermin inequality
    Returns:
        bool: True if the inequality is violated, False otherwise
    """
    return int(value >= 1.0)

def Svetlichny_violation(value, N):
    """
    Checks if Svetlichny inequality is violated

    Args:
        value (_type_): measured value of the Svetlichny inequality
        N (int): number of qubits.
    Returns:
        bool: True if the inequality is violated, False otherwise
    """
    return int(value >= 2**(N-1))

def Svetlichny_violation_bound(N):
    """
    Returns the theoretical bound for Svetlichny inequality violation
    Args:
        N (int): number of qubits.
    Returns:
        int: theoretical bound for Svetlichny inequality violation
    """
    return (2**(N-1))



def MMD(x, y, kernel, bandwidth=None):
    """Compute the empirical maximum mean discrepancy. 
        The lower this value,
        the higher the evidence that both distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    # x and y as torch tensors
    x = torch.tensor(x, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)
    x = x.unsqueeze(0)   
    y = y.unsqueeze(0)
    # compute kernels
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    # compute distances used later when bandwidths are defined
    dxx = rx.t() + rx - 2. * xx 
    dyy = ry.t() + ry - 2. * yy 
    dxy = rx.t() + ry - 2. * zz 

    # initialize as zeros
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "rbf":
        if bandwidth is None:
            bandwidth = [0.01, 0.1, 1, 10, 100, 1000]
        for a in bandwidth:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
    else:
        raise ValueError('Unknown kernel: {}'.format(kernel))

    return torch.mean(XX + YY - 2. * XY).numpy()

def compute_mmd_mermin(df_list, bandwidth=None):
    mmd_list = np.zeros(len(df_list))
    # get distribution and ideal distribution from dataframe and compute MMD
    for (index, dataframe) in enumerate(df_list):
        noisy_dist = dataframe['Mermin_violation'].to_list()
        ideal_dist = dataframe['theoretical_value'].to_list()
        mmd_value = MMD(noisy_dist, ideal_dist, kernel = "rbf", bandwidth=bandwidth)
        mmd_list[index] = mmd_value
    return mmd_list

def compute_mmd_svet(df_list, bandwidth=None):
    mmd_list = np.zeros(len(df_list))
    # get distribution and ideal distribution from dataframe and compute MMD
    for (index, dataframe) in enumerate(df_list):
        noisy_dist = dataframe['Svetlichny_violation'].to_list()
        ideal_dist = dataframe['theoretical_value'].to_list()
        mmd_value = MMD(noisy_dist, ideal_dist, kernel = "rbf", bandwidth=bandwidth)
        mmd_list[index] = mmd_value
    return mmd_list


