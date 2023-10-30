"""
TSP functions, this file is meant to contain functions that are directly related to the TSP algorith and the estimation
of mutual information

Camilo Ramírez C. - FCFM - Universidad de Chile
"""
from utils.util_functions import reshape_array_as_2d
from utils.tsp import TSP
from typing import Union, Tuple
from tqdm import tqdm

import numpy as np


def estimate_mi(x: np.ndarray, y: np.ndarray, l_bn: float = 0.167, w_bn: float = 5e-2,
                lambda_factor: float = 2e-5) -> Tuple[float, int, float, int]:
    """
    Estimates the mutual information from the samples of two variables X and Y vía TSP, and returns the estimation of
    MI with the size of the tree previous and after the regularization

    Parameters
    ----------
    x: np.ndarray
        Samples of X
    y: np.ndarray
        Samples of Y
    l_bn: float
        Exponent of the TSP cells refinement threshold, note that bn = w_bn * n**l_bn, this value must be in the
        interval (0, 1/3)
    w_bn: float
        Weighting factor of the TSP cells refinement threshold, note that bn = w_bn * n**l_bn
    lambda_factor: float
        Regularization factor for the tree structured partitions

    Returns
    -------
    Tuple[float, int, float, int]
        A tuple that contains:
        - The estimated mutual information after the regularization
        - The size of the tree after the regularization
        - The estimated mutual information previous to the regularization
        - The size of the tree previous to the regularization
    """
    x = np.copy(reshape_array_as_2d(x), order='F')
    y = np.copy(reshape_array_as_2d(y), order='F')
    tsp = TSP(l_bn=l_bn, w_bn=w_bn, lambda_factor=lambda_factor)
    tsp.grow(x=x, y=y)
    prev_size: int = tsp.size()
    prev_emi: float = tsp.emi()
    tsp.regularize()
    post_size: int = tsp.size()
    post_emi: float = tsp.emi()
    return post_emi, post_size, prev_emi, prev_size


def compute_emi_evolution(x: np.ndarray, y: np.ndarray, stride: int = 1, l_bn: float = 0.167, w_bn: float = 5e-2,
                          lambda_factor: float = 2.3e-5, show_tqdm: bool = False) -> np.ndarray:
    """
    Function that computes the emi evolution of X and Y from their samples considering a window of growing size over
    the iterations, this window starts at index 0, and ends at index iteration*stride

    Parameters
    ----------
    x: np.ndarray
        Samples of X
    y: np.ndarray
        Samples of Y
    stride: int
        The step size of the numbers of samples that are considered over the iterations
    l_bn: float
        Exponent of the TSP cells refinement threshold, note that bn = w_bn * n**l_bn
    w_bn: float
        Weighting factor of the TSP cells refinement threshold, note that bn = w_bn * n**l_bn, this value must be
        in the interval (0, 1/3)
    lambda_factor: float
        Regularization factor for the tree structured partitions
    show_tqdm: bool
        Boolean that indicates if tqdm is shown or not

    Raises
    ------
    ValueError
        If the stride is an invalid number, i.e., a non-positive number or a number greater than the size of the arrays

    Returns
    -------
    evolution_output: np.ndarray
        A numpy array that contains the evolution of every value that estimate_mi returns, this array contains 5
        indexes on its first dimension, every one consists on the following arrays
        - Number of samples
        - Evolution of the EMI after regularization
        - Evolution of the tree size after regularization
        - Evolution of the EMI previous regularization
        - Evolution of the tree size previous regularization

    See also
    --------
    The documentation of the function estimate_mi
    """
    if stride <= 0 or type(stride) != int:
        raise ValueError("The stride must be a positive integer")
    min_sample_count: int = min(x.shape[0], y.shape[0])
    if stride > min_sample_count:
        raise ValueError("The stride value must be less than the minimum between the sample size of x and y")
    total_iterations: int = int(np.floor(min_sample_count/stride))
    evolution_output: np.ndarray = np.empty(shape=(5, total_iterations))
    iter_idx: int
    iter_range: Union[range, tqdm] = tqdm(range(total_iterations)) if show_tqdm else range(total_iterations)
    for iter_idx in iter_range:
        iteration_emi: (float, int, float, int) = estimate_mi(x=x[:stride*(iter_idx+1)], y=y[:stride*(iter_idx+1)],
                                                              l_bn=l_bn, w_bn=w_bn, lambda_factor=lambda_factor)
        evolution_output[0, iter_idx] = stride*(iter_idx+1)
        output_idx: int
        for output_idx in range(4):
            evolution_output[output_idx+1, iter_idx] = iteration_emi[output_idx]
    return evolution_output
