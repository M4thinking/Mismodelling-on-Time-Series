"""
Utilities for this repository
This package is meant to contain functions and classes that are useful for the implementations and experiments

Camilo Ramírez C. - FCFM - Universidad de Chile
"""
from util_functions import add_tsp_path, reshape_array_as_2d, data_plot, print_emi_output, emi_evolution_plot,\
                                 plot_signals
from tsp_functions import estimate_mi, compute_emi_evolution
from tsp import TSP

from typing import Union, List

import numpy as np

Number = Union[int, float]
NumberArray = Union[List[Number], np.ndarray]
