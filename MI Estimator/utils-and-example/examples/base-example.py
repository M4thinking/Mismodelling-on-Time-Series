"""
Basic example of the estimation of mutual information on random variables via the TSP-IT implementation

Camilo Ramírez C. - FCFM - Universidad de Chile
"""
#! /usr/bin/env python3
import numpy as np
import sys
import os
path= os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'utils')
print("parent of parent + utils", path)

sys.path.append("/media/sebag/Disco Externo/Drive-Seba/UChile/10mo semestre/Teoria de la Información/Mismodelling-on-Time-Series/MI Estimator/utils-and-example")
from utils.util_functions import data_plot, print_emi_output, emi_evolution_plot
from utils.tsp_functions import estimate_mi, compute_emi_evolution
from numpy.random import Generator

import matplotlib.pyplot as plt
import numpy as np

# Initializes a NumPy random generator
rng: Generator = np.random.default_rng(seed=1234)

# Sampling of X and Y, in where both distribute as Normal(0, 1)
data: np.ndarray = rng.normal(size=(2*10**3, 2))
x: np.ndarray = data[:, 0]
y: np.ndarray = data[:, 1]

# Estimation of the mutual information
post_emi, post_size, prev_emi, prev_size = estimate_mi(x=x, y=y)

# Printing of the emi results in console
print_emi_output(post_emi, post_size, prev_emi, prev_size)

# Build and show the scatter-plot of X and Y
splot: plt.Figure = data_plot(x=x, y=y, mode="scatter", alpha=0.3)
splot.show()

# Compute the emi and tree size evolution as the sample size increases
emi_size_evolution = compute_emi_evolution(x=x, y=y, stride=1, show_tqdm=True)

# Plots the evolution of the EMI
evol_plot: plt.Figure = emi_evolution_plot(evolution_array=emi_size_evolution, plot_post_emi=True, plot_post_size=True,
                                           plot_prev_emi=True, plot_prev_size=True)
evol_plot.savefig('evol_plot.png')
