import sys
import os
print("parent of parent + utils", os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'utils'))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'utils'))

from utils.util_functions import data_plot, print_emi_output, emi_evolution_plot
from utils.tsp_functions import estimate_mi, compute_emi_evolution

from numpy.random import Generator

import matplotlib.pyplot as plt
import numpy as np


class MI_estimator:
    def __init__(self, model, eeg_signals):
        self.model = model
        self.eeg_signals = eeg_signals

    def calculate_residuals(self):
        residuals = {}
        mutual_info = {}
        for patient, signals in self.eeg_signals.items():
            residuals[patient] = np.zeros_like(signals)
            mutual_info[patient] = np.zeros_like(signals)
            for i in range(signals.shape[0]):
                predicted = self.model.predict(signals[i, :])

                # Calculo de los residuos
                residuals[patient][i, :] = signals[i, :] - predicted

                # calculo de la información mutua
                post_emi, post_size, prev_emi, prev_size = estimate_mi(x=signals[i, :], y=predicted)

                # se guarda en el diccionario
                mutual_info[patient][i, :] = post_emi
        return residuals, mutual_info

    def plot_data(self, data, patient, channel):
        splot: plt.Figure = data_plot(x=range(len(data[patient][channel, :])), y=data[patient][channel, :], mode="scatter", alpha=0.3)
        splot.show()


class MI_estimator:
    def __init__(self, model, eeg_signals):
        self.model = model
        self.eeg_signals = eeg_signals

    def calculate_residuals(self):
        num_patients, num_channels, _ = self.eeg_signals.shape
        residuals = np.zeros_like(self.eeg_signals)
        mutual_info = np.zeros((num_patients, num_channels))

        for patient in range(num_patients):
            for channel in range(num_channels):
                signal = self.eeg_signals[patient, channel, :]
                predicted = self.model.predict(signal)

                # Calculo de los residuos
                residuals[patient, channel, :] = signal - predicted

                # calculo de la información mutua
                post_emi, post_size, prev_emi, prev_size = estimate_mi(x=signal, y=predicted)

                # se guarda en el diccionario
                mutual_info[patient, channel] = post_emi

        return residuals, mutual_info

    def plot_data(self, data, patient, channel):
        splot: plt.Figure = data_plot(x=range(data.shape[2]), y=data[patient, channel, :], mode="scatter", alpha=0.3)
        splot.show()

    def compute_emi_evolution(self, patient, channel, stride=1):
        signal = self.eeg_signals[patient, channel, :]
        predicted = self.model.predict(signal)
        residuals = signal - predicted

        # computea la evolucion del emi
        emi_size_evolution = compute_emi_evolution(x=signal, y=residuals, stride=stride, show_tqdm=True)

        return emi_size_evolution

    def plot_emi_evolution(self, emi_size_evolution):
        # evolución del emi
        evol_plot: plt.Figure = emi_evolution_plot(evolution_array=emi_size_evolution, plot_post_emi=True, plot_post_size=True,
                                                   plot_prev_emi=True, plot_prev_size=True)
        evol_plot.show()


class MI_estimator:
    def __init__(self, model, eeg_signals):
        assert len(eeg_signals.shape) == 3
        self.model = model
        self.eeg_signals = eeg_signals

    def calculate_residuals(self):
        num_patients, num_channels, _ = self.eeg_signals.shape
        residuals = np.zeros_like(self.eeg_signals)
        mutual_info = np.zeros((num_patients, num_channels))

        for patient in range(num_patients):
            for channel in range(num_channels):
                signal = self.eeg_signals[patient, channel, :]
                predicted = self.model.predict(signal)

                # calculo de los residuos
                residuals[patient, channel, :] = signal - predicted

                # calculo de la información mutua
                post_emi, post_size, prev_emi, prev_size = estimate_mi(x=signal, y=predicted)

                # se guarda en el diccionario
                mutual_info[patient, channel] = post_emi

        return residuals, mutual_info

    def plot_data(self, data, patient, channel):
        assert patient < data.shape[0]
        assert channel < data.shape[1]
        splot: plt.Figure = data_plot(x=range(data.shape[2]), y=data[patient, channel, :], mode="scatter", alpha=0.3)
        splot.show()

    def compute_emi_evolution(self, patient, channel, stride=1):
        assert patient < self.eeg_signals.shape[0]
        assert channel < self.eeg_signals.shape[1]
        signal = self.eeg_signals[patient, channel, :]
        predicted = self.model.predict(signal)
        residuals = signal - predicted

        # computea la evolucion del emi
        emi_size_evolution = compute_emi_evolution(x=signal, y=residuals, stride=stride, show_tqdm=True)
        return emi_size_evolution



#Si pasamos un paciente a la vez, es decir, usando un for para un array 2D, entonces:

class MI_estimator:
    def __init__(self, model, eeg_signals):
        assert len(eeg_signals.shape) == 2
        self.model = model
        self.eeg_signals = eeg_signals

    def calculate_residuals(self):
        num_channels, _ = self.eeg_signals.shape
        residuals = np.zeros_like(self.eeg_signals)
        mutual_info = np.zeros(num_channels)

        for channel in range(num_channels):
            signal = self.eeg_signals[channel, :]
            predicted = self.model.predict(signal)

            # calculo de los residuos
            residuals[channel, :] = signal - predicted

            # calculo de la información mutua
            post_emi, post_size, prev_emi, prev_size = estimate_mi(x=signal, y=predicted)

            # se guarda en el diccionario
            mutual_info[channel] = post_emi

        return residuals, mutual_info

    def plot_data(self, data, channel):
        assert channel < data.shape[0]
        splot: plt.Figure = data_plot(x=range(data.shape[1]), y=data[channel, :], mode="scatter", alpha=0.3)
        splot.show()

    def compute_emi_evolution(self, channel, stride=1):
        assert channel < self.eeg_signals.shape[0]
        signal = self.eeg_signals[channel, :]
        predicted = self.model.predict(signal)
        residuals = signal - predicted

        # se computea el la evolución del emi 
        emi_size_evolution = compute_emi_evolution(x=signal, y=residuals, stride=stride, show_tqdm=True)
        return emi_size_evolution

