from transformer import TransformerModel 
from torchvision import transforms
from datasets import EegDataModule
from datasets import EegDatasetNominal
import torch, os, tqdm
import matplotlib.pyplot as plt
import numpy as np
    
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor

import sys
import os
print("parent of parent + utils", os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'MI Estimator/Utils and example/utils'))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'MI Estimator/Utils and example/utils'))

# from util_functions import data_plot, print_emi_output, emi_evolution_plot
# from tsp_functions import estimate_mi, compute_emi_evolution

from numpy.random import Generator

import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Si pasamos un paciente a la vez, es decir, usando un for para un array 2D, entonces:

class MI_estimator:
    def __init__(self, model, eeg_signals, y):
        self.model = model
        self.eeg_signals = eeg_signals
        self.y = y
    def calculate_residuals(self):
        #assert len(eeg_signals.shape) == 2
        x=self.eeg_signals.shape[2]
        num_channels  = x
        print("num_channels", num_channels)
        residuals = np.zeros((100,self.eeg_signals.shape[2]))
        mutual_info = np.zeros(num_channels)
        print('eeeg_signals shape', self.eeg_signals.shape)
        signal = self.eeg_signals
        # Test
        #predicted = self.model(signal)

        for _ in range(100):
            y_hat = self.model(signal)[:, 0, :]
            signal = torch.cat((signal[:, 1:, :], y_hat.unsqueeze(1)), dim=1)
        output = signal[:, -100:, :]
        output = output.unsqueeze(0)
        signal = signal.cpu().detach().numpy()
        #assert len(signal.shape) == 2
        y = self.y.cpu().detach().numpy()
        #predicted = predicted.cpu().detach().numpy()
        output = output.cpu().detach().numpy()

        # for channel in range(num_channels):
        #     # calculo de los residuos
        #     print('signal shape', signal[0,:,channel].shape)
        #     print('predicted shape', output[:,channel].shape)
        #     print('y shape', y[0,:,channel].shape)
        #     residuals[:,channel] = y[0,:,channel]- output[0,:,channel]

            
        #     # calculo de la información mutua
        #     post_emi, post_size, prev_emi, prev_size = estimate_mi(x=signal[0,:,channel], y=residuals[0,:,channel])

        #     # se guarda en el diccionario
        #     mutual_info[channel] = post_emi

        # return residuals, mutual_info

    # def plot_data(self, data, channel):
    #     assert channel < data.shape[0]
    #     splot: plt.Figure = data_plot(x=range(data.shape[1]), y=data[channel, :], mode="scatter", alpha=0.3)
    #     splot.show()

    # def compute_emi_evolution(self, channel,y, stride=1):
    #     #assert channel < self.eeg_signals.shape[2]
    #     signal = self.eeg_signals
    #     predicted = self.model(signal)
    #     signal = signal.cpu().detach().numpy()
    #     #assert len(signal.shape) == 2
    #     y = y.cpu().detach().numpy()
    #     predicted = predicted.cpu().detach().numpy()
    #     residuals = y - predicted

    #     # se computea el la evolución del emi 
    #     emi_size_evolution = compute_emi_evolution(x=signal, y=residuals, stride=stride, show_tqdm=True)
    #     return emi_size_evolution
def get_chunked_example_with_label(idx,in_size=100, out_size=25, step=100):
    transform =  transforms.Compose([transforms.ToTensor()])
    dataset = EegDatasetNominal(transform)
    x, y = dataset[idx]
    x_chunked = torch.stack([x[:, i:i+in_size] for i in range(0, x.shape[1]-in_size-out_size, step)], dim=0)
    y = y[:x_chunked.shape[0]]
    print(x_chunked.shape)
    print(len(y))
    return x_chunked, y

                            
def main(version=4, in_size=256, out_size=64, step=64):
    # Cargar datos
    dm = EegDataModule(batch_size=256, in_size=in_size, out_size=out_size, step=step)
    dm.setup(force=False)
    # Entrenar
    best_model_path = f'tst_logs/transformer/version_{version}/checkpoints/'
    # hyperparams = {'d_model': 100,          # Dimensión de los embeddings
    #                 'nhead': 10,             # Número de cabezas de atención en paralelo
    #                 'num_layers': 8,        # Número de capas de la red
    #                 'dim_feedforward': 512, # Dimensión de la capa que sigue a la atención
    #                 'dropout': 0.1,         # Dropout
    #                 'learning_rate': 1e-4,  # Tasa de aprendizaje
    #                 'max_length': 256,      # Tamaño máximo de la secuencia de entrada
    #                 'reduction': None,         # Reducción de la dimensión de la salida Entrada/reduction
    #                 }
    
    # if not os.path.exists(best_model_path):
    #     best_model_path = None
    #     model = TransformerModel(**hyperparams).to(device); print('Modelo creado')
    # else:
    try:
        best_model_path += [f for f in os.listdir(best_model_path) if '.ckpt' in f][0]
        model = TransformerModel.load_from_checkpoint(best_model_path); print('Modelo cargado desde: ', best_model_path)
    except:
        print('Modelo no encontrado')
        return
          
    # Cantidad de parámetros
    print('Cantidad de parámetros: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    x, y = dm.test_dataset[1]
    x = x.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device)
    # Calcular Información Mutua
  
    EMI = MI_estimator(model, x, y)
    EMI.calculate_residuals()
    print( 'EMI shape',EMI.shape)
    #print("Residual size:", .shape)
    #print("MI size:", .shape)
 



if __name__ == '__main__':
    get_chunked_example_with_label(0)
    # main(version=0, in_size=100, out_size=100, step=25)
   