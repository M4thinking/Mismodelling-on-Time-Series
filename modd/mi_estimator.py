from sympy import true
from transformer import TransformerModel
from torchvision import transforms
from datasets import EegDatasetNominal, EegDatasetConvulsions
import torch, os, tqdm, sys, joblib
import numpy as np, matplotlib.pyplot as plt

print("Added path: ", os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'MI Estimator/Utils and example/utils'))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'MI Estimator/Utils and example/utils'))

from util_functions import data_plot, print_emi_output, emi_evolution_plot
from tsp_functions import estimate_mi, compute_emi_evolution

from numpy.random import Generator

import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MI_estimator:
    def __init__(self, model):
        self.model = model

    def filename(self, sample_idx, nominal):
        self.filename = f'{sample_idx}_{int(nominal)}'

    def plot_data(self, data, channel):
        assert channel < data.shape[0]
        splot: plt.Figure = data_plot(x=range(data.shape[1]), y=data[channel, :], mode="scatter", alpha=0.3)
        splot.show()

    def compute_emi_evolution(self, channel,y, stride=1):
        assert channel < self.input_scale.shape[0]
        signal = self.input_scale[0,:,channel]
        predicted = self.model(signal)
        signal = signal.cpu().detach().numpy()
        assert len(signal.shape) == 2
        y = y.cpu().detach().numpy()
        predicted = predicted.cpu().detach().numpy()
        residuals = y - predicted

        # se computea el la evolución del emi 
        emi_size_evolution = compute_emi_evolution(x=signal, y=residuals, stride=stride, show_tqdm=True)
        return emi_size_evolution
    

    def get_chunked_sample_wlabel(self, idx, in_size=100, step=100, nominal = True, test = True):
        # Cargar datos
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = EegDatasetNominal(transform) if nominal else EegDatasetConvulsions(transform)
        _ , _ ,self.names, self.real_idx = dataset.my_item(idx)
        x, y = dataset[idx]
        if test == True:
            x = x[:512]
        # Chuncked data
        print('Original data shape & labels', x.shape, y.shape)
        input_chunks = torch.stack([x[:, i:i+in_size] for i in range(0, x.shape[1], step)], dim=0)
        input_chunks = input_chunks.permute(0,2,1).to(device)
        self.input_chunks = input_chunks
        self.labels = y[1:input_chunks.shape[0]] # N-1 labels (N, 1) para comparar con la predicción del MI
        print('Chunked data shape & labels', input_chunks.shape, self.labels.shape)


    def get_prediction(self, batch_size=512):
        input_chunks = self.input_chunks.float()  # (N, 100, 21)
        output_chunks = torch.zeros_like(input_chunks)  # (N, 100, 21)
        
        # Escalamiento
        self.pwd = os.path.dirname(os.path.abspath(__file__))
        scaler = joblib.load(os.path.join(self.pwd, 'scaler.pkl'))
        print("Scaler mean & stv shape:", scaler.center_.shape, scaler.scale_.shape)
        normalizer = lambda x: torch.from_numpy(scaler.transform(x.cpu().numpy())).to(device)
        input_chunks = torch.stack([normalizer(input_chunks[i]) for i in range(len(input_chunks))], dim=0)
        self.input_scale = input_chunks
    
        num_chunks = input_chunks.shape[0]
        num_batches = num_chunks // batch_size
        remainder = num_chunks % batch_size

        def process_input(current_input):
            with torch.no_grad(): # No calcular gradientes
                self.model.eval() # Desactivar dropout
                for _ in range(len(current_input[0])):  # iterar sobre las 100 muestras
                    y_hat = self.model(current_input)[:, 0, :]   # obtener solo la primera pred (:, 21)
                    current_input = torch.cat((current_input[:, 1:, :], y_hat.unsqueeze(1)), dim=1)  # nuevo input (100, 21)
                return current_input

        for i in tqdm.tqdm(range(num_batches)):
            start = i * batch_size
            end = start + batch_size
            output_chunks[start:end] = process_input(input_chunks[start:end])  # (batch_size, 100, 21)

        if remainder > 0:
            print('Remainder')
            start = num_batches * batch_size
            output_chunks[start:] = process_input(input_chunks[start:]) # (remainder, 100, 21)
        
        # N-1 x 100 x 21 para plotear
        self.x_t = input_chunks[:-1].cpu().numpy() 
        self.y_t = input_chunks[1:].cpu().numpy()
        self.y_hat_t = output_chunks[:-1].cpu().numpy()
        
        # N-1 x 100 x 21 para estimar MI
        self.mi_input = self.x_t # X para el MI
        self.mi_residuals = self.y_t - self.y_hat_t # Y para el MI

        #añadir ruido
        
        #self.mi_input += np.random.default_rng(1).uniform(-1e-5, 1e-5, self.mi_input.shape)
        self.mi_input += np.random.default_rng(1).normal(0, 1e-5, self.mi_input.shape)

        #self.mi_residuals += np.random.default_rng(2).uniform(-1e-5, 1e-5, self.mi_residuals.shape)
        self.mi_residuals += np.random.default_rng(2).normal(0, 1e-5, self.mi_residuals.shape)

        print('X, Y, Y_pred shapes', self.x_t.shape, self.y_t.shape, self.y_hat_t.shape)
        print('MI input & residuals shapes', self.mi_input.shape, self.mi_residuals.shape)
    
    def plot_signal_prediction(self,idx):
        fig, axs = plt.subplots(7, 3, figsize=(15, 15), sharex=True)
        axs_f = axs.flatten()
        for i in range(21):
            axs_f[i].plot(np.arange(100), self.x_t[idx,:,i], label='Input' if i==0 else "")
            axs_f[i].plot(np.arange(100)+100, self.y_t[idx,:,i], label='Output' if i==0 else "")
            axs_f[i].plot(np.arange(100)+100, self.y_hat_t[idx,:,i], label='Prediction' if i==0 else "")
            axs_f[i].set_title(f'Canal {self.names[i]}')
        # Colocamos la leyenda en la parte inferior
        fig.legend(loc='lower center', ncol=3, fontsize = 15, bbox_to_anchor=(0.5, 0.03))
        # Colocamos el título en la parte superior
        fig.suptitle(f'Individuo {self.real_idx} - EEG s/ convulsión - Entrada, Predicción y Salida por Canal', fontsize=20, y=0.95)
        plt.savefig(f'input_vs_output&prediction_id_{self.real_idx}_sinconv.png')


    def plot_signal_prediction_bar_plot(self,idx):
        fig, axs = plt.subplots(7, 3, figsize=(15, 15), sharex=True)
        axs_f = axs.flatten()
        for i in range(21):
            # Configuración de las barras
            bar_width = 0.4
            # Mis detecciones reales
            detecciones_reales = self.y_t[idx,:,i]
            # Mis detecciones predichas
            detecciones_predichas = self.y_hat_t[idx,:,i]
            bar_positions = np.arange(len(detecciones_reales))
            # Dibujar las barras de detecciones reales en rojo
            axs_f[i].bar(bar_positions - bar_width/2, detecciones_reales, width=bar_width, color='red', alpha=0.7, label='Detecciones reales' if i==0 else "")
            # Dibujar las barras de detecciones predichas en verde
            axs_f[i].bar(bar_positions + bar_width/2, detecciones_predichas, width=bar_width, color='green', alpha=0.7, label='Detecciones predichas' if i==0 else "")      
            axs_f[i].set_title(self.names[i])  # Asignar el nombre como título del subgráfico

        # Colocamos la leyenda en la parte inferior
        fig.legend(loc='lower center', ncol=3, fontsize = 15, bbox_to_anchor=(0.5, 0.03))
        # Colocamos el título en la parte superior
        fig.suptitle(f'Individuo {self.real_idx} - EEG s/ convulsión - Comparación de Detecciones Reales y Predichas por Canal', fontsize=20, y=0.95)
        plt.savefig(f'input_vs_output&prediction_bar_plot_id_{self.real_idx}_sinconv.png')



    def calculate_mi(self, threshold):
        secs, samples, channels = self.mi_input.shape
        print('secs, samples, channels', secs, samples, channels)
        mutual_info = np.zeros((secs, channels))
        #print dtype self.mi_input and self.mi_residuals
        print('dtype mi_input', self.mi_input.dtype)
        print('dtype mi_residuals', self.mi_residuals.dtype)
        for sec in tqdm.tqdm(range(secs)):
            for channel in range(channels):
                x = self.mi_input[sec,:,channel]
                y = self.mi_residuals[sec,:,channel]
                x = x[:, np.newaxis]
                y = y[:, np.newaxis]
                assert x.shape == y.shape == (100, 1)
                post_emi, post_size, prev_emi, prev_size = estimate_mi(x=x, y=y)
                # calculo de la información mutua
                #post_emi, post_size, prev_emi, prev_size = estimate_mi(x=self.mi_input[sec,:,channel], y=self.mi_residuals[sec,:,channel])
                # se guarda en el diccionario    
                mutual_info[sec, channel] = post_emi

        self.mutual_info = mutual_info
        self.mi_mean = mutual_info.mean(axis=1)
        print('Mutual info shape', mutual_info.shape)
        print('Mutual Information:', self.mutual_info)

    def plot_mean_mi_and_label(self):
        fig, axs = plt.subplots(2, 1, figsize=(20, 5))
        axs[0].plot(self.mi_mean)
        axs[0].set_title(f'Mean MI - Individuo {self.real_idx} - EGG s/ Convulsiones')
        axs[1].plot(self.labels)
        axs[1].set_title(f'Labels - Individuo {self.real_idx} - EGG s/ Convulsiones')
        plt.subplots_adjust(hspace = 0.5)  # Ajusta el espacio entre los subplots
        plt.savefig(f'mean_mi_and_labels_id_{self.real_idx}_sinconv.png')


    def detection_threshold(self, threshold=0.5):
        detection_predictions = np.zeros_like(self.mi_mean)
        detection_predictions[self.mi_mean > threshold] = 1
        # Calculo de la accuracy
        accuracy = np.sum(detection_predictions == self.labels) / len(self.labels)
        # IoU
        intersection = np.sum(detection_predictions * self.labels) 
        union = np.sum(detection_predictions + self.labels) - intersection
        iou = intersection / union
        return accuracy, iou
    
    def roc_curve(self):
        # Aplicar todos los thresholds
        thresholds = np.linspace(0, 1, 500)
        accuracies = np.zeros_like(thresholds)
        ious = np.zeros_like(thresholds)
        for i, threshold in enumerate(thresholds):
            accuracies[i], ious[i] = self.detection_threshold(threshold)
        # Plotear
        fig, axs = plt.subplots()
        axs.plot(thresholds, accuracies, label='Accuracy', color='red')
        axs.plot(thresholds, ious, label='IoU', color='blue')
        axs.legend()
        axs.set_title(f'ROC curve - Individuo {self.real_idx} - EGG s/ Convulsiones')
        fig.savefig(f'roc_curve_id_{self.real_idx}_convulsion.png')
        
        # Retornar el mejor threshold de accurac, iou, o la suma de ambos
        acc_idx, iou_idx, acc_iou_idx = np.argmax(accuracies), np.argmax(ious), np.argmax(accuracies+ious)
        acc_t, iou_t, acc_iou_t = thresholds[acc_idx], thresholds[iou_idx], thresholds[acc_iou_idx]

        print('Best thresholds: Acc, IoU, Acc+IoU', acc_t, iou_t, acc_iou_t)
        print('Best values: Acc, IoU, Acc+IoU w acc_th', accuracies[acc_idx], ious[iou_idx], accuracies[acc_iou_idx]+ious[acc_iou_idx])
        print('Best values: Acc, IoU, Acc+IoU w iou_th', accuracies[acc_idx], ious[iou_idx], accuracies[acc_iou_idx]+ious[acc_iou_idx])
        print('Best values: Acc, IoU, Acc+IoU w acc_iou_th', accuracies[acc_idx], ious[iou_idx], accuracies[acc_iou_idx]+ious[acc_iou_idx])
        return acc_t, iou_t, acc_iou_t

def main():
    best_model_path = os.path.join(os.getcwd(), 'modd/tst_logs/transformer/version_0/checkpoints/')
    try:
        best_model_path += [f for f in os.listdir(best_model_path) if '.ckpt' in f][0]
        model = TransformerModel.load_from_checkpoint(best_model_path); print('Modelo cargado desde: ', best_model_path)
    except:
        print('Modelo no encontrado')
        return
    
    # Cantidad de parámetros
    print('Cantidad de parámetros: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    sample_idx = 0
    nominal = True
    time = 3000
    threshold = 0.5

    EMI = MI_estimator(model)
    EMI.filename(sample_idx, nominal)
    EMI.get_chunked_sample_wlabel(sample_idx, in_size=100, step=100, nominal=nominal)
    EMI.get_prediction(batch_size=512)
    EMI.plot_signal_prediction(time)
    EMI.calculate_mi(threshold)
    EMI.plot_mean_mi_and_label()
    EMI.roc_curve()
    EMI.plot_signal_prediction_bar_plot(time)
    
if __name__ == '__main__':
    main()
    #pass