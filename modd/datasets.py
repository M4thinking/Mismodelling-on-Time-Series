import pandas as pd, numpy as np, matplotlib.pyplot as plt
import mne, re
from torch.utils.data import Dataset

class EegDatasetBase(Dataset):
    def __init__(self, transform=None, window_size=256, step=256):
        super().__init__()
        self.transform = transform
        self.labels = pd.read_csv('../data/labels.csv')
        self.window_size = window_size
        self.step = step
        
    def __len__(self):
        return len(self.labels.columns)
    
    def item(self, idx):
        raw_file = f'../data/eeg{idx+1}.edf'
        raw = mne.io.read_raw_edf(raw_file, preload=True, verbose=False)
        data = raw.get_data()
        names = map(str.upper, raw.ch_names)
        names = [n + "-REF" if not re.search("-REF", n) else n for n in names]
        sample = data[np.argsort(names), :]
        names = np.sort(names)
        # Labels son la columna idx+1 del csv, un vector de 0s y 1s
        label = self.labels.iloc[:, idx].values.astype('float')
        return sample, label, names, idx+1
    
    def __getitem__(self, idx):
        sample, label, _, _ = self.item(idx)
        if self.transform:
            sample = self.transform(sample)
        segments = np.array([sample[:, i:i + self.window_size] for i in range(0, sample.shape[1] - self.window_size + 1, self.step)])
        return segments, label
    
    def plot(self, idx, save_svg=False):
        sample, label, names, real_idx = self.item(idx)
        detections = label
        fig, axs = plt.subplots(len(names) + 1, 1, figsize=(30, 12))
        
        for ax, name, signal in zip(axs[:-1], names, sample):
            ax.plot(signal, linewidth=0.3)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim([0, len(sample[0])])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylabel(name, rotation=0, labelpad=50, fontsize=11)

        for i, detection in enumerate(detections):
            if detection == 1:
                axs[-1].axvline(x=256 * i, color='red', linewidth=0.1)
                axs[-1].axvline(x=256 * (i + 1), color='red', linewidth=0.1)

        axs[-1].set_xticks(np.arange(0, len(sample[0]), 256 * 60 * 10))
        axs[-1].set_xticklabels(np.arange(0, len(sample[0]) // (256 * 60) + 1, 10))

        axs[10].text(-0.06, 0.2, 'Voltaje ($\mu V$)',
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=axs[10].transAxes,
                     rotation=90,
                     fontsize=17)
        
        axs[-1].set_xlabel('Tiempo (minutos)', fontsize=17)
        axs[-1].set_ylabel('Detecciones', rotation=0, labelpad=50, fontsize=11)
        axs[-1].set_xlim([0, len(sample[0])])
        axs[-1].set_yticks([])

        axs[0].set_title(f'Individuo {real_idx} - 21 Señales EEG y Anotaciones De Convulsiones', fontsize=20)

        if save_svg:
            plt.savefig(f'eeg_{real_idx}.svg', format='svg', dpi=600, bbox_inches='tight', pad_inches=0.1)
        plt.show()

class EegDataset(EegDatasetBase):
    def __init__(self, transform=None):
        super().__init__(transform)

class EegDatasetNominal(EegDatasetBase):
    def __init__(self, transform=None):
        super().__init__(transform)
        # Eliminar columnas donde exista un 1 (solo dejar si hay 0s o NaNs)
        self.columns = self.labels.loc[:, (self.labels != 1).all(axis=0)].columns
    
    def my_item(self, idx): # Cambio de nombre por problemas del hook con la clase base
        idx = int(self.columns[idx])-1
        return super().item(idx)
    
    def __getitem__(self, idx):
        idx = int(self.columns[idx])-1
        return super().__getitem__(idx)
    
    def __len__(self):
        return len(self.columns)
    
class TransformerDataset(EegDataset):
    def __init__(self, transform=None, window_size=256, step=256):
        super().__init__(transform, window_size, step)
    
    def __getitem__(self, idx):
        segments, label = super().__getitem__(idx)
        # (batch, n_channels, n_steps_in)
        segments = segments.transpose(1, 0, 2)
        # (batch, n_channels, n_steps_out)
        label = label.reshape(1, -1, 1)
        return segments, label