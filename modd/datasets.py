import pandas as pd, numpy as np, matplotlib.pyplot as plt
import mne, re, os, warnings
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torch import Generator, from_numpy
import pytorch_lightning as pl
import tqdm, joblib
from sklearn.preprocessing import RobustScaler
import pickle

data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

class EegDatasetBase(Dataset):
    def __init__(self, transform=None, window_size=100, step=25):
        super().__init__()
        self.transform = transform
        self.labels = pd.read_csv(os.path.join(data_path, 'labels.csv'))
        self.window_size = window_size
        self.step = step
        
    def __len__(self):
        return len(self.labels.columns)
    
    def item(self, idx):
        # raw_file = f'../data/eeg{idx+1}.edf'
        raw_file = os.path.join(data_path, f'eeg{idx+1}.edf')
        with warnings.catch_warnings(): # Ignore warnings
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_edf(raw_file, preload=True, verbose=False)

        # --- FILTRADO ---
        notch_frequencies = [50, 100]
        raw.notch_filter(freqs = notch_frequencies, verbose = False)
        raw.filter(1, 50, fir_design = 'firwin', verbose = False)
        raw.resample(sfreq = 100)
        # ----------------
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
        return sample[0], label
        # segments = np.array([sample[:, i:i + self.window_size] for i in range(0, sample.shape[1] - self.window_size + 1, self.step)])
        # print("segments.shape2", segments.shape)
        # return segments, label
    
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
                axs[-1].axvline(x=self.window_size * i, color='red', linewidth=0.1)
                axs[-1].axvline(x=self.window_size * (i + 1), color='red', linewidth=0.1)

        axs[-1].set_xticks(np.arange(0, len(sample[0]), self.window_size * 60 * 10))
        axs[-1].set_xticklabels(np.arange(0, len(sample[0]) // (self.window_size * 60) + 1, 10))

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
    
class EegDatasetConvulsions(EegDatasetBase):
    def __init__(self, transform=None):
        super().__init__(transform)
        # Dejar solo las columnas donde exista al menos un 1
        self.columns = self.labels.loc[:, (self.labels == 1).any(axis=0)].columns
        
    def my_item(self, idx): # Cambio de nombre por problemas del hook con la clase base
        idx = int(self.columns[idx])-1
        return super().item(idx)
    
    def __getitem__(self, idx):
        idx = int(self.columns[idx])-1
        return super().__getitem__(idx)
    
    def __len__(self):
        return len(self.columns)

class TransformerDataset(Dataset):
    # Chunk de datos de tamaño in_size, con step de tamaño step
    def __init__(self, data, in_size=256, out_size=64, step=64, type='train'):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.step = step
        self.data = data # (N, 21, len_data)
        self.type = type
        self.X = None
        self.Y = None
        self.pwd = os.path.dirname(os.path.abspath(__file__))
        
    def fit_scaler(self):
        scaler = RobustScaler()
        min_len = np.min([sample.shape[1] for sample, _ in self.data])
        data = np.concatenate([sample[:, :min_len] for sample, _ in self.data], axis=1)
        scaler.fit(data.T)
        joblib.dump(scaler, os.path.join(self.pwd, 'scaler.pkl'))
        
    def setup(self, force = False, path = 'data'):
        # Data path
        self.data_path = os.path.join(self.pwd, path)
        # Crear carpeta data si no existe
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        self.len = 0
        # Si se fuerza o el data_path esta vacio o no existe
        if force or not os.path.exists(self.data_path) or len(os.listdir(os.path.join(self.data_path, self.type))) == 0:
            # Borrar datos en data/{train, val, test}
            if os.path.exists(os.path.join(self.data_path, self.type)):
                for f in os.listdir(os.path.join(self.data_path, self.type)):
                    os.remove(os.path.join(self.data_path, self.type, f))
            
            if not os.path.exists(os.path.join(self.data_path, self.type)):
                os.mkdir(os.path.join(self.data_path, self.type))
                
            scaler = joblib.load(os.path.join(self.pwd, 'scaler.pkl'))
            idx = 0
            for sample, _ in tqdm.tqdm(self.data):
                sample = scaler.transform(sample.T).T # Escaler permite pre filtrar outliers
                for i in range(0, sample.shape[1] - self.in_size - self.out_size + 1, self.step):
                    sx = sample[:, i:i + self.in_size].astype('float32')
                    sy = sample[:, i + self.in_size:i + self.in_size + self.out_size].astype('float32')
                    if sx.var(axis=0).mean() < 0.01: continue
                    s = np.concatenate([sx, sy], axis=1)
                    mean = np.mean(s, axis=1)
                    std = np.std(s, axis=1)
                    sx = from_numpy( (sx - mean[:, None]) / std[:, None] ).float().T
                    sy = from_numpy( (sy - mean[:, None]) / std[:, None] ).float().T
                    # Guardar en un pickle (en una carpeta data/{train, val, test})
                    with open(os.path.join(self.data_path, self.type, f'{idx}.pkl'), 'wb') as file:
                        pickle.dump((sx, sy), file)
                    self.len += 1
                    idx += 1
        else:
            self.len = len(os.listdir(os.path.join(self.data_path, self.type)))
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        # Cargar el pickle
        with open(os.path.join(self.data_path, self.type, f'{idx}.pkl'), 'rb') as file:
            sx, sy = pickle.load(file)
        return sx, sy
        
    
class EegDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, in_size=256, out_size=64, step=64, partition=[25, 4, 4]):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.in_size = in_size
        self.out_size = out_size
        self.step = step
        self.pwd = os.path.dirname(os.path.abspath(__file__))
        self.partition = partition
        
    def setup(self, stage: str = None, force = False, force_scaler=False, path = 'data'):
        transform =  transforms.Compose([transforms.ToTensor()])
        data = EegDatasetNominal(transform)
        train_data, val_data, test_data = random_split(data, self.partition, generator=Generator().manual_seed(42))
        # print('Train:', len(train_data), 'Val:', len(val_data), 'Test:', len(test_data))
        
        self.train_dataset = TransformerDataset(train_data, self.in_size, self.out_size, self.step, 'train')
        if not os.path.exists(os.path.join(self.pwd, 'scaler.pkl')) or force_scaler: self.train_dataset.fit_scaler()
        self.train_dataset.setup(force, path);
        # print('Train chunks:', len(self.train_dataset))
        
        self.val_dataset = TransformerDataset(val_data, self.in_size, self.out_size, self.step, 'val')
        self.val_dataset.setup(force, path)
        # print('Val chunks:', len(self.val_dataset))
        
        self.test_dataset = TransformerDataset(test_data, self.in_size, self.out_size, self.step, 'test')
        self.test_dataset.setup(force, path)
        # print('Test chunks:', len(self.test_dataset))
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
    
if __name__ == '__main__':
    data = EegDataModule(batch_size=32, in_size=100, out_size=1, step=1, partition=[25, 4, 4])
    # # Force para recalcular los datasets y el scaler
    data.setup(force=True, force_scaler=False)
    # Graficar un ejemplo
    example = data.train_dataset[0]
    print(example[0].shape, example[1].shape)
    fig, axs = plt.subplots(7, 3, figsize=(15, 15), sharex=True, sharey=True)
    flat_axs = axs.flatten()
    input = example[0].numpy()
    output = example[1].numpy()
    mean = np.mean(np.concatenate([input, output]), axis=0)
    std = np.std(np.concatenate([input, output]), axis=0)
    normalized = (np.concatenate([input, output]) - mean) / std
    input = normalized[:100]
    output = normalized[100:]
    
    for i in range(21):
        flat_axs[i].plot(np.arange(0, 100), input[:, i], label='Input')
        flat_axs[i].plot(np.arange(100, 125), output[:, i], label='Target')
        flat_axs[i].set_ylim([-3, 3])
        flat_axs[i].legend(fontsize=6)
        flat_axs[i].set_title(f'Channel {i}', fontsize=6)
    plt.show()
    
    
    
    
    
    
    # # Revisar rango de cada canal en el dataset de entrenamiento
    # transform =  transforms.Compose([transforms.ToTensor()])
    # data_nominal = EegDatasetNominal(transform)
    # min_len = np.min([sample.shape[1] for sample, _ in data_nominal]) 
    
    # data_scaled = []
    # diffs = []
    # for sample, _ in data_nominal:
    #     # from torch to numpy
    #     sample = sample.numpy()
    #     # Borrar columnas donde exista un 0
    #     print("Cantidad de columnas borradas:", np.sum((sample == 0).all(axis=0)))
    #     # Encontrar segmentos constantes
        
    #     sample = sample[:, (sample != 0).any(axis=0)]
    #     mean = np.mean(sample, axis=1)
    #     std = np.std(sample, axis=1)
    #     diff = np.diff(sample, axis=1)
    #     diffs.append(diff)
    #     normalized = (sample - mean[:, None]) / std[:, None]
    #     normalized = normalized[:, :min_len]
    #     data_scaled.append(normalized)
    # diffs = np.concatenate(diffs, axis=0)
    # data = np.stack(data_scaled, axis=0)
    # assert data.shape == (33, 21, min_len)
    # # Graficar los 21 canales con su rango de valores
    # mean_max_min = False
    # if mean_max_min:
    #     min_curve = np.min(data, axis=0)
    #     max_curve = np.max(data, axis=0)
    #     mean = np.mean(data, axis=0)
    #     fig, axs = plt.subplots(7, 3, figsize=(15, 15), sharex=True, sharey=True)
    #     flat_axs = axs.flatten()
    #     for i in range(21):
    #         flat_axs[i].plot(np.arange(0, min_len), mean[i, :], label='Mean')
    #         flat_axs[i].plot(np.arange(0, min_len), min_curve[i, :], label='Min')
    #         flat_axs[i].plot(np.arange(0, min_len), max_curve[i, :], label='Max')
    #         flat_axs[i].set_ylim([-3, 3])
    #         flat_axs[i].legend(fontsize=6)
    #         flat_axs[i].set_title(f'Channel {i}', fontsize=6)
    #     plt.show()
    # else:
    #     # example
    #     fig, axs = plt.subplots(7, 3, figsize=(15, 15), sharex=True, sharey=True)
    #     flat_axs = axs.flatten()
    #     for i in range(21):
    #         # flat_axs[i].plot(np.arange(0, min_len), data[20, i, :], label='Mean')
    #         # Graficar solo diferencias de cada canal para 1 ejemplo
    #         flat_axs[i].plot(np.arange(0, len(diffs[20, i, :])), diffs[20, i, :], label='Diff')
    #         flat_axs[i].set_ylim([-3, 3])
    #         flat_axs[i].legend(fontsize=6)
    #         flat_axs[i].set_title(f'Channel {i}', fontsize=6)
    #     plt.show()