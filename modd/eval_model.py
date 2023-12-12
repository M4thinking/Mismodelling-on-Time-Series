from transformer import TransformerModel
from datasets import EegDataModule
import torch, os, tqdm
import matplotlib.pyplot as plt
import numpy as np
    

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(version=0, in_size=256, out_size=64, step=64):
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
    
    
    # Calcular metricas de rmse en el conjunto de entrenamiento, validacion y test
    def metrics(y_hat, y):
        mse = ((y_hat - y)**2).mean()
        mae = np.abs(y_hat - y).mean()
        mape = np.abs(y_hat - y).mean() / np.abs(y).mean()
        rmse = np.sqrt(mse)
        y_bar = y.mean()
        r2 = 1 - ((y_hat - y)**2).sum() / ((y - y_bar)**2).sum()
        print('MSE:', mse, 'MAE:', mae, 'MAPE:', mape, 'RMSE:', rmse, 'R2:', r2)
        return mse, mae, mape, rmse, r2
    
    # Entrenamiento
    model.eval()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    # Train
    y_hat = []
    y = []
    for x, y_ in tqdm.tqdm(train_loader):
        y_hat.append(model(x.to(device)).cpu().detach().numpy())
        y.append(y_.cpu().detach().numpy())
    y_hat = np.concatenate(y_hat, axis=0)
    y = np.concatenate(y, axis=0)
    print('Train metrics:')
    metrics(y_hat, y)
    
    # Val
    y_hat = []
    y = []
    for x, y_ in tqdm.tqdm(val_loader):
        y_hat.append(model(x.to(device)).cpu().detach().numpy())
        y.append(y_.cpu().detach().numpy())
    y_hat = np.concatenate(y_hat, axis=0)
    y = np.concatenate(y, axis=0)
    print('Val metrics:')
    metrics(y_hat, y)

    # Test
    y_hat = []
    y = []
    for x, y_ in tqdm.tqdm(test_loader):
        y_hat.append(model(x.to(device)).cpu().detach().numpy())
        y.append(y_.cpu().detach().numpy())
    y_hat = np.concatenate(y_hat, axis=0)
    y = np.concatenate(y, axis=0)
    print('Test metrics:')
    metrics(y_hat, y)
    
    
    x, y = dm.val_dataset[1]
    x = x.unsqueeze(0).to(device)
    x_in = x.clone()
    y = y.unsqueeze(0).to(device)
    print('Input size:', x.shape)
    print('Target size:', y.shape)
    
    model.eval()

    # Inferir, obtener la primera predicción, y concatenarla a la entrada para la siguiente predicción
    y_hat =  torch.zeros((1, out_size, 21)).to(device)
    for i in range(out_size):
        y_hat[:, i, :] = model(x_in)[:, 0, :]
        print(x_in[:, 1:, :].shape, 'device:', x_in[:, 1:, :].device)
        x_in = torch.cat((x_in[:, 1:, :], y_hat[:, i, :].unsqueeze(1)), dim=1)
    y_hat = torch.tensor(y_hat).to(device)

    # Graficar y y_hat
    _ , axs = plt.subplots(7, 3, figsize=(15, 15), sharex=True, sharey=True)
    flat_axs = axs.flatten()
    for i in range(21):
        flat_axs[i].plot(np.arange(0, in_size), x[0, :, i].cpu().detach().numpy(), label='Input')
        flat_axs[i].plot(np.arange(in_size, in_size + out_size), y[0, :, i].cpu().detach().numpy(), label='Target')
        flat_axs[i].plot(np.arange(in_size, in_size + out_size), y_hat[0, :, i].cpu().detach().numpy(), label='Output')
        flat_axs[i].legend(fontsize=6)
        flat_axs[i].set_title(f'Channel {i}', fontsize=6)
    plt.show()


def plot_hist_var(in_size=256, out_size=64, step=64):
    dm = EegDataModule(batch_size=256, in_size=in_size, out_size=out_size, step=step)
    dm.setup(force=False, force_scaler=False)
    train_dataset = dm.train_dataset
    
    # Varianza de los datos (N, L, C) -> (N,)
    var = []
    for x, y in tqdm.tqdm(train_dataset):
        var.append(y.var(axis=0).mean())
    var = np.array(var)
    var = var[var < 0.1] # 0.05 x
    # Grafico de histograma de varianza
    plt.hist(var, bins=100)
    plt.show()

if __name__ == '__main__':
    main(version=2, in_size=100, out_size=25, step=25)
    # plot_hist_var(in_size=100, out_size=25, step=25)