from transformer import TransformerModel
from datasets import EegDataModule
from torch import nn
import torch, os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(version=0):
    # Cargar datos
    dm = EegDataModule(batch_size=64, in_size=100, out_size=25, step=25)
    # Entrenar
    best_model_path = f'tst_logs/transformer/version_{version}/checkpoints/'
    hyperparams = {'d_model': 40,          # Dimensión de los embeddings
                    'nhead': 4,             # Número de cabezas de atención en paralelo
                    'num_layers': 5,        # Número de capas de la red
                    'dim_feedforward': 128, # Dimensión de la capa que sigue a la atención
                    'dropout': 0.2,         # Dropout
                    'learning_rate': 1e-4,  # Tasa de aprendizaje
                    'max_length': 100,      # Tamaño máximo de la secuencia de entrada
                    'reduction': None,         # Reducción de la dimensión de la salida Entrada/reduction
                    }
    
    if not os.path.exists(best_model_path):
        best_model_path = None
        # loss_fn = nn.L1Loss(reduction='sum') # MAE
        model = TransformerModel(**hyperparams, loss_fn=nn.L1Loss()).to(device); print('Modelo creado')
    else:
        best_model_path += [f for f in os.listdir(best_model_path) if '.ckpt' in f][0]
        model = TransformerModel.load_from_checkpoint(best_model_path); print('Modelo cargado desde: ', best_model_path)
    
    # Cantidad de parámetros
    print('Cantidad de parámetros: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    # Hiperparámetros
    print('Hiperparámetros: \n', model.hparams)
    
    # Logger
    logger = TensorBoardLogger('tst_logs', name='transformer')
    # Callbacks
    callbacks = [
        ModelCheckpoint(monitor='val/loss_epoch', mode='min', save_top_k=1),
        EarlyStopping(monitor='val/loss_epoch', patience=20, mode='min'),
        LearningRateMonitor(logging_interval='step')
    ]
    # Entrenar o cargar modelo
    trainer = pl.Trainer(max_epochs=1, logger=logger, callbacks=callbacks, accelerator='gpu', log_every_n_steps=1)
    print('Entrenando...'); trainer.fit(model, dm, ckpt_path=best_model_path)
    
        
if __name__ == '__main__':
    main(version=6) # V0/1 es 256/64, V2 es 100/25, V3 es 100/25 tamañao medio, V4 es 100/25 tamaño ultra pequeño