import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import math

torch.set_float32_matmul_precision('high')

class Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1, max_length=256, channels=21, reduction=None, teacher_forcing=True):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(channels, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers, norm=nn.LayerNorm(d_model))
        self.positional_encoding = self.generate_positional_encoding(d_model, max_length)
        self.decoder = nn.Linear(d_model, channels)
        self.cnn = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2, stride=2), # //2
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2, stride=2), # //2
            nn.ReLU(),
        )
        # Cambiar largo de secuencia de salida (Entrada/reduction) (ej: 256/4 = 64)
        self.avgpool = None if reduction is None else nn.AvgPool1d(reduction, stride=reduction)
        self.init_weights()
        self.teacher_forcing = teacher_forcing
        
    def init_weights(self):
        # Inicializar pesos de la capa de embedding
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        # Inicializar pesos de la capa de convolución para relu
        for m in self.cnn.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x : (batch_size, seq_len, d_model), pos_enc: (1, seq_len, d_model), mask: (batch_size, seq_len, seq_len)
        b, s, c = x.shape # batch_size, seq_len, channels
        x = x.transpose(1, 2) # (batch_size, channels, seq_len)
        x = self.cnn(x)
        x = x.transpose(1, 2) # (batch_size, seq_len, channels)
        x = self.embedding(x) # (batch_size, reduced_seq_len, d_model)
        x = x + self.positional_encoding[:x.size(1), :].unsqueeze(0).to(x.device)
        causal_mask = self.generate_causal_mask(s).to(x.device)
        x = self.transformer_encoder(src=x,is_causal = True, src_key_padding_mask=None, mask=causal_mask)
        x = self.decoder(x)
        if self.avgpool is not None:
            x = self.avgpool(x.transpose(1, 2)).transpose(1, 2)
            x = x.reshape(b, -1, c)
        return x

    def generate_causal_mask(self, seq_len):
        return nn.Transformer().generate_square_subsequent_mask(seq_len)
        
    def generate_positional_encoding(self, d_model, max_length): # Attention is all you need
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) # ej: 0::2 -> 0, 2, 4, 6, 8, ...
        pe[:, 1::2] = torch.cos(position * div_term) # ej: 1::2 -> 1, 3, 5, 7, 9, ...
        return pe

class TransformerModel(pl.LightningModule):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1, learning_rate=1e-3, max_length=256, reduction = None, loss_fn=nn.MSELoss()):
        super(TransformerModel, self).__init__()
        # self.hparams.update(hparams)
        self.save_hyperparameters(ignore=['loss_fn'])
        self.model = Transformer(d_model, nhead, num_layers, dim_feedforward, dropout, max_length=max_length, reduction=reduction)
        self.learning_rate = learning_rate  # Agrega el hiperparámetro learning_rate
        self.loss_function = loss_fn

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # Teacher forcing
        # if self.model.teacher_forcing:
            # for _ in range(y.shape[1]):
            #     y_hat = self.forward(x)[:, 0, :]
            #     x = torch.cat((x[:, 1:, :], y_hat.unsqueeze(1)), dim=1)
            # output = x[:, -y.shape[1]:, :]
        # else:
        output = self.forward(x)
        loss = self.loss_function(output, y)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = self.loss_function(output, y)
        self.log('val/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
        
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-09, betas=(0.9, 0.98))
    
    # Scheduler
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        
        # # warm up lr similar to torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)
        # if self.trainer.global_step < 500:
        #     lr_scale = min(1., float(self.trainer.global_step + 1) / 500.)
        #     for pg in optimizer.param_groups:
        #         pg['lr'] = lr_scale * self.learning_rate
        # # update params
        len_dataloader = 1842
        lr = self.hparams.learning_rate
        step = epoch * len_dataloader + batch_idx + 1
        warmup_steps = 4000
        lr = self.hparams.d_model**-0.5 * min(step**-0.5, step * warmup_steps**-1.5)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step(closure=optimizer_closure)
    
    def predict_future(self, x, n_steps):
        # Predecir recursivamente n_steps pasos
        for _ in range(n_steps):
            x = self.forward(x)
        return x
    
    
if __name__ == '__main__':
    # Test entrenamiento con logger
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning.callbacks import LearningRateMonitor
    from torch.utils.data import DataLoader, Dataset
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    import os
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # First check if model exists in tb_logs/transformer/...
    sample_rate = 100
    hyperparams = {'d_model': 32,
                   'nhead': 4,
                   'num_layers': 2,
                   'dim_feedforward': 64,
                   'dropout': 0.1,
                   'learning_rate': 1e-3,
                   'max_length': sample_rate,
                   'reduction': None,
                   }
    
    best_model_path = 'tst_logs/transformer/version_2/checkpoints/'
    if not os.path.exists(best_model_path):
        model = TransformerModel(**hyperparams).to(device)
        print('Modelo creado')
    else:
        ckpt = [f for f in os.listdir(best_model_path) if '.ckpt' in f][0]
        best_model_path += ckpt
        model = TransformerModel.load_from_checkpoint(best_model_path)
        print('Modelo cargado desde: ', best_model_path)
    
    # Test
    x = torch.randn(32, sample_rate, 21).to(device)
    y = model(x)
    print("Output size:", y.shape)

    
    # Data sintética
    time        = np.arange(0, sample_rate*96, 0.1)
    amplitude   = np.sin(time) + np.sin(time*0.05) +np.sin(time*0.12) *np.random.normal(-0.2, 0.2, len(time))
    scaler = MinMaxScaler(feature_range=(-1, 1)) 
    amplitude = scaler.fit_transform(amplitude.reshape(-1, 1)).reshape(-1)
    # Repetir 21 veces para simular 21 canales
    data_21_channels = np.tile(amplitude, (21,1)).T
    # Primeros 100*sample_rate datos para train, resto para val
    train_data = data_21_channels[:64*9*sample_rate]
    val_data = data_21_channels[64*9*sample_rate:]
    # Chunks de sample_rate datos
    train_data = np.array(np.split(train_data, len(train_data)/sample_rate))
    val_data = np.array(np.split(val_data, len(val_data)/sample_rate))

    # Dataset is same x and y [batch_size, seq_len, channels]
    class Dataset(Dataset):
        def __init__(self, x):
            self.x = torch.tensor(x, dtype=torch.float32)
        def __len__(self):
            return len(self.x)-1
        def __getitem__(self, idx):
            # y es el siguiente de x, pero solo el primer cuarto de los datos
            return self.x[idx], self.x[idx+1, :sample_rate//4]
        
    train_dataset = Dataset(train_data)
    val_dataset = Dataset(val_data)
    # Dimensiones de los datos
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Every 5 steps log the training loss
    logger = TensorBoardLogger('tst_logs', name='transformer', log_graph=True)
    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss',
        save_top_k=1,
    )
    early_stop_callback = EarlyStopping(
        monitor='val/loss',
        patience=10,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # Entrenar desde cero si no existe el checkpoint
    if not os.path.exists(best_model_path):
        trainer = pl.Trainer(max_epochs=2, logger=logger, callbacks=[lr_monitor, checkpoint_callback, early_stop_callback], accelerator='gpu', log_every_n_steps=6)
        trainer.fit(model, train_loader, val_loader)
    else:
        trainer = pl.Trainer(max_epochs=4, logger=logger, callbacks=[lr_monitor, checkpoint_callback, early_stop_callback], accelerator='gpu', log_every_n_steps=6)
        trainer.fit(model, train_loader, val_loader, ckpt_path=best_model_path)