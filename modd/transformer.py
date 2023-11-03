import torch
import torch.nn as nn
import pytorch_lightning as pl

# Modelo autoregresivo (batch, n_channels, n_steps_in) -> (batch, n_channels, n_steps_out)
# - para predecir a n_steps pasos basado en transformer
# - con n_channels=21 de entrada, 21 series de tiempo
# - Largo de ventana de entrada máximo de 256 hz * 60s y mínimo de 256 hz * 1s
# - modelo debe ser causal, es decir, solo puede ver el pasado
class AutoregressiveTransformer(pl.LightningModule):
    def __init__(self, n_channels, n_heads, n_layers, max_win_size=256*60, dropout=0.1):
        super().__init__()
        self.n_channels = n_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_win_size = max_win_size
        self.dropout = dropout
        self.pos_encoder = nn.TransformerEncoderLayer(d_model=n_channels, nhead=n_heads, dropout=dropout, batch_first=True)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=n_channels, nhead=n_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers, norm=nn.LayerNorm(n_channels))
        self.linear = nn.Linear(n_channels, n_channels)
        
    def forward(self, x):
        # x: (batch, n_channels, n_steps_in)
        batch_size, n_channels, n_steps_in = x.size()
        
        x = x.transpose(1, 2) # (batch, n_steps_in, n_channels)
        
        # Asegurarse de que la longitud de la ventana cumple con el máximo
        if n_steps_in > self.max_win_size:
            x = x[:, :, -self.max_win_size:]
        
        # Crear una máscara de relleno
        pad_mask = torch.zeros(batch_size, n_steps_in, dtype=torch.bool)
        
        # Aplicar el Transformer y la máscara de relleno
        x = self.transformer_encoder(x, src_key_padding_mask=pad_mask)
        x = self.linear(x)
        
        # x: (batch, n_steps_in, n_channels)
        x = x.transpose(1, 2)
        return x
    
    def predict_future(self, x, n_steps):
        # Predecir recursivamente n_steps pasos
        for _ in range(n_steps):
            x = self.forward(x)
        return x
    
    def loss(self, y_pred, y_true):
        # Calcular la pérdida de reconstrucción
        return nn.MSELoss()(y_pred, y_true)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def predict(self, x):
        # x: (batch, n_channels, n_steps_in)
        return self.forward(x)
    
    def predict_future(self, x, n_steps):
        # x: (batch, n_channels, n_steps_in)
        return self.predict_future(x, n_steps)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)