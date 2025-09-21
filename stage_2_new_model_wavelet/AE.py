import torch
import torch.nn as nn
from .KAN_MoE import KANMoELayer

class KAN_Encoder(nn.Module):
    def __init__(self, num_levels, seq_len, latent_dim=32):
        super().__init__()
        # KAN Layer để biến đổi features
        self.kan_layer = KANMoELayer(in_features=num_levels, out_features=latent_dim)
        # Pooling để nén chiều thời gian
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x shape: [batch, num_levels, seq_len]
        
        # 1. Hoán vị để KAN xử lý
        x = x.permute(0, 2, 1) # -> [batch, seq_len, num_levels]
        
        # 2. Qua lớp KAN để biến đổi feature
        x = self.kan_layer(x) # -> [batch, seq_len, latent_dim]
        
        # 3. Hoán vị lại để pooling
        x = x.permute(0, 2, 1) # -> [batch, latent_dim, seq_len]
        
        # 4. Pooling để nén thành vector duy nhất
        latent_vector = self.pooling(x).squeeze(-1) # -> [batch, latent_dim]
        
        return latent_vector

class KAN_Decoder(nn.Module):
    def __init__(self, latent_dim, num_levels, seq_len):
        super().__init__()
        self.seq_len = seq_len
        # KAN Layer để tái tạo features
        self.kan_layer = KANMoELayer(in_features=latent_dim, out_features=num_levels)

    def forward(self, x):
        # x shape: [batch, latent_dim]
        
        # 1. "Kéo dài" vector tiềm ẩn thành một chuỗi
        # unsqueeze để thêm chiều seq_len, expand để lặp lại
        x = x.unsqueeze(1).expand(-1, self.seq_len, -1) # -> [batch, seq_len, latent_dim]
        
        # 2. Qua KAN để tái tạo features
        x = self.kan_layer(x) # -> [batch, seq_len, num_levels]
        
        # 3. Hoán vị về shape ban đầu
        reconstructed_coeffs = x.permute(0, 2, 1) # -> [batch, num_levels, seq_len]
        
        return reconstructed_coeffs

class KAN_Autoencoder(nn.Module):
    def __init__(self, num_levels, seq_len, latent_dim=32):
        super().__init__()
        self.encoder = KAN_Encoder(num_levels, seq_len, latent_dim)
        self.decoder = KAN_Decoder(latent_dim, num_levels, seq_len)
        
    def forward(self, x):
        latent_vector = self.encoder(x)
        reconstructed_output = self.decoder(latent_vector)
        return reconstructed_output

# Alias để tương thích với code khác
TimeSeriesAutoencoder = KAN_Autoencoder