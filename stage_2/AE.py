# AE.py

import torch
import torch.nn as nn
from KAN_MoE import KANMoELayer

# <<< THAY ĐỔI 1 >>>: Thêm một khối CNN để xử lý thông tin thời gian
class TemporalBlock(nn.Module):
    """
    Khối Tích chập 1D để học các mẫu hình thời gian từ chuỗi đặc trưng.
    """
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        # Dùng padding='same' để giữ nguyên độ dài chuỗi
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.relu = nn.ReLU()

    def forward(self, x):
        # Input x shape: [batch_size, sequence_length, features]
        # Conv1d yêu cầu shape: [batch_size, features, sequence_length]
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.relu(x)
        # Chuyển về shape ban đầu: [batch_size, sequence_length, features]
        x = x.permute(0, 2, 1)
        return x

class AnomalyEncoder(nn.Module):
    def __init__(self, input_dim=1, kan_out_features=32, num_experts=8, dropout_rate=0.2):
        super().__init__()
        self.trend_learner = KANMoELayer(in_features=input_dim, out_features=kan_out_features, num_experts=num_experts)
        self.detail_learner = KANMoELayer(in_features=input_dim, out_features=kan_out_features, num_experts=num_experts)
        
        # <<< THAY ĐỔI 2 >>>: Thêm khối TemporalBlock vào Encoder
        # Đầu vào là kan_out_features * 2, đầu ra có thể giữ nguyên hoặc giảm chiều
        self.temporal_processor = TemporalBlock(in_channels=kan_out_features * 2, out_channels=kan_out_features * 2)
        
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, a: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        learned_a_features = self.trend_learner(a)
        learned_d_features = self.detail_learner(d)
        
        combined_features = torch.cat([learned_a_features, learned_d_features], dim=-1)
        
        # Cho các đặc trưng đi qua khối CNN để học phụ thuộc thời gian
        temporal_features = self.temporal_processor(combined_features)
        
        latent_vector = self.dropout(temporal_features)
        return latent_vector

# <<< THAY ĐỔI 3 >>>: Thêm một khối CNN chuyển vị để giải nén
class TemporalUpsampleBlock(nn.Module):
    """
    Khối Tích chập chuyển vị 1D để giải nén đặc trưng thời gian.
    """
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        # Dùng ConvTranspose1d để thực hiện thao tác ngược lại với Conv1d
        self.conv_transpose1d = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_transpose1d(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        return x

class AnomalyDecoder(nn.Module):
    def __init__(self, latent_dim=64, output_dim=1, num_experts=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.reconstructor_in_features = latent_dim // 2

        # <<< THAY ĐỔI 4 >>>: Thêm khối TemporalUpsampleBlock vào Decoder
        self.temporal_upsampler = TemporalUpsampleBlock(in_channels=latent_dim, out_channels=latent_dim)
        
        self.trend_reconstructor = KANMoELayer(
            in_features=self.reconstructor_in_features,
            out_features=output_dim,
            num_experts=num_experts
        )
        self.detail_reconstructor = KANMoELayer(
            in_features=self.reconstructor_in_features,
            out_features=output_dim,
            num_experts=num_experts
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Cho vector tiềm ẩn đi qua khối giải nén thời gian trước
        upsampled_z = self.temporal_upsampler(z)
        
        # Tách vector đã giải nén
        z_a = upsampled_z[..., :self.reconstructor_in_features]
        z_d = upsampled_z[..., self.reconstructor_in_features:]
        
        reconstructed_a = self.trend_reconstructor(z_a)
        reconstructed_d = self.detail_reconstructor(z_d)
        
        return reconstructed_a, reconstructed_d

class TimeSeriesAutoencoder(nn.Module):
    """
    Mô hình Autoencoder hoàn chỉnh, KẾT HỢP KAN-MoE và CNN.
    """
    def __init__(self, input_dim=1, kan_out_features=32, num_experts=8):
        super().__init__()
        latent_dim = kan_out_features * 2
        
        self.encoder = AnomalyEncoder(
            input_dim=input_dim, 
            kan_out_features=kan_out_features, 
            num_experts=num_experts
        )
        self.decoder = AnomalyDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            num_experts=num_experts
        )

    def forward(self, a: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent_vector = self.encoder(a, d)
        reconstructed_a, reconstructed_d = self.decoder(latent_vector)
        return reconstructed_a, reconstructed_d