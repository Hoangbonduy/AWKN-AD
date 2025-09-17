import torch
import torch.nn as nn
from .KAN_MoE import KANMoELayer

class TemporalBlock(nn.Module):
    """
    Khối Tích chập 1D (CNN) để học các mẫu hình cục bộ.
    (Giữ nguyên)
    """
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        return x

### THAY ĐỔI BẮT ĐẦU: Đảo ngược vai trò CNN và GRU ###

class AnomalyEncoder(nn.Module):
    """
    Encoder mới:
    - CNN xử lý đặc trưng của trend.
    - GRU xử lý đặc trưng của residual.
    """
    def __init__(self, input_dim=1, kan_out_features=32, gru_hidden_dim=64, num_experts=8, dropout_rate=0.2):
        super().__init__()
        self.trend_learner = KANMoELayer(in_features=input_dim, out_features=kan_out_features, num_experts=num_experts)
        self.detail_learner = KANMoELayer(in_features=input_dim, out_features=kan_out_features, num_experts=num_experts)
        
        # CNN để xử lý chuỗi đặc trưng của trend
        self.temporal_processor = TemporalBlock(
            in_channels=kan_out_features, 
            out_channels=kan_out_features # Giữ nguyên số features
        )
        
        # GRU để xử lý chuỗi đặc trưng của residual (nhiễu)
        self.gru_processor = nn.GRU(
            input_size=kan_out_features, 
            hidden_size=gru_hidden_dim, 
            batch_first=True
        )
        
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, a: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # 1. Học đặc trưng riêng biệt
        learned_a_features = self.trend_learner(a)
        learned_d_features = self.detail_learner(d)
        
        # 2. Xử lý song song
        # Trend features đi qua CNN
        trend_spikes = self.temporal_processor(learned_a_features)
        
        # Residual features đi qua GRU
        residual_context, _ = self.gru_processor(learned_d_features)
        
        # 3. Kết hợp đặc trưng
        combined_features = torch.cat([trend_spikes, residual_context], dim=-1)
        
        latent_vector = self.dropout(combined_features)
        return latent_vector

class TemporalUpsampleBlock(nn.Module):
    """
    Khối giải nén CNN.
    (Giữ nguyên)
    """
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        self.conv_transpose1d = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_transpose1d(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        return x

class AnomalyDecoder(nn.Module):
    """
    Decoder mới:
    - Tách latent vector thành Trend Spikes và Residual Context.
    - Dùng CNN chuyển vị để tái tạo trend.
    - Dùng GRU để tái tạo residual.
    """
    def __init__(self, latent_dim, trend_latent_dim, residual_latent_dim, output_dim=1, num_experts=8):
        super().__init__()
        self.trend_latent_dim = trend_latent_dim
        self.residual_latent_dim = residual_latent_dim

        # Lớp CNN chuyển vị để giải nén Trend Spikes
        self.temporal_upsampler = TemporalUpsampleBlock(
            in_channels=self.trend_latent_dim, 
            out_channels=self.trend_latent_dim
        )
        
        # Lớp GRU để tái tạo từ Residual Context
        self.gru_reconstructor = nn.GRU(
            input_size=self.residual_latent_dim, 
            hidden_size=self.residual_latent_dim,
            batch_first=True
        )
        
        self.trend_reconstructor = KANMoELayer(
            in_features=self.trend_latent_dim,
            out_features=output_dim,
            num_experts=num_experts
        )
        self.detail_reconstructor = KANMoELayer(
            in_features=self.residual_latent_dim,
            out_features=output_dim,
            num_experts=num_experts
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. Tách latent vector
        trend_spikes = z[..., :self.trend_latent_dim]
        residual_context = z[..., self.trend_latent_dim:]
        
        # 2. Tái tạo song song
        upsampled_trend_features = self.temporal_upsampler(trend_spikes)
        reconstructed_residual_features, _ = self.gru_reconstructor(residual_context)
        
        # 3. Tinh chỉnh bằng KANMoE
        reconstructed_a = self.trend_reconstructor(upsampled_trend_features)
        reconstructed_d = self.detail_reconstructor(reconstructed_residual_features)
        
        return reconstructed_a, reconstructed_d

class TimeSeriesAutoencoder(nn.Module):
    """
    Mô hình Autoencoder hoàn chỉnh, kết hợp KAN-MoE, CNN cho trend, và GRU cho residual.
    """
    def __init__(self, input_dim=1, kan_out_features=16, gru_hidden_dim=64, num_experts=8):
        super().__init__()
        # Tính toán các chiều (dimension)
        trend_latent_dim = kan_out_features 
        residual_latent_dim = gru_hidden_dim
        latent_dim = trend_latent_dim + residual_latent_dim
        
        self.encoder = AnomalyEncoder(
            input_dim=input_dim, 
            kan_out_features=kan_out_features, 
            gru_hidden_dim=gru_hidden_dim,
            num_experts=num_experts
        )
        self.decoder = AnomalyDecoder(
            latent_dim=latent_dim,
            trend_latent_dim=trend_latent_dim,
            residual_latent_dim=residual_latent_dim,
            output_dim=input_dim,
            num_experts=num_experts
        )

    def forward(self, a: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent_vector = self.encoder(a, d)
        reconstructed_a, reconstructed_d = self.decoder(latent_vector)
        return reconstructed_a, reconstructed_d