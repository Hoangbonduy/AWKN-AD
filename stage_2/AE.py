import torch
import torch.nn as nn
from KAN_MoE import KANMoELayer

class AnomalyEncoder(nn.Module):
    """
    Bộ mã hóa (Encoder) cho chuỗi thời gian.
    
    Nhận đầu vào là chuỗi xu hướng (a) và chuỗi chi tiết (d),
    học và nén chúng thành một vector đặc trưng tiềm ẩn (latent vector).
    """
    def __init__(self, input_dim=1, kan_out_features=32, num_experts=8):
        """
        Khởi tạo Encoder.
        
        Args:
            input_dim (int): Số chiều của feature đầu vào cho mỗi chuỗi (thường là 1).
            kan_out_features (int): Số chiều đặc trưng đầu ra của mỗi mạng KAN-MoE.
            num_experts (int): Số lượng "chuyên gia" spline trong mỗi mạng KAN-MoE.
        """
        super().__init__()
        # Mạng KAN-MoE để học đặc trưng từ chuỗi xu hướng 'a'
        self.trend_learner = KANMoELayer(in_features=input_dim, out_features=kan_out_features, num_experts=num_experts)
        # Mạng KAN-MoE để học đặc trưng từ chuỗi chi tiết 'd'
        self.detail_learner = KANMoELayer(in_features=input_dim, out_features=kan_out_features, num_experts=num_experts)

    def forward(self, a: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # a, d shape: [batch_size, sequence_length, 1]
        
        # Xử lý song song
        learned_a_features = self.trend_learner(a) # -> [batch_size, sequence_length, kan_out_features]
        learned_d_features = self.detail_learner(d) # -> [batch_size, sequence_length, kan_out_features]

        # Ghép các đặc trưng đã học lại với nhau
        # Kích thước cuối cùng là [batch_size, sequence_length, kan_out_features * 2]
        combined_features = torch.cat([learned_a_features, learned_d_features], dim=-1)
        
        return combined_features

class AnomalyDecoder(nn.Module):
    """
    Bộ giải mã (Decoder) cho chuỗi thời gian.
    
    Nhận đầu vào là vector đặc trưng tiềm ẩn, cố gắng tái tạo lại
    chuỗi xu hướng (a) và chuỗi chi tiết (d) ban đầu.
    Sử dụng kiến trúc MLP đơn giản để đảm bảo tốc độ dự đoán (inference) nhanh.
    """
    def __init__(self, latent_dim=64, hidden_dim=128, output_dim=1):
        """
        Khởi tạo Decoder.
        
        Args:
            latent_dim (int): Số chiều của vector tiềm ẩn đầu vào (bằng kan_out_features * 2).
            hidden_dim (int): Số nơ-ron trong các tầng ẩn của MLP.
            output_dim (int): Số chiều đầu ra (thường là 1).
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.trend_reconstructor_dim = latent_dim // 2
        self.detail_reconstructor_dim = latent_dim // 2

        # Mạng MLP để tái tạo chuỗi xu hướng 'a'
        self.trend_reconstructor = nn.Sequential(
            nn.Linear(self.trend_reconstructor_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Mạng MLP để tái tạo chuỗi chi tiết 'd'
        self.detail_reconstructor = nn.Sequential(
            nn.Linear(self.detail_reconstructor_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # z shape: [batch_size, sequence_length, latent_dim]
        
        # Tách vector tiềm ẩn z thành hai phần cho a và d
        z_a = z[..., :self.trend_reconstructor_dim] # Lấy nửa đầu
        z_d = z[..., self.trend_reconstructor_dim:] # Lấy nửa sau

        # Tái tạo lại a và d
        reconstructed_a = self.trend_reconstructor(z_a) # -> [batch_size, sequence_length, 1]
        reconstructed_d = self.detail_reconstructor(z_d) # -> [batch_size, sequence_length, 1]
        
        return reconstructed_a, reconstructed_d

class TimeSeriesAutoencoder(nn.Module):
    """
    Mô hình Autoencoder hoàn chỉnh, kết hợp Encoder và Decoder.
    """
    def __init__(self, input_dim=1, kan_out_features=32, num_experts=8, decoder_hidden_dim=128):
        super().__init__()
        latent_dim = kan_out_features * 2
        
        self.encoder = AnomalyEncoder(
            input_dim=input_dim, 
            kan_out_features=kan_out_features, 
            num_experts=num_experts
        )
        
        self.decoder = AnomalyDecoder(
            latent_dim=latent_dim,
            hidden_dim=decoder_hidden_dim,
            output_dim=input_dim
        )

    def forward(self, a: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # a, d shape: [batch_size, sequence_length, 1]
        
        # 1. Mã hóa: Biến đổi (a, d) thành vector tiềm ẩn z
        latent_vector = self.encoder(a, d)
        
        # 2. Giải mã: Tái tạo (a', d') từ vector tiềm ẩn z
        reconstructed_a, reconstructed_d = self.decoder(latent_vector)
        
        return reconstructed_a, reconstructed_d

# --- ĐOẠN MÃ ĐỂ KIỂM TRA ---
# Bạn có thể chạy trực tiếp file `python AE.py` để kiểm tra
# if __name__ == '__main__':
#     # --- THỬ NGHIỆM KHỞI TẠO VÀ CHẠY MODEL ---
    
#     # Các tham số giả định
#     batch_size = 16      # Xử lý 16 chuỗi thời gian cùng lúc
#     sequence_length = 100 # Mỗi chuỗi dài 100 điểm dữ liệu
#     kan_features = 32    # Mỗi KAN-MoE sẽ tạo ra vector 32 chiều
    
#     # Tạo dữ liệu giả (dummy data)
#     a_input = torch.randn(batch_size, sequence_length, 1)
#     d_input = torch.randn(batch_size, sequence_length, 1)
    
#     print("--- Khởi tạo Mô hình Autoencoder ---")
#     model = TimeSeriesAutoencoder(
#         input_dim=1,
#         kan_out_features=kan_features,
#         num_experts=8,
#         decoder_hidden_dim=128
#     )
#     print(model)
    
#     # Đếm số lượng tham số để biết độ phức tạp của model
#     total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"\nTổng số tham số có thể huấn luyện: {total_params:,}")
    
#     print("\n--- Chạy thử một lượt forward ---")
#     # Cho dữ liệu đi qua mô hình
#     a_reconstructed, d_reconstructed = model(a_input, d_input)
    
#     print(f"Input shape (a):\t\t{a_input.shape}")
#     print(f"Output shape (a_reconstructed): {a_reconstructed.shape}")
#     print("-" * 40)
#     print(f"Input shape (d):\t\t{d_input.shape}")
#     print(f"Output shape (d_reconstructed): {d_reconstructed.shape}")
    
#     # Kiểm tra tính toán hàm loss (sử dụng Huber Loss như đã thảo luận)
#     loss_fn = nn.HuberLoss()
#     loss_a = loss_fn(a_reconstructed, a_input)
#     loss_d = loss_fn(d_reconstructed, d_input)
#     total_loss = loss_a + loss_d
    
#     print(f"\nGiá trị loss ví dụ (Huber Loss): {total_loss.item():.4f}")