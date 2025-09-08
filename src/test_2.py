import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from stage_1.KAMA import kama_decomposition
from AE import TimeSeriesAutoencoder

def visualize_features(features_tensor, place_id):
    """Trực quan hóa features bằng t-SNE"""
    print("\n=== TRỰC QUAN HÓA DỮ LIỆU ===")
            
    # Chuyển tensor thành numpy và reshape để có shape [n_samples, n_features]
    features_np = features_tensor.squeeze(0).detach().numpy()  # [seq_len, feature_dim]
    print(f"Shape của features để t-SNE: {features_np.shape}")
    
    # Áp dụng t-SNE để giảm chiều từ feature_dim về 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features_np)
            
    # Vẽ đồ thị scatter plot
    plt.figure(figsize=(12, 8))
            
    # Tạo colormap theo thời gian
    time_indices = np.arange(len(features_2d))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                                c=time_indices, cmap='viridis', s=50, alpha=0.7)
    plt.colorbar(scatter, label='Time Index')
    plt.title(f't-SNE Visualization of Autoencoder Latent Features\nplaceId: {place_id}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True, alpha=0.3)
            
    # Lưu đồ thị
    plt.savefig(f'tsne_autoencoder_placeId_{place_id}.png', dpi=300, bbox_inches='tight')
    plt.show()
            
    print(f"Đã lưu đồ thị t-SNE vào file: tsne_autoencoder_placeId_{place_id}.png")

if __name__ == "__main__":
    # --- 1. Nạp và Chuẩn bị Dữ liệu ---
    DATA_PATH = 'data/cleaned_data_after_idx30.csv'

    KAN_out_features = 32  # Số đặc trưng đầu ra từ mỗi KANMoELayer
    feature_dim = KAN_out_features * 2  # Tổng số đặc trưng sau

    try:
        df = pd.read_csv(DATA_PATH)
        print(f"Đã đọc thành công file dữ liệu. Tổng số dòng: {len(df)}")
        
        # Lấy dữ liệu của một placeId duy nhất để làm ví dụ
        if not df.empty:
            first_place_id = df['placeId'].iloc[0]
            time_series_np = df[df['placeId'] == first_place_id]['view'].values
            print(f"Đang xử lý placeId: {first_place_id} với chuỗi dài {len(time_series_np)} điểm.")

            # --- 2. Chạy Tầng 1: Phân rã KAMA ---
            print("\nBắt đầu Tầng 1: Phân rã KAMA...")
            a_np, d_np = kama_decomposition(time_series_np)
            print("Hoàn thành Tầng 1. Đã tạo ra chuỗi 'a' và 'd'.")
            print(f"Shape của a: {a_np.shape}, Shape của d: {d_np.shape}")

            # --- 3. Chạy Tầng 2: Autoencoder ---
            print("\nBắt đầu Tầng 2: Autoencoder với KAN-MoE...")
            
            # Chuyển numpy arrays sang PyTorch Tensors với đúng shape [batch, seq_len, in_features]
            a_tensor = torch.FloatTensor(a_np.copy()).unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
            d_tensor = torch.FloatTensor(d_np.copy()).unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
            
            print(f"Shape của a_tensor: {a_tensor.shape}")
            print(f"Shape của d_tensor: {d_tensor.shape}")
            
            # Khởi tạo Autoencoder
            autoencoder = TimeSeriesAutoencoder(
                input_dim=1,
                kan_out_features=KAN_out_features,
                num_experts=8,
                decoder_hidden_dim=128
            )
            
            print(f"Đã khởi tạo Autoencoder với {sum(p.numel() for p in autoencoder.parameters() if p.requires_grad):,} tham số")
            
            # Forward pass qua Autoencoder
            with torch.no_grad():  # Không cần gradient cho inference
                # Lấy latent features từ encoder
                latent_features = autoencoder.encoder(a_tensor, d_tensor)
                
                # Reconstruction (optional - để kiểm tra chất lượng)
                reconstructed_a, reconstructed_d = autoencoder(a_tensor, d_tensor)
            
            print("Hoàn thành Tầng 2.")
            print("\n=== KẾT QUẢ CUỐI CÙNG ===")
            print(f"Shape của latent features: {latent_features.shape}")
            print(f"Shape của reconstructed_a: {reconstructed_a.shape}")
            print(f"Shape của reconstructed_d: {reconstructed_d.shape}")
            print(f"(Giải thích: Batch size=1, Độ dài chuỗi={len(time_series_np)}, Số đặc trưng={feature_dim})")
            
            # --- 4. Tính toán reconstruction loss (để đánh giá chất lượng) ---
            loss_fn = nn.HuberLoss()
            loss_a = loss_fn(reconstructed_a, a_tensor)
            loss_d = loss_fn(reconstructed_d, d_tensor)
            total_loss = loss_a + loss_d
            
            print(f"\nReconstruction Loss:")
            print(f"  - Loss A (trend): {loss_a.item():.6f}")
            print(f"  - Loss D (detail): {loss_d.item():.6f}")
            print(f"  - Total Loss: {total_loss.item():.6f}")
            
            # --- 5. Trực quan hóa latent features ---
            visualize_features(latent_features, first_place_id)

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file tại '{DATA_PATH}'. Vui lòng kiểm tra lại đường dẫn.")
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")