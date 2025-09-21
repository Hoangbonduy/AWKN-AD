import sys
import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stage_2_new_model_wavelet.wavelet import dwt_transform, get_max_wavelet_level
# THAY ĐỔI: Tên file import vẫn là AE, nhưng nội dung bên trong đã là kiến trúc KAN mới
from stage_2_new_model_wavelet.AE import TimeSeriesAutoencoder

class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

# Hàm masked_loss cải tiến cho dynamic padding
def masked_loss(pred, target, loss_fn):
    """
    Tính loss với mask tự động cho dynamic padding
    """
    # Tạo mask tự động: non-zero elements are valid
    mask = (target != 0).float()
    loss = loss_fn(pred, target)
    
    if mask.dim() < loss.dim():
        mask = mask.unsqueeze(-1).expand_as(loss) if loss.dim() > mask.dim() else mask
    
    # Apply mask
    masked_loss_values = loss * mask
    total_loss = masked_loss_values.sum()
    total_valid_elements = mask.sum()
    
    # Tránh chia cho 0
    return total_loss / (total_valid_elements + 1e-8)

if __name__ == "__main__":
    # --- 1. Cấu hình ---
    DATA_PATH = 'data/cleaned_data_no_zero_periods_filtered.csv'  # Sử dụng dataset đầy đủ
    MODEL_SAVE_DIR = 'saved_models_wavelet_kan' # Thư mục mới cho dynamic model
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'wavelet_kan_autoencoder_dynamic.pth')

    WAVELET_TYPE = 'db4'
    BATCH_SIZE = 128
    NUM_EPOCHS = 200
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    LATENT_DIM = 32 # MỚI: Thêm siêu tham số cho KAN Autoencoder

    # --- 2. Nạp và Chuẩn bị Dữ liệu với Full Dataset ---
    df = pd.read_csv(DATA_PATH)
    place_ids = df['placeId'].unique()
    print(f"Đã đọc dữ liệu. Tổng số địa điểm: {len(place_ids)}")
    
    # Sử dụng tất cả địa điểm, không filter
    all_d_coeffs_data, all_lengths = [], []
    print("Bắt đầu xử lý dữ liệu và biến đổi Wavelet...")
    
    skipped_count = 0
    for place_id in place_ids:
        place_data = df[df['placeId'] == place_id]['view'].values
        if len(place_data) < 30:  # Chỉ skip những chuỗi quá ngắn
            skipped_count += 1
            continue
            
        scaler = RobustScaler()
        place_data_scaled = scaler.fit_transform(place_data.reshape(-1, 1)).flatten()
        level = get_max_wavelet_level(len(place_data_scaled), WAVELET_TYPE)
        d_coeffs = dwt_transform(place_data_scaled, wavelet=WAVELET_TYPE, level=level)
        all_d_coeffs_data.append(d_coeffs)
        all_lengths.append(len(place_data_scaled))

    print(f"Đã xử lý {len(all_d_coeffs_data)} địa điểm, skip {skipped_count} địa điểm (quá ngắn)")
    print(f"Độ dài series: min={min(all_lengths)}, max={max(all_lengths)}, mean={np.mean(all_lengths):.1f}")

    # --- 3. Dynamic Padding Dữ liệu ---
    # Tính toán thống kê để thiết lập padding hợp lý
    all_coeff_lengths = [len(c) for coeffs in all_d_coeffs_data for c in coeffs]
    all_num_levels = [len(coeffs) for coeffs in all_d_coeffs_data]
    
    # Sử dụng percentile để tránh outliers ảnh hưởng quá nhiều
    universal_max_len_d = int(np.percentile(all_coeff_lengths, 95))  # 95th percentile thay vì max
    max_num_levels = max(all_num_levels)
    
    print(f"Thống kê coefficient lengths: min={min(all_coeff_lengths)}, max={max(all_coeff_lengths)}, 95th={universal_max_len_d}")
    print(f"Max number of levels: {max_num_levels}")
    
    # Dynamic padding với truncation cho các coefficient quá dài
    padded_d_list = []
    truncated_count = 0
    for coeffs in all_d_coeffs_data:
        padded_coeffs_for_ts = []
        for c in coeffs:
            if len(c) <= universal_max_len_d:
                # Pad nếu ngắn hơn universal max length
                padded_c = np.pad(c, (0, universal_max_len_d - len(c)), 'constant')
            else:
                # Truncate nếu dài hơn universal max length
                padded_c = c[:universal_max_len_d]
                truncated_count += 1
            padded_coeffs_for_ts.append(padded_c)
        
        # Pad levels nếu thiếu
        while len(padded_coeffs_for_ts) < max_num_levels:
            padded_coeffs_for_ts.append(np.zeros(universal_max_len_d))
        
        padded_d_list.append(np.stack(padded_coeffs_for_ts, axis=0))
    
    all_d_tensor = torch.FloatTensor(np.array(padded_d_list))
    print(f"Dữ liệu đã được xử lý. Shape D Coeffs: {all_d_tensor.shape}")
    print(f"Số coefficient bị truncate: {truncated_count} / {len(all_coeff_lengths)} ({truncated_count/len(all_coeff_lengths)*100:.1f}%)")

    # --- 4. Chia Train/Test và tạo DataLoader ---
    # THAY ĐỔI: Không cần mask nữa vì kiến trúc mới không dùng GRU và loss được tính trên toàn bộ
    indices = np.arange(len(all_d_tensor))
    train_indices, test_indices = train_test_split(indices, test_size=0.1, random_state=42)
    train_dataset = TensorDataset(all_d_tensor[train_indices])
    test_dataset = TensorDataset(all_d_tensor[test_indices])
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # --- 5. Khởi tạo Model, Loss, Optimizer ---
    # THAY ĐỔI: Khởi tạo model KAN mới với các tham số tương ứng
    model = TimeSeriesAutoencoder(
        num_levels=max_num_levels,
        seq_len=universal_max_len_d,
        latent_dim=LATENT_DIM
    )
    print(f"Đã khởi tạo model KAN Wavelet AE với {sum(p.numel() for p in model.parameters() if p.requires_grad):,} tham số.")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # THAY ĐỔI: Dùng MSELoss với reduction='mean' để trả về scalar
    loss_fn = nn.HuberLoss(reduction='mean', delta=1.0)
    early_stopping = EarlyStopping(patience=5)
    train_losses, val_losses = [], []
    best_model_state = None

    # --- 6. Vòng lặp Huấn luyện & Đánh giá với Dynamic Masking ---
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_train_loss = 0.0
        for d_coeffs_batch, in train_loader:
            optimizer.zero_grad()
            reconstructed_d_coeffs = model(d_coeffs_batch)
            
            # Sử dụng masked loss để bỏ qua các vùng padding
            loss = masked_loss(reconstructed_d_coeffs, d_coeffs_batch, loss_fn)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for d_coeffs_batch, in test_loader:
                reconstructed_d_coeffs = model(d_coeffs_batch)
                
                # Sử dụng masked loss cho validation
                loss = masked_loss(reconstructed_d_coeffs, d_coeffs_batch, loss_fn)
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(test_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < early_stopping.best_loss:
            print(f"  -> Val loss cải thiện: {early_stopping.best_loss:.6f} -> {avg_val_loss:.6f}. Lưu model.")
            best_model_state = model.state_dict().copy()
        
        if early_stopping(avg_val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # --- 7. Lưu Model Tốt Nhất ---
    if best_model_state:
        # THAY ĐỔI: Lưu cấu hình của model KAN
        model_config = {
            'num_levels': max_num_levels,
            'seq_len': universal_max_len_d,
            'latent_dim': LATENT_DIM
        }
        
        wavelet_params = {
            'type': WAVELET_TYPE,
            'universal_max_len': universal_max_len_d,
            'max_num_levels': max_num_levels
        }
        
        torch.save({
            'model_state_dict': best_model_state,
            'model_config': model_config,
            'best_val_loss': early_stopping.best_loss,
            'wavelet_params': wavelet_params,
            # Các thông tin training khác không đổi
        }, MODEL_SAVE_PATH)
        print(f"\nĐã lưu model KAN tốt nhất vào: {MODEL_SAVE_PATH}")
        
        # THAY ĐỔI: Ghi file config text cho model KAN
        config_text_path = os.path.join(MODEL_SAVE_DIR, 'model_config.txt')
        with open(config_text_path, 'w', encoding='utf-8') as f:
            f.write("=== KAN WAVELET AUTOENCODER MODEL CONFIGURATION (DYNAMIC PADDING) ===\n")
            f.write(f"Number of Wavelet Levels: {model_config['num_levels']}\n")
            f.write(f"Sequence Length (Padded): {model_config['seq_len']}\n")
            f.write(f"Latent Dimension: {model_config['latent_dim']}\n")
            f.write(f"Total Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
            f.write(f"Truncated Coefficients: {truncated_count} / {len(all_coeff_lengths)} ({truncated_count/len(all_coeff_lengths)*100:.1f}%)\n")
            f.write("\n=== WAVELET PARAMETERS ===\n")
            f.write(f"Wavelet Type: {wavelet_params['type']}\n")
            f.write(f"Universal Max Length: {wavelet_params['universal_max_len']} (95th percentile)\n")
            f.write(f"Max Number of Levels: {wavelet_params['max_num_levels']}\n")
            f.write(f"Training Dataset: Full dataset with dynamic padding\n")
        print(f"Đã lưu thông tin cấu hình vào: {config_text_path}")

    # --- 8. Vẽ Đồ thị Huấn luyện (Không thay đổi) ---
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss for KAN Wavelet AE')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(MODEL_SAVE_DIR, 'training_curves.png'))
    plt.close()
    print("Đã lưu đồ thị quá trình huấn luyện.")