import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from stage_1.STL import stl_decomposition, clean_and_augment_d
from stage_2.AE import TimeSeriesAutoencoder
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import RobustScaler

# EarlyStopping class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
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

# --- CÁC HÀM TIỆN ÍCH (KHÔNG THAY ĐỔI) ---
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
    plt.close()
            
    print(f"Đã lưu đồ thị t-SNE vào file: tsne_autoencoder_placeId_{place_id}.png")

def create_padded_data_with_mask(all_a_data, all_d_data, max_length):
    """Tạo padded data với mask"""
    padded_a_data = []
    padded_d_data = []
    masks = []
    
    for a_seq, d_seq in zip(all_a_data, all_d_data):
        actual_length = len(a_seq)
        pad_length = max_length - actual_length
        
        # Padding với giá trị 0
        a_padded = np.pad(a_seq, (0, pad_length), mode='constant', constant_values=0)
        d_padded = np.pad(d_seq, (0, pad_length), mode='constant', constant_values=0)
        
        # Tạo mask: 1 cho dữ liệu thật, 0 cho padding
        mask = np.concatenate([
            np.ones(actual_length),      # Phần dữ liệu thật
            np.zeros(pad_length)         # Phần padding
        ])
        
        padded_a_data.append(a_padded)
        padded_d_data.append(d_padded)
        masks.append(mask)
    
    return (
        torch.FloatTensor(padded_a_data).unsqueeze(-1),
        torch.FloatTensor(padded_d_data).unsqueeze(-1),
        torch.FloatTensor(masks)
    )

def masked_loss(pred, target, mask, loss_fn):
    """Tính loss chỉ trên phần không phải padding"""
    # pred: [batch, seq, features], target: [batch, seq, features], mask: [batch, seq]
    
    # Tính loss element-wise (loss_fn đã được khởi tạo với reduction='none')
    loss = loss_fn(pred, target)  # [batch, seq, features]
    
    # Expand mask to match loss dimensions
    expanded_mask = mask.unsqueeze(-1).expand_as(loss)  # [batch, seq, features]
    
    # Apply mask: zero out padding positions
    masked_loss_values = loss * expanded_mask
    
    # Tính trung bình chỉ trên valid positions
    total_loss = masked_loss_values.sum()
    total_valid_elements = expanded_mask.sum()
    
    return total_loss / (total_valid_elements + 1e-8)  # Avoid division by zero


if __name__ == "__main__":
    # --- 1. Nạp và Chuẩn bị Dữ liệu ---
    DATA_PATH = 'data/cleaned_data_after_idx30.csv'
    MODEL_SAVE_DIR = 'saved_models'  # Thư mục lưu model cuối cùng
    MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'autoencoder_model.pth')

    KAN_out_features = 32  # Số đặc trưng đầu ra từ mỗi KANMoELayer
    feature_dim = KAN_out_features * 2  # Tổng số đặc trưng sau

    try:
        df = pd.read_csv(DATA_PATH)
        print(f"Đã đọc thành công file dữ liệu. Tổng số dòng: {len(df)}")
        
        # Lấy tất cả các placeId
        place_ids = df['placeId'].unique()
        print(f"Tổng số địa điểm: {len(place_ids)}")
        
        # --- THAY ĐỔI 1: XÓA THAM SỐ `decoder_hidden_dim` ---
        # Khởi tạo Autoencoder một lần
        autoencoder = TimeSeriesAutoencoder(
            input_dim=1,
            kan_out_features=KAN_out_features,
            num_experts=8
        )
        
        print(f"Đã khởi tạo Autoencoder với {sum(p.numel() for p in autoencoder.parameters() if p.requires_grad):,} tham số")
        
        # Tạo thư mục lưu model nếu chưa tồn tại
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        
        # Lưu kết quả cho tất cả địa điểm
        all_results = {}
        
        print("\nBắt đầu xử lý tất cả địa điểm...")

        all_a_data = []
        all_d_data = []
        sequence_lengths = []

        for i, place_id in enumerate(place_ids):
            if i % 50 == 0:
                print(f"Đang xử lý địa điểm {i+1}/{len(place_ids)}: {place_id}")

            # Lấy dữ liệu cho địa điểm hiện tại
            place_data = df[df['placeId'] == place_id]['view'].values

            scaler = RobustScaler()
            place_data_scaled = scaler.fit_transform(place_data.reshape(-1, 1)).flatten()

            a_np, d_np = stl_decomposition(place_data_scaled)

            d_np = clean_and_augment_d(d_np, method='mad')  # Clean và augment detail component

            all_a_data.append(a_np)
            all_d_data.append(d_np)
            sequence_lengths.append(len(a_np))

        # Tìm độ dài tối đa để padding
        max_length = max(sequence_lengths)
        print(f"Độ dài chuỗi tối đa: {max_length}")
        print(f"Độ dài chuỗi tối thiểu: {min(sequence_lengths)}")
        
        # Padding tất cả chuỗi về cùng độ dài
        all_a_tensor, all_d_tensor, masks = create_padded_data_with_mask(all_a_data, all_d_data, max_length)
        
        # Chia dữ liệu thành train và test
        indices = np.arange(len(all_a_tensor))
        train_indices, test_indices = train_test_split(indices, test_size=0.1, random_state=42)
        
        # Tạo train dataset
        train_a_tensor = all_a_tensor[train_indices]
        train_d_tensor = all_d_tensor[train_indices]
        train_masks = masks[train_indices]
        train_dataset = TensorDataset(train_a_tensor, train_d_tensor, train_masks)
        train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
        
        # Tạo test dataset
        test_a_tensor = all_a_tensor[test_indices]
        test_d_tensor = all_d_tensor[test_indices]
        test_masks = masks[test_indices]
        test_dataset = TensorDataset(test_a_tensor, test_d_tensor, test_masks)
        test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
        
        print(f"\nTrain samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        print(f"Hoàn thành xử lý {len(place_ids)} địa điểm!")

        NUM_EPOCHS = 100 # Ví dụ: Huấn luyện qua toàn bộ dữ liệu 100 lần
        autoencoder.train() # Chuyển model sang chế độ huấn luyện

        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3, weight_decay=1e-4)
        early_stopping = EarlyStopping(patience=5, min_delta=1e-4)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(NUM_EPOCHS):
            # Training phase
            autoencoder.train()
            epoch_train_losses = []
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}")

            for batch in train_loader:
                a_batch, d_batch, mask_batch = batch

                # Forward pass
                reconstructed_a, reconstructed_d = autoencoder(a_batch, d_batch)

                # Tính loss với mask
                loss_fn = nn.HuberLoss(reduction='none')
                loss_a = masked_loss(reconstructed_a, a_batch, mask_batch, loss_fn)
                loss_d = masked_loss(reconstructed_d, d_batch, mask_batch, loss_fn)
                total_loss = loss_a + loss_d

                # Backward pass và cập nhật trọng số
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                epoch_train_losses.append(total_loss.item())

            # Validation phase
            autoencoder.eval()
            epoch_val_losses = []
            
            with torch.no_grad():
                for batch in test_loader:
                    a_batch, d_batch, mask_batch = batch
                    
                    # Forward pass
                    reconstructed_a, reconstructed_d = autoencoder(a_batch, d_batch)
                    
                    # Tính validation loss
                    loss_fn = nn.HuberLoss(reduction='none')
                    loss_a = masked_loss(reconstructed_a, a_batch, mask_batch, loss_fn)
                    loss_d = masked_loss(reconstructed_d, d_batch, mask_batch, loss_fn)
                    total_loss = loss_a + loss_d
                    
                    epoch_val_losses.append(total_loss.item())

            avg_train_loss = np.mean(epoch_train_losses)
            avg_val_loss = np.mean(epoch_val_losses)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = autoencoder.state_dict().copy()
                print(f"  → New best validation loss: {best_val_loss:.6f}")
            
            # Early stopping check
            if early_stopping(avg_val_loss):
                print(f"Early stopping triggered at epoch {epoch+1}")
                print(f"Best validation loss: {best_val_loss:.6f}")
                break
        
        # Load best model state
        if best_model_state is not None:
            autoencoder.load_state_dict(best_model_state)
            print(f"Loaded best model with validation loss: {best_val_loss:.6f}")

        # Plot training curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved training curves to training_curves.png")

        # Đánh giá model sau khi training
        autoencoder.eval()
        
        # Tính reconstruction loss cho từng địa điểm (sử dụng toàn bộ dataset)
        all_results = {}
        with torch.no_grad():
            for i, place_id in enumerate(place_ids):
                a_seq = all_a_tensor[i:i+1]  # Lấy 1 sample từ toàn bộ dataset
                d_seq = all_d_tensor[i:i+1]
                mask_seq = masks[i:i+1]  # Lấy mask tương ứng
                
                # Forward pass
                reconstructed_a, reconstructed_d = autoencoder(a_seq, d_seq)
                
                # Tính loss với mask và HuberLoss
                loss_fn = nn.HuberLoss(reduction='none')
                loss_a = masked_loss(reconstructed_a, a_seq, mask_seq, loss_fn).item()
                loss_d = masked_loss(reconstructed_d, d_seq, mask_seq, loss_fn).item()
                total_loss = loss_a + loss_d
                
                # Lấy latent features
                latent_features = autoencoder.encoder(a_seq, d_seq)
                
                all_results[place_id] = {
                    'reconstruction_loss': total_loss,
                    'loss_a': loss_a,
                    'loss_d': loss_d,
                    'sequence_length': sequence_lengths[i],
                    'latent_features': latent_features.squeeze(0).numpy()
                }

        # --- Phân tích tổng quan ---
        print("\n=== PHÂN TÍCH TỔNG QUAN ===")
        
        # Thống kê reconstruction loss
        all_losses_eval = [result['reconstruction_loss'] for result in all_results.values()]
        print(f"Reconstruction Loss - Trung bình: {np.mean(all_losses_eval):.6f}")
        print(f"Reconstruction Loss - Std: {np.std(all_losses_eval):.6f}")
        print(f"Reconstruction Loss - Min: {np.min(all_losses_eval):.6f}")
        print(f"Reconstruction Loss - Max: {np.max(all_losses_eval):.6f}")
        
        # Tìm địa điểm có loss cao nhất (potential anomalies)
        sorted_places = sorted(all_results.items(), key=lambda x: x[1]['reconstruction_loss'], reverse=True)
        
        print(f"\nTop 10 địa điểm có reconstruction loss cao nhất (có thể bất thường):")
        for i, (place_id, result) in enumerate(sorted_places[:10]):
            print(f"  {i+1}. PlaceId {place_id}: Loss = {result['reconstruction_loss']:.6f}, Seq_len = {result['sequence_length']}")
        
        # --- Lưu kết quả ra file ---
        print(f"\n=== LUU KẾT QUẢ ===")
        
        # Tạo DataFrame với reconstruction losses
        results_df = pd.DataFrame([
            {
                'placeId': place_id,
                'reconstruction_loss': result['reconstruction_loss'],
                'loss_a': result['loss_a'],
                'loss_d': result['loss_d'],
                'sequence_length': result['sequence_length']
            }
            for place_id, result in all_results.items()
        ])
        
        results_df.to_csv('autoencoder_results_all_places.csv', index=False)
        print(f"Đã lưu kết quả vào file: autoencoder_results_all_places.csv")
        
        # --- Lưu model ---
        print(f"\n=== LƯU MODEL ===")
        
        # --- THAY ĐỔI 2: XÓA `decoder_hidden_dim` KHỎI CONFIG LƯU TRỮ ---
        # Lưu toàn bộ model state dict
        model_info = {
            'model_state_dict': autoencoder.state_dict(),
            'model_config': {
                'input_dim': 1,
                'kan_out_features': KAN_out_features,
                'num_experts': 8
                # ĐÃ XÓA: 'decoder_hidden_dim': 128
            },
            'training_stats': {
                'num_places_processed': len(place_ids),
                'num_train_samples': len(train_dataset),
                'num_test_samples': len(test_dataset),
                'final_train_loss': train_losses[-1] if train_losses else 0,
                'best_val_loss': best_val_loss,
                'total_epochs': len(train_losses),
                'avg_reconstruction_loss': np.mean(all_losses_eval),
                'std_reconstruction_loss': np.std(all_losses_eval),
                'min_reconstruction_loss': np.min(all_losses_eval),
                'max_reconstruction_loss': np.max(all_losses_eval),
                'train_losses': train_losses,
                'val_losses': val_losses
            }
        }
        
        torch.save(model_info, MODEL_SAVE_PATH)
        print(f"Đã lưu model vào: {MODEL_SAVE_PATH}")
        
        # --- THAY ĐỔI 3: XÓA `decoder_hidden_dim` KHỎI FILE CONFIG TEXT ---
        # Lưu thêm file model config riêng để dễ đọc
        config_path = os.path.join(MODEL_SAVE_DIR, 'model_config.txt')
        with open(config_path, 'w') as f:
            f.write("=== AUTOENCODER MODEL CONFIGURATION ===\n")
            f.write(f"Input Dimension: {model_info['model_config']['input_dim']}\n")
            f.write(f"KAN Output Features: {model_info['model_config']['kan_out_features']}\n")
            f.write(f"Number of Experts: {model_info['model_config']['num_experts']}\n")
            # ĐÃ XÓA: Dòng ghi "Decoder Hidden Dimension"
            f.write(f"Total Parameters: {sum(p.numel() for p in autoencoder.parameters() if p.requires_grad):,}\n")
            f.write(f"\n=== TRAINING STATISTICS ===\n")
            f.write(f"Places Processed: {model_info['training_stats']['num_places_processed']}\n")
            f.write(f"Train Samples: {model_info['training_stats']['num_train_samples']}\n")
            f.write(f"Test Samples: {model_info['training_stats']['num_test_samples']}\n")
            f.write(f"Total Epochs: {model_info['training_stats']['total_epochs']}\n")
            f.write(f"Final Train Loss: {model_info['training_stats']['final_train_loss']:.6f}\n")
            f.write(f"Best Validation Loss: {model_info['training_stats']['best_val_loss']:.6f}\n")
            f.write(f"Average Reconstruction Loss: {model_info['training_stats']['avg_reconstruction_loss']:.6f}\n")
            f.write(f"Std Reconstruction Loss: {model_info['training_stats']['std_reconstruction_loss']:.6f}\n")
            f.write(f"Min Reconstruction Loss: {model_info['training_stats']['min_reconstruction_loss']:.6f}\n")
            f.write(f"Max Reconstruction Loss: {model_info['training_stats']['max_reconstruction_loss']:.6f}\n")
        
        print(f"Đã lưu thông tin cấu hình model vào: {config_path}")

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file tại '{DATA_PATH}'. Vui lòng kiểm tra lại đường dẫn.")
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")