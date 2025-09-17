import sys
import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import RobustScaler

# Thêm đường dẫn để import các module tùy chỉnh
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stage_1.KAMA import kama_decomposition
from stage_1.STL import stl_decomposition
# Đảm bảo file AE.py đã được cập nhật với kiến trúc HE-KAN
from stage_2_new_model.AE import TimeSeriesAutoencoder

### THAY ĐỔI 1: CẬP NHẬT HÀM TẢI MODEL ###
def load_he_kan_model(model_path, device='cpu'):
    """
    Tải mô hình HE-KAN đã được huấn luyện.
    Hàm này đã được cập nhật để đọc config mới với 'gru_hidden_dim'.
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['model_config']
    
    # Khởi tạo mô hình với kiến trúc HE-KAN mới
    model = TimeSeriesAutoencoder(
        input_dim=config.get('input_dim', 1),
        kan_out_features=config.get('kan_out_features', 16),
        gru_hidden_dim=config.get('gru_hidden_dim', 32), # Tham số mới cho GRU
        num_experts=config.get('num_experts', 8)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device) # Chuyển model đến device
    
    print(f"HE-KAN model loaded successfully on {device}!")
    if 'training_stats' in checkpoint:
        print(f"Training stats: {checkpoint['training_stats']}")
    
    return model, checkpoint.get('training_stats', {})

# --- CÁC HÀM TIỆN ÍCH ---
def load_ground_truth_labels(place_id, labels_dir):
    """Tải nhãn bất thường ground truth cho một địa điểm"""
    label_file = os.path.join(labels_dir, f'label_{place_id}.csv')
    
    if os.path.exists(label_file):
        labels_df = pd.read_csv(label_file)
        labels_df['is_anomaly'] = labels_df['label']
        labels_df['date'] = pd.to_datetime(labels_df['date'])
        return labels_df
    else:
        # print(f"Warning: No ground truth labels found for place {place_id}")
        return None

def group_consecutive_indices(indices, max_gap=1):
    """Nhóm các chỉ số liên tiếp thành các cụm"""
    if len(indices) == 0: return []
    groups = []
    start = end = indices[0]
    for i in range(1, len(indices)):
        if indices[i] - end <= max_gap + 1: end = indices[i]
        else:
            groups.append((start, end))
            start = end = indices[i]
    groups.append((start, end))
    return groups

def filter_anomaly_groups(anomaly_indices, min_group_size=5):
    """Lọc bỏ các nhóm anomaly có ít hơn min_group_size điểm liên tiếp"""
    if len(anomaly_indices) == 0: return np.array([])
    anomaly_groups = group_consecutive_indices(anomaly_indices, max_gap=1)
    filtered_indices = []
    for start_idx, end_idx in anomaly_groups:
        if end_idx - start_idx + 1 >= min_group_size:
            for idx in range(start_idx, end_idx + 1):
                if idx in anomaly_indices: filtered_indices.append(idx)
    return np.array(filtered_indices)

### THÊM HÀM MỚI ###
def moving_average(data, window_size=7):
    """Tính trung bình trượt để làm mượt chuỗi loss."""
    return np.convolve(data, np.ones(window_size), 'same') / window_size

# Hàm vẽ biểu đồ (plot_combined_results) giữ nguyên như file cũ của bạn
def plot_combined_results(
    place_id, time_series, dates,
    trend_anomaly_indices, point_anomaly_indices,
    ground_truth_labels=None, loss_a=None, loss_d=None,
    threshold_d=None, threshold_a_smoothed=None, smoothed_loss_a=None, save_path=None
):
    """Vẽ biểu đồ chuỗi thời gian, cập nhật để hiển thị loss đã làm mượt."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 18), sharex=True)
    
    # === BIỂU ĐỒ 1: CHUỖI THỜI GIAN GỐC ===
    ax1.plot(dates, time_series, 'b-', linewidth=1, alpha=0.7, label='Original Time Series')
    if ground_truth_labels is not None:
        gt_dates = ground_truth_labels[ground_truth_labels['is_anomaly'] == 1]['date']
        for i, date in enumerate(gt_dates):
            ax1.axvspan(date - pd.Timedelta(days=0.5), date + pd.Timedelta(days=0.5),
                       alpha=0.4, color='yellow', label='Ground Truth Anomaly' if i == 0 else "")
    if len(trend_anomaly_indices) > 0:
        groups = group_consecutive_indices(trend_anomaly_indices, max_gap=2)
        for i, (start, end) in enumerate(groups):
            ax1.axvspan(dates.iloc[max(0, start)], dates.iloc[min(len(dates)-1, end)],
                       alpha=0.4, color='orange', label='Predicted Trend Anomaly' if i == 0 else "")
    if len(point_anomaly_indices) > 0:
        ax1.scatter(dates.iloc[point_anomaly_indices], time_series[point_anomaly_indices],
                   color='red', s=100, zorder=5, label='Predicted Point Anomaly')
    ax1.set_title(f'Time Series - PlaceId: {place_id}', fontsize=16)
    ax1.set_ylabel('View Count', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # === BIỂU ĐỒ 2: RECONSTRUCTION LOSS (A COMPONENT) - ĐÃ LÀM MƯỢT ===
    if loss_a is not None and smoothed_loss_a is not None:
        ax2.plot(dates, loss_a, 'g-', linewidth=1, alpha=0.3, label='Original Loss (A)')
        ax2.plot(dates, smoothed_loss_a, 'g-', linewidth=2, label='Smoothed Loss (A)')
        if threshold_a_smoothed is not None:
            ax2.axhline(y=threshold_a_smoothed, color='orange', linestyle='--', linewidth=2,
                       label=f'Trend Threshold ({threshold_a_smoothed:.4f})')
        if len(trend_anomaly_indices) > 0:
            groups = group_consecutive_indices(trend_anomaly_indices, max_gap=2)
            for i, (start, end) in enumerate(groups):
                ax2.axvspan(dates.iloc[max(0, start)], dates.iloc[min(len(dates)-1, end)],
                           alpha=0.4, color='orange', label='Predicted Trend Anomaly' if i == 0 else "")
        ax2.set_title('Reconstruction Loss for Trend Anomalies (A component)', fontsize=16)
        ax2.set_ylabel('Loss Value', fontsize=12)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)

    # === BIỂU ĐỒ 3: RECONSTRUCTION LOSS (D COMPONENT) ===
    if loss_d is not None:
        ax3.plot(dates, loss_d, 'm-', linewidth=1, alpha=0.7, label='Reconstruction Loss (D component)')
        if threshold_d is not None:
            ax3.axhline(y=threshold_d, color='red', linestyle='--', linewidth=2,
                       label=f'Point Threshold ({threshold_d:.4f})')
        if len(point_anomaly_indices) > 0:
            ax3.scatter(dates.iloc[point_anomaly_indices], loss_d[point_anomaly_indices],
                       color='red', s=100, zorder=5, label='Predicted Point Anomaly')
        ax3.set_title('Reconstruction Loss for Point Anomalies (D component)', fontsize=16)
        ax3.set_ylabel('Loss Value', fontsize=12)
        ax3.legend(fontsize=12)
        ax3.grid(True, alpha=0.3)

    ax3.set_xlabel('Date', fontsize=12)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    plt.close()


def inference_on_places(data_path, labels_dir, model_path, num_places=30):
    """Chạy suy luận trên `num_places` địa điểm đầu tiên từ tập dữ liệu"""
    df = pd.read_csv(data_path)
    place_ids = df['placeId'].unique()[:num_places]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running combined inference on {len(place_ids)} places on {device}...")
    
    model, _ = load_he_kan_model(model_path, device)
    
    output_dir = 'inference_results_he_kan'
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    # Tham số cho việc phát hiện anomaly
    point_percentile = 99.0
    trend_percentile = 95.0 # Hạ ngưỡng cho trend
    trend_smoothing_window = 7
    trend_min_duration = 5

    for i, place_id in enumerate(place_ids):
        print(f"\nProcessing place {i+1}/{len(place_ids)}: {place_id}")
        
        place_data = df[df['placeId'] == place_id].copy().sort_values('date').reset_index(drop=True)
        time_series = place_data['view'].values
        dates = pd.to_datetime(place_data['date'])
        
        ground_truth = load_ground_truth_labels(place_id, labels_dir)
        
        # Tiền xử lý
        scaler = RobustScaler()
        time_series_scaled = scaler.fit_transform(time_series.reshape(-1, 1)).flatten()
        a_np, _ = stl_decomposition(time_series_scaled, period=29, robust=True)
        _, d_np = kama_decomposition(time_series_scaled)
        
        # Chuyển thành tensor
        a_tensor = torch.FloatTensor(a_np.copy()).unsqueeze(0).unsqueeze(-1).to(device)
        d_tensor = torch.FloatTensor(d_np.copy()).unsqueeze(0).unsqueeze(-1).to(device)
        
        # Forward pass
        with torch.no_grad():
            reconstructed_a, reconstructed_d = model(a_tensor, d_tensor)
            loss_fn = torch.nn.HuberLoss(reduction='none') # Huber loss ít nhạy cảm với outlier hơn MSE
            loss_a = loss_fn(reconstructed_a, a_tensor).squeeze().cpu().numpy()
            loss_d = loss_fn(reconstructed_d, d_tensor).squeeze().cpu().numpy()

        ### THAY ĐỔI 2: LOGIC PHÁT HIỆN ANOMALY MỚI ###

        # 1. Phát hiện Point Anomalies (từ loss_d) - Logic không đổi
        threshold_d = np.percentile(loss_d, point_percentile)
        point_anomaly_indices = np.where(loss_d >= threshold_d)[0]
        
        # 2. Phát hiện Trend Anomalies (từ loss_a) - Logic mới
        smoothed_loss_a = moving_average(loss_a, window_size=trend_smoothing_window)
        threshold_a_smoothed = np.percentile(smoothed_loss_a, trend_percentile)
        raw_trend_indices = np.where(smoothed_loss_a >= threshold_a_smoothed)[0]
        trend_anomaly_indices = filter_anomaly_groups(raw_trend_indices, min_group_size=trend_min_duration)
        
        print(f"  - Point anomalies ({point_percentile}th percentile on loss_d): {len(point_anomaly_indices)} points")
        print(f"  - Trend anomalies (after smoothing & filtering, >={trend_min_duration} days): {len(trend_anomaly_indices)} points")

        # Vẽ biểu đồ
        plot_path = os.path.join(output_dir, f'combined_{i+1}_{place_id}.png')
        plot_combined_results(
            place_id, time_series, dates,
            trend_anomaly_indices, point_anomaly_indices,
            ground_truth, loss_a, loss_d,
            threshold_d, threshold_a_smoothed, smoothed_loss_a, plot_path
        )
        
        results.append({
            'place_id': place_id,
            'num_trend_anomalies': len(trend_anomaly_indices),
            'num_point_anomalies': len(point_anomaly_indices),
            'threshold_a_smoothed': threshold_a_smoothed,
            'threshold_d': threshold_d,
            'avg_loss_a': np.mean(loss_a),
            'avg_loss_d': np.mean(loss_d),
            'has_ground_truth': ground_truth is not None,
            'num_ground_truth_anomalies': len(ground_truth[ground_truth['is_anomaly'] == 1]) if ground_truth is not None else 0,
        })

    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, 'inference_summary_he_kan.csv')
    results_df.to_csv(results_path, index=False)
    
    print("\n=== HE-KAN INFERENCE SUMMARY ===")
    print(f"Total places processed: {len(place_ids)}")
    print(f"Average trend anomalies per place: {results_df['num_trend_anomalies'].mean():.2f}")
    print(f"Average point anomalies per place: {results_df['num_point_anomalies'].mean():.2f}")
    print(f"Results saved to: {results_path}")
    

if __name__ == "__main__":
    # Đường dẫn
    DATA_PATH = 'data/cleaned_data_no_zero_periods_filtered.csv'
    LABELS_DIR = 'new_labels_2'
    # THAY ĐỔI: Trỏ đến file model HE-KAN đã được huấn luyện
    MODEL_PATH = 'saved_models_new_model/autoencoder_model.pth' 
    
    for path in [DATA_PATH, MODEL_PATH, LABELS_DIR]:
        if not os.path.exists(path):
            print(f"Error: Path not found at {path}")
            exit(1)
    
    try:
        inference_on_places(DATA_PATH, LABELS_DIR, MODEL_PATH, num_places=30)
        print("\nInference completed successfully!")
    except Exception as e:
        print(f"\nError during inference: {e}")
        import traceback
        traceback.print_exc()