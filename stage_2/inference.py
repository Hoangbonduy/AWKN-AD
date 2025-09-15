import sys
import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import RobustScaler

# Giả sử các file này tồn tại trong các thư mục tương ứng
# Thêm đường dẫn để import các module tùy chỉnh
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stage_1.KAMA import kama_decomposition
from stage_1.STL import stl_decomposition
from stage_2.AE import TimeSeriesAutoencoder

def load_model(model_path):
    """Tải mô hình autoencoder đã được huấn luyện"""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Trích xuất cấu hình mô hình
    config = checkpoint['model_config']
    
    # Khởi tạo mô hình với cấu hình đã lưu
    model = TimeSeriesAutoencoder(
        input_dim=config['input_dim'],
        kan_out_features=config['kan_out_features'],
        num_experts=config['num_experts']
    )
    
    # Tải các trọng số đã huấn luyện
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Training stats: {checkpoint['training_stats']}")
    
    return model, checkpoint['training_stats']

def load_ground_truth_labels(place_id, labels_dir):
    """Tải nhãn bất thường ground truth cho một địa điểm"""
    label_file = os.path.join(labels_dir, f'label_{place_id}.csv')
    
    if os.path.exists(label_file):
        labels_df = pd.read_csv(label_file)
        labels_df['is_anomaly'] = labels_df['label']
        labels_df['date'] = pd.to_datetime(labels_df['date'])
        return labels_df
    else:
        print(f"Warning: No ground truth labels found for place {place_id}")
        return None

def group_consecutive_indices(indices, max_gap=1):
    """Nhóm các chỉ số liên tiếp thành các cụm"""
    if len(indices) == 0:
        return []
    
    groups = []
    start = indices[0]
    end = indices[0]
    
    for i in range(1, len(indices)):
        if indices[i] - end <= max_gap + 1:
            end = indices[i]
        else:
            groups.append((start, end))
            start = indices[i]
            end = indices[i]
    
    groups.append((start, end))
    return groups

def filter_anomaly_groups(anomaly_indices, min_group_size=3):
    """
    Lọc bỏ các nhóm anomaly có ít hơn min_group_size điểm liên tiếp
    """
    if len(anomaly_indices) == 0:
        return np.array([])
    
    # Nhóm các chỉ số liên tiếp
    anomaly_groups = group_consecutive_indices(anomaly_indices, max_gap=1)
    
    # Lọc ra các nhóm đủ lớn
    filtered_indices = []
    for start_idx, end_idx in anomaly_groups:
        group_size = end_idx - start_idx + 1
        if group_size >= min_group_size:
            # Thêm tất cả chỉ số trong nhóm này
            for idx in range(start_idx, end_idx + 1):
                if idx in anomaly_indices:  # Đảm bảo nó ban đầu là anomaly
                    filtered_indices.append(idx)
    
    return np.array(filtered_indices)

def plot_combined_results(
    place_id, time_series, dates,
    trend_anomaly_indices, other_anomaly_indices,
    ground_truth_labels=None, loss_a=None, loss_d=None,
    threshold_d=None, save_path=None
):
    """Vẽ biểu đồ chuỗi thời gian với cả hai loại bất thường và ground truth"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 18), sharex=True)
    
    # === BIỂU ĐỒ 1: CHUỖI THỜI GIAN GỐC ===
    ax1.plot(dates, time_series, 'b-', linewidth=1, alpha=0.7, label='Original Time Series')
    
    # Vẽ Ground Truth (vàng)
    if ground_truth_labels is not None:
        gt_anomaly_dates = ground_truth_labels[ground_truth_labels['is_anomaly'] == 1]['date']
        for i, gt_date in enumerate(gt_anomaly_dates):
            ax1.axvspan(gt_date - pd.Timedelta(days=0.5), gt_date + pd.Timedelta(days=0.5),
                       alpha=0.4, color='yellow',
                       label='Ground Truth Anomaly' if i == 0 else "")

    # Vẽ Trend Anomalies (cam)
    if len(trend_anomaly_indices) > 0:
        anomaly_groups = group_consecutive_indices(trend_anomaly_indices, max_gap=2)
        for i, (start_idx, end_idx) in enumerate(anomaly_groups):
            start_date = dates.iloc[max(0, start_idx-1)]
            end_date = dates.iloc[min(len(dates)-1, end_idx+1)]
            ax1.axvspan(start_date, end_date,
                       alpha=0.4, color='orange',
                       label='Predicted Trend Anomaly' if i == 0 else "")

    # Vẽ Other Anomalies (đỏ)
    if len(other_anomaly_indices) > 0:
        ax1.scatter([dates.iloc[i] for i in other_anomaly_indices],
                   [time_series[i] for i in other_anomaly_indices],
                   color='red', s=100, zorder=5, label='Predicted Point Anomaly')
    
    ax1.set_title(f'Time Series - PlaceId: {place_id}', fontsize=16)
    ax1.set_ylabel('View Count', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # === BIỂU ĐỒ 2: RECONSTRUCTION LOSS (A COMPONENT) ===
    if loss_a is not None:
        ax2.plot(dates, loss_a, 'g-', linewidth=1, alpha=0.7, label='Reconstruction Loss (A component)')
        if len(trend_anomaly_indices) > 0:
            anomaly_groups = group_consecutive_indices(trend_anomaly_indices, max_gap=2)
            for i, (start_idx, end_idx) in enumerate(anomaly_groups):
                start_date = dates.iloc[max(0, start_idx-1)]
                end_date = dates.iloc[min(len(dates)-1, end_idx+1)]
                ax2.axvspan(start_date, end_date,
                           alpha=0.4, color='orange',
                           label='Predicted Trend Anomaly' if i == 0 else "")
        ax2.set_title('Reconstruction Loss for Trend Anomalies (A component)', fontsize=16)
        ax2.set_ylabel('Loss Value', fontsize=12)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)

    # === BIỂU ĐỒ 3: RECONSTRUCTION LOSS (D COMPONENT) ===
    if loss_d is not None:
        ax3.plot(dates, loss_d, 'm-', linewidth=1, alpha=0.7, label='Reconstruction Loss (D component)')
        if threshold_d is not None:
            ax3.axhline(y=threshold_d, color='red', linestyle='--', linewidth=2,
                       label=f'99th Percentile Threshold ({threshold_d:.4f})')
        if len(other_anomaly_indices) > 0:
            ax3.scatter([dates.iloc[i] for i in other_anomaly_indices],
                       [loss_d[i] for i in other_anomaly_indices],
                       color='red', s=100, zorder=5, label='Predicted Point Anomaly')
        ax3.set_title('Reconstruction Loss for Point Anomalies (D component)', fontsize=16)
        ax3.set_ylabel('Loss Value', fontsize=12)
        ax3.legend(fontsize=12)
        ax3.grid(True, alpha=0.3)

    # Cài đặt chung
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
    
    print(f"Running combined inference on {len(place_ids)} places...")
    
    model, _ = load_model(model_path)
    
    output_dir = 'inference_results'
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    # Ngưỡng percentile cho từng loại bất thường
    trend_percentile = 97.0
    other_percentile = 98.0
    
    for i, place_id in enumerate(place_ids):
        print(f"\nProcessing place {i+1}/{len(place_ids)}: {place_id}")
        
        place_data = df[df['placeId'] == place_id].copy().sort_values('date').reset_index(drop=True)
        time_series = place_data['view'].values
        dates = pd.to_datetime(place_data['date'])
        
        ground_truth = load_ground_truth_labels(place_id, labels_dir)
        
        scaler = RobustScaler()
        time_series_scaled = scaler.fit_transform(time_series.reshape(-1, 1)).flatten()
        
        _, d_np = kama_decomposition(time_series_scaled)
        a_np, _ = stl_decomposition(time_series_scaled)
        
        a_tensor = torch.FloatTensor(a_np.copy()).unsqueeze(0).unsqueeze(-1)
        d_tensor = torch.FloatTensor(d_np.copy()).unsqueeze(0).unsqueeze(-1)
        
        with torch.no_grad():
            reconstructed_a, reconstructed_d = model(a_tensor, d_tensor)
            loss_fn = torch.nn.MSELoss(reduction='none')
            loss_a = loss_fn(reconstructed_a, a_tensor).squeeze().numpy()
            loss_d = loss_fn(reconstructed_d, d_tensor).squeeze().numpy()
        
        # 1. Phát hiện Trend Anomalies (từ loss_a)
        threshold_a = np.percentile(loss_a, trend_percentile)
        trend_anomaly_indices_raw = np.where(loss_a > threshold_a)[0]
        
        # Lọc bỏ nhóm trend anomalies có ít hơn 3 điểm liên tiếp
        trend_anomaly_indices = filter_anomaly_groups(trend_anomaly_indices_raw, min_group_size=3)
        
        # 2. Phát hiện Other Anomalies (từ loss_d)
        threshold_d = np.percentile(loss_d, other_percentile)
        other_anomaly_indices = np.where(loss_d > threshold_d)[0]
        
        print(f"  - Raw trend anomalies ({trend_percentile}th percentile on loss_a): {len(trend_anomaly_indices_raw)} points")
        print(f"  - Filtered trend anomalies (≥3 consecutive points): {len(trend_anomaly_indices)} points")
        print(f"  - Raw point anomalies ({other_percentile}th percentile on loss_d): {len(other_anomaly_indices)} points")
        print(f"  - Filtered point anomalies (≥3 consecutive points): {len(other_anomaly_indices)} points")

        plot_path = os.path.join(output_dir, f'combined_{i+1}_{place_id}.png')
        plot_combined_results(
            place_id, time_series, dates,
            trend_anomaly_indices, other_anomaly_indices,
            ground_truth, loss_a, loss_d,
            threshold_d, plot_path
        )
        
        results.append({
            'place_id': place_id,
            'num_trend_anomalies_raw': len(trend_anomaly_indices_raw),
            'num_trend_anomalies_filtered': len(trend_anomaly_indices),
            'num_point_anomalies_raw': len(other_anomaly_indices),
            'num_point_anomalies_filtered': len(other_anomaly_indices),
            'threshold_a': threshold_a,
            'threshold_d': threshold_d,
            'avg_loss_a': np.mean(loss_a),
            'avg_loss_d': np.mean(loss_d),
            'has_ground_truth': ground_truth is not None,
            'num_ground_truth_anomalies': len(ground_truth[ground_truth['is_anomaly'] == 1]) if ground_truth is not None else 0,
        })

    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, 'inference_summary_combined.csv')
    results_df.to_csv(results_path, index=False)
    
    print("\n=== INFERENCE SUMMARY ===")
    print(f"Total places processed: {len(place_ids)}")
    print(f"Average raw trend anomalies per place: {results_df['num_trend_anomalies_raw'].mean():.2f}")
    print(f"Average filtered trend anomalies per place: {results_df['num_trend_anomalies_filtered'].mean():.2f}")
    print(f"Average raw point anomalies per place: {results_df['num_point_anomalies_raw'].mean():.2f}")
    print(f"Average filtered point anomalies per place: {results_df['num_point_anomalies_filtered'].mean():.2f}")
    print(f"Results saved to: {results_path}")
    
    # Thống kê về việc lọc
    trend_filtered_out = results_df['num_trend_anomalies_raw'].sum() - results_df['num_trend_anomalies_filtered'].sum()
    point_filtered_out = results_df['num_point_anomalies_raw'].sum() - results_df['num_point_anomalies_filtered'].sum()
    print(f"\n=== FILTERING SUMMARY ===")
    print(f"Trend anomalies filtered out: {trend_filtered_out} ({trend_filtered_out/results_df['num_trend_anomalies_raw'].sum()*100:.1f}%)")
    print(f"Point anomalies filtered out: {point_filtered_out} ({point_filtered_out/results_df['num_point_anomalies_raw'].sum()*100:.1f}%)")

if __name__ == "__main__":
    # Đường dẫn
    DATA_PATH = 'data/cleaned_data_no_zero_periods_filtered.csv'
    LABELS_DIR = 'new_labels_2'
    MODEL_PATH = 'saved_models_2/autoencoder_model.pth'
    
    # Kiểm tra sự tồn tại của file
    for path in [DATA_PATH, LABELS_DIR, MODEL_PATH]:
        if not os.path.exists(path):
            print(f"Error: Path not found at {path}")
            exit(1)
    
    # Chạy suy luận
    try:
        inference_on_places(DATA_PATH, LABELS_DIR, MODEL_PATH, num_places=30)
        print("\nInference completed successfully!")
    except Exception as e:
        print(f"\nError during inference: {e}")
        import traceback
        traceback.print_exc()