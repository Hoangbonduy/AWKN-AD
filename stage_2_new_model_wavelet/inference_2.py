import sys
import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stage_2_new_model_wavelet.wavelet import dwt_transform, get_max_wavelet_level, inverse_dwt_transform
# THAY ĐỔI: Tên file import vẫn là AE, nhưng nội dung bên trong đã là kiến trúc KAN mới
from stage_2_new_model_wavelet.AE import TimeSeriesAutoencoder

# Các hàm load_ground_truth_labels và plot_inference_results không thay đổi

def load_ground_truth_labels(place_id, labels_folder):
    label_file = os.path.join(labels_folder, f'label_{place_id}.csv')
    if os.path.exists(label_file):
        df_labels = pd.read_csv(label_file)
        anomaly_indices = df_labels[df_labels['label'] == 1].index.tolist()
        return anomaly_indices
    else:
        print(f"Không tìm thấy file ground truth: {label_file}")
        return []

def load_model_config(config_path):
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print("Model Config:")
            print(content)
            # THAY ĐỔI: Parse config cho model KAN
            for line in content.split('\n'):
                if 'Number of Wavelet Levels:' in line:
                    config['num_wavelet_levels'] = int(line.split(':')[1].strip())
                elif 'Sequence Length (Padded):' in line:
                    config['seq_len'] = int(line.split(':')[1].strip())
                elif 'Latent Dimension:' in line:
                    config['latent_dim'] = int(line.split(':')[1].strip())
                elif 'Wavelet Type:' in line:
                    config['wavelet_type'] = line.split(':')[1].strip()
                elif 'Universal Max Length:' in line:
                    config['universal_max_len'] = int(line.split(':')[1].strip())
                elif 'Max Number of Levels:' in line:
                    config['max_num_levels'] = int(line.split(':')[1].strip())
    return config

def load_wavelet_model(model_path, model_config, device='cpu'):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # THAY ĐỔI: Khởi tạo model KAN từ config với tên parameter đúng
    model = TimeSeriesAutoencoder(
        num_levels=model_config.get('num_wavelet_levels'),
        seq_len=model_config.get('seq_len'),
        latent_dim=model_config.get('latent_dim')
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    print("Đã tải model KAN thành công!")
    return model

# NEW: residual có dấu + Z-score robust bằng MAD
def compute_signed_zscore(y_scaled, y_hat, win=7):
    y_scaled = np.asarray(y_scaled, float)
    y_hat = np.asarray(y_hat, float)
    base = pd.Series(y_hat).rolling(win, center=True, min_periods=1).median().values
    resid = (y_scaled - y_hat) / (np.abs(base) + 1.0)   # lỗi tương đối có dấu
    med = np.median(resid)
    mad = np.median(np.abs(resid - med)) + 1e-8
    z = (resid - med) / (1.4826 * mad)
    return resid, z

# NEW: residual tương đối có dấu (dùng để lấy percentile hai phía)
def compute_relative_residual(y_scaled, y_hat, win=7):
    y_scaled = np.asarray(y_scaled, float)
    y_hat = np.asarray(y_hat, float)
    base = pd.Series(y_hat).rolling(win, center=True, min_periods=1).median().values
    resid_rel = (y_scaled - y_hat) / (np.abs(base) + 1.0)
    return resid_rel

# NEW: ngưỡng hai phía theo percentile, ví dụ p=98.0
def two_sided_percentile_thresholds(resid_rel, p=98.0, min_side=10):
    r = np.asarray(resid_rel, float)
    if r.size == 0:
        return 0.0, 0.0
    pos = r[r > 0]
    neg_abs = -r[r < 0]  # độ lớn phía âm
    thr_up = np.percentile(pos if len(pos) >= min_side else r, p)
    thr_down_abs = np.percentile(neg_abs if len(neg_abs) >= min_side else -r, p)
    return float(thr_up), float(thr_down_abs)

# REPLACE: vẽ hai ngưỡng spike/drop
def plot_inference_results(place_id, time_series, dates, resid_rel, spikes_idx, drops_idx, ground_truth_anomalies, thr_up, thr_down_abs, FOLDER_PATH):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), sharex=True)
    if len(ground_truth_anomalies) > 0:
        for idx in ground_truth_anomalies:
            if idx < len(dates):
                ax1.axvspan(dates.iloc[idx], dates.iloc[min(idx+1, len(dates)-1)], alpha=0.3, color='yellow', label='Ground Truth Anomalies' if idx == ground_truth_anomalies[0] else "")
                ax2.axvspan(dates.iloc[idx], dates.iloc[min(idx+1, len(dates)-1)], alpha=0.3, color='yellow', label='Ground Truth Anomalies' if idx == ground_truth_anomalies[0] else "")
    ax1.plot(dates, time_series, 'b-', linewidth=1, alpha=0.7, label='Chuỗi Thời Gian Gốc')
    if len(spikes_idx) > 0:
        ax1.scatter(dates.iloc[spikes_idx], np.asarray(time_series)[spikes_idx], color='red', s=80, label='Spike')
    if len(drops_idx) > 0:
        ax1.scatter(dates.iloc[drops_idx], np.asarray(time_series)[drops_idx], color='dodgerblue', s=80, label='Drop')
    ax1.set_title(f'Kết Quả Phát Hiện Bất Thường - PlaceId: {place_id}', fontsize=16)
    ax1.set_ylabel('View Count', fontsize=12)
    ax1.legend(fontsize=12); ax1.grid(True, alpha=0.3)

    ax2.plot(dates, resid_rel, 'm-', linewidth=1, alpha=0.8, label='Residual tương đối (y - ŷ)/(|baseline|+1)')
    ax2.axhline(y=thr_up, color='red', linestyle='--', linewidth=2, label=f'Ngưỡng Spike +{thr_up:.3f} (98th dương)')
    ax2.axhline(y=-thr_down_abs, color='blue', linestyle='--', linewidth=2, label=f'Ngưỡng Drop -{thr_down_abs:.3f} (98th âm)')
    if len(spikes_idx) > 0:
        ax2.scatter(dates.iloc[spikes_idx], np.asarray(resid_rel)[spikes_idx], color='red', s=80)
    if len(drops_idx) > 0:
        ax2.scatter(dates.iloc[drops_idx], np.asarray(resid_rel)[drops_idx], color='dodgerblue', s=80)
    ax2.set_title('Điểm Số Bất Thường (Residual có dấu, ngưỡng percentile hai phía)', fontsize=16)
    ax2.set_ylabel('Residual tương đối'); ax2.set_xlabel('Ngày'); ax2.legend(fontsize=12); ax2.grid(True, alpha=0.3)

    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    output_filename = f'{FOLDER_PATH}/kan_wavelet_{place_id}.png'
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"Đã lưu biểu đồ kết quả vào file: {output_filename}")
    plt.close()

# Hàm run_inference_single_place về cơ bản không đổi, chỉ dùng các hàm đã được cập nhật ở trên
def run_inference_single_place(place_id, df, model, model_config, labels_folder, FOLDER_PATH):
    place_df = df[df['placeId'] == place_id].copy()
    if len(place_df) == 0:
        print(f"Không tìm thấy dữ liệu cho place_id: {place_id}")
        return
    time_series = place_df['view'].values
    dates = pd.to_datetime(place_df['date'])
    ground_truth_anomalies = load_ground_truth_labels(place_id, labels_folder)

    scaler = RobustScaler()
    time_series_scaled = scaler.fit_transform(time_series.reshape(-1, 1)).flatten()

    level = get_max_wavelet_level(len(time_series_scaled), model_config['wavelet_type'])
    d_coeffs = dwt_transform(time_series_scaled, wavelet=model_config['wavelet_type'], level=level)
    universal_max_len_d = model_config['universal_max_len']
    max_num_levels = model_config['max_num_levels']

    padded_coeffs_for_ts = []
    for c in d_coeffs:
        padded_c = np.pad(c, (0, universal_max_len_d - len(c)), 'constant')
        padded_coeffs_for_ts.append(padded_c)
    while len(padded_coeffs_for_ts) < max_num_levels:
        padded_coeffs_for_ts.append(np.zeros(universal_max_len_d))

    d_coeffs_padded = np.stack(padded_coeffs_for_ts, axis=0)
    d_coeffs_tensor = torch.FloatTensor(d_coeffs_padded).unsqueeze(0)

    with torch.no_grad():
        reconstructed_d_coeffs_tensor = model(d_coeffs_tensor)

    reconstructed_d_coeffs_padded = reconstructed_d_coeffs_tensor.squeeze(0).cpu().numpy()
    reconstructed_coeffs_unpadded = []
    for i in range(len(d_coeffs)):
        original_len = len(d_coeffs[i])
        reconstructed_coeffs_unpadded.append(reconstructed_d_coeffs_padded[i, :original_len])

    reconstructed_signal = inverse_dwt_transform(reconstructed_coeffs_unpadded, model_config['wavelet_type'])
    reconstructed_signal = reconstructed_signal[:len(time_series_scaled)]

    # NEW: residual có dấu tương đối và ngưỡng hai phía theo percentile 98th
    resid_rel = compute_relative_residual(time_series_scaled, reconstructed_signal, win=7)
    PERCENTILE = 98.0
    thr_up, thr_down_abs = two_sided_percentile_thresholds(resid_rel, p=PERCENTILE, min_side=10)
    spikes_idx = np.where(resid_rel > thr_up)[0].tolist()
    drops_idx  = np.where(resid_rel < -thr_down_abs)[0].tolist()

    print(f"Place ID: {place_id}")
    print(f"  - Ngưỡng: +{thr_up:.4f} / -{thr_down_abs:.4f} (p={PERCENTILE}%)")
    print(f"  - Dự đoán: spike={len(spikes_idx)}, drop={len(drops_idx)}, tổng={len(set(spikes_idx + drops_idx))}")
    print(f"  - Ground truth: {len(ground_truth_anomalies)} điểm bất thường")

    plot_inference_results(place_id, time_series, dates, resid_rel, spikes_idx, drops_idx, ground_truth_anomalies, thr_up, thr_down_abs, FOLDER_PATH)

def run_inference_batch():
    # THAY ĐỔI: Đường dẫn đến model và config KAN mới
    DATA_PATH = 'data/cleaned_data_no_zero_periods_filtered.csv'
    MODEL_PATH = 'saved_models_wavelet_kan/wavelet_kan_autoencoder.pth'
    CONFIG_PATH = 'saved_models_wavelet_kan/model_config.txt'
    LABELS_FOLDER = 'new_labels_2'
    FOLDER_PATH = 'inference_results_wavelet_kan_2' # THAY ĐỔI: Thư mục output mới
    
    os.makedirs(FOLDER_PATH, exist_ok=True)
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CONFIG_PATH):
        print(f"Lỗi: Không tìm thấy file model hoặc config trong thư mục '{os.path.dirname(MODEL_PATH)}'.")
        print("Vui lòng chạy lại file main.py để huấn luyện và tạo ra các file này trước.")
        return
    
    model_config = load_model_config(CONFIG_PATH)
    print(f"Model config loaded: {model_config}")
    
    model = load_wavelet_model(MODEL_PATH, model_config)
    
    df = pd.read_csv(DATA_PATH)
    unique_places = df['placeId'].unique()[:30]
    print(f"\nSẽ xử lý {len(unique_places)} địa điểm đầu tiên:")
    print(unique_places)
    
    for i, place_id in enumerate(unique_places, 1):
        print(f"\n--- Xử lý {i}/{len(unique_places)}: Place ID {place_id} ---")
        try:
            run_inference_single_place(place_id, df, model, model_config, LABELS_FOLDER, FOLDER_PATH)
        except Exception as e:
            print(f"Lỗi khi xử lý place_id {place_id}: {e}")
            continue
    
    print(f"\nHoàn thành! Đã xuất kết quả vào thư mục: {FOLDER_PATH}")

if __name__ == "__main__":
    print("Bắt đầu inference với KAN wavelet autoencoder cho 30 địa điểm đầu tiên...")
    run_inference_batch()