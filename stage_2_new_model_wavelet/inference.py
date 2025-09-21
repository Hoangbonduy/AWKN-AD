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

# Import TSAD_eval metrics
try:
    from stage_2_new_model_wavelet.TSAD_eval import Composite_f, Affiliation, Temporal_distance
    TSAD_AVAILABLE = True
except ImportError:
    print("Warning: TSAD_eval not available. Metrics will be limited.")
    TSAD_AVAILABLE = False

# Các hàm load_ground_truth_labels và plot_inference_results không thay đổi

def validate_and_fix_coeffs(coeffs, wavelet='db4'):
    """
    Validate và fix coefficient shapes để đảm bảo tương thích với inverse wavelet transform.
    Wavelet coefficients phải tuân theo cấu trúc hierarchical cụ thể:
    - coeffs[0] là approximation coefficient (cA)
    - coeffs[1:] là detail coefficients (cD) từ level cao nhất xuống level thấp nhất
    - Mỗi level phải có kích thước tương thích với level khác
    """
    if len(coeffs) == 0:
        return coeffs
    
    # Kiểm tra và điều chỉnh kích thước coefficients
    fixed_coeffs = []
    
    # Approximation coefficient (level đầu tiên)
    cA = coeffs[0]
    fixed_coeffs.append(cA)
    
    # Detail coefficients - cần đảm bảo kích thước tương thích
    for i in range(1, len(coeffs)):
        cD = coeffs[i]
        
        # Đối với detail coefficients, kích thước phải tuân theo quy tắc của wavelet
        # Thông thường detail coeff có kích thước tương tự hoặc ít hơn approximation coeff
        expected_len = len(fixed_coeffs[i-1])
        
        if len(cD) > expected_len * 2:
            # Nếu quá dài, truncate
            cD = cD[:expected_len]
        elif len(cD) < max(1, expected_len // 2):
            # Nếu quá ngắn, pad
            pad_size = max(1, expected_len // 2) - len(cD)
            cD = np.pad(cD, (0, pad_size), mode='constant', constant_values=0)
        
        fixed_coeffs.append(cD)
    
    return fixed_coeffs

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
                    # Lấy số đầu tiên, bỏ qua text bổ sung
                    value = line.split(':')[1].strip().split()[0]  # Chỉ lấy phần đầu tiên
                    config['universal_max_len'] = int(value)
                elif 'Max Number of Levels:' in line:
                    config['max_num_levels'] = int(line.split(':')[1].strip())
    return config

def calculate_metrics(predicted_anomalies, ground_truth_anomalies, series_length):
    """
    Tính các metric CompF1, AffF1, TD cho anomaly detection
    """
    metrics = {}
    
    if len(ground_truth_anomalies) == 0:
        metrics['CompF1'] = np.nan
        metrics['AffF1'] = np.nan
        metrics['TD'] = np.nan
        return metrics
    
    # Validate and clean indices
    def validate_indices(indices, max_length):
        """Validate và clean indices để tránh lỗi negative values"""
        if len(indices) == 0:
            return []
        indices = np.array(indices)
        # Loại bỏ indices âm và vượt quá series length
        valid_indices = indices[(indices >= 0) & (indices < max_length)]
        return valid_indices.tolist()
    
    # Clean indices trước khi tính metric
    clean_predicted = validate_indices(predicted_anomalies, series_length)
    clean_ground_truth = validate_indices(ground_truth_anomalies, series_length)
    
    if len(clean_ground_truth) == 0:
        metrics['CompF1'] = np.nan
        metrics['AffF1'] = np.nan
        metrics['TD'] = np.nan
        return metrics
    
    # 1. CompF1 (Composite F1) - Sử dụng TSAD_eval
    if TSAD_AVAILABLE:
        try:
            comp_f = Composite_f(series_length, clean_ground_truth, clean_predicted)
            metrics['CompF1'] = comp_f.get_score()
        except Exception as e:
            print(f"Error calculating CompF1: {e}")
            metrics['CompF1'] = np.nan
    else:
        metrics['CompF1'] = np.nan
    
    # 2. AffF1 (Affiliation F1) - Sử dụng TSAD_eval
    if TSAD_AVAILABLE:
        try:
            aff_f = Affiliation(series_length, clean_ground_truth, clean_predicted)
            metrics['AffF1'] = aff_f.get_score()
        except Exception as e:
            print(f"Error calculating AffF1: {e}")
            metrics['AffF1'] = np.nan
    else:
        metrics['AffF1'] = np.nan
    
    # 3. TD (Temporal Distance) - Sử dụng TSAD_eval
    if TSAD_AVAILABLE:
        try:
            td = Temporal_distance(series_length, clean_ground_truth, clean_predicted)
            metrics['TD'] = td.get_score()
        except Exception as e:
            print(f"Error calculating TD: {e}")
            metrics['TD'] = np.nan
    else:
        metrics['TD'] = np.nan
    
    return metrics

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

# Hàm plot_inference_results cập nhật để hiển thị metrics
def plot_inference_results(place_id, time_series, dates, anomaly_scores, predicted_anomalies, ground_truth_anomalies, threshold, metrics, FOLDER_PATH):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 14), sharex=True)
    if len(ground_truth_anomalies) > 0:
        for idx in ground_truth_anomalies:
            if idx < len(dates):
                ax1.axvspan(dates.iloc[idx], dates.iloc[min(idx+1, len(dates)-1)], 
                           alpha=0.3, color='yellow', label='Ground Truth Anomalies' if idx == ground_truth_anomalies[0] else "")
                ax2.axvspan(dates.iloc[idx], dates.iloc[min(idx+1, len(dates)-1)], 
                           alpha=0.3, color='yellow', label='Ground Truth Anomalies' if idx == ground_truth_anomalies[0] else "")
    ax1.plot(dates, time_series, 'b-', linewidth=1, alpha=0.7, label='Chuỗi Thời Gian Gốc')
    if len(predicted_anomalies) > 0:
        ax1.scatter(dates.iloc[predicted_anomalies], time_series[predicted_anomalies], 
                    color='red', s=100, zorder=5, label='Điểm Bất Thường Dự Đoán (Model)')
    
    # Thêm thông tin metrics vào title
    metrics_text = ""
    if not np.isnan(metrics.get('CompF1', np.nan)):
        metrics_text += f"CompF1: {metrics['CompF1']:.3f} | "
    if not np.isnan(metrics.get('AffF1', np.nan)):
        metrics_text += f"AffF1: {metrics['AffF1']:.3f} | "
    if not np.isnan(metrics.get('TD', np.nan)):
        metrics_text += f"TD: {metrics['TD']:.1f}"
    
    title = f'Kết Quả Phát Hiện Bất Thường - PlaceId: {place_id}'
    if metrics_text:
        title += f'\n{metrics_text}'
    
    ax1.set_title(title, fontsize=16)
    ax1.set_ylabel('View Count', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax2.plot(dates, anomaly_scores, 'm-', linewidth=1, alpha=0.8, label='Anomaly Score (Lỗi Tái Tạo)')
    ax2.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
               label=f'Ngưỡng Percentile ({threshold:.4f})')
    if len(predicted_anomalies) > 0:
        ax2.scatter(dates.iloc[predicted_anomalies], anomaly_scores[predicted_anomalies], 
                    color='red', s=100, zorder=5)
    ax2.set_title('Điểm Số Bất Thường Dựa Trên Lỗi Tái Tạo Wavelet', fontsize=16)
    ax2.set_ylabel('MSE Error', fontsize=12)
    ax2.set_xlabel('Ngày', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    output_filename = f'{FOLDER_PATH}/kan_wavelet_{place_id}.png' # THAY ĐỔI: Tên file output
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"Đã lưu biểu đồ kết quả vào file: {output_filename}")
    plt.close()

# Hàm run_inference_single_place với dynamic padding support
def run_inference_single_place(place_id, df, model, model_config, labels_folder, FOLDER_PATH):
    place_df = df[df['placeId'] == place_id].copy()
    if len(place_df) == 0:
        print(f"Không tìm thấy dữ liệu cho place_id: {place_id}")
        return None
    
    time_series = place_df['view'].values
    dates = pd.to_datetime(place_df['date'])
    ground_truth_anomalies = load_ground_truth_labels(place_id, labels_folder)
    
    # Skip các series quá ngắn
    if len(time_series) < 30:
        print(f"  - Series quá ngắn ({len(time_series)} < 30), skip địa điểm này")
        return None
    
    scaler = RobustScaler()
    time_series_scaled = scaler.fit_transform(time_series.reshape(-1, 1)).flatten()
    level = get_max_wavelet_level(len(time_series_scaled), model_config['wavelet_type'])
    d_coeffs = dwt_transform(time_series_scaled, wavelet=model_config['wavelet_type'], level=level)
    universal_max_len_d = model_config['universal_max_len']
    max_num_levels = model_config['max_num_levels']
    
    # Dynamic padding với truncation cho các coefficient quá dài (giống như main.py)
    truncated_count = 0
    padded_coeffs_for_ts = []
    for i, c in enumerate(d_coeffs):
        if len(c) <= universal_max_len_d:
            # Pad nếu coefficient ngắn hơn universal max length
            padded_c = np.pad(c, (0, universal_max_len_d - len(c)), 'constant')
        else:
            # Truncate nếu coefficient dài hơn universal max length
            padded_c = c[:universal_max_len_d]
            truncated_count += 1
            truncated_count += 1
        padded_coeffs_for_ts.append(padded_c)
    
    # Pad levels nếu thiếu
    while len(padded_coeffs_for_ts) < max_num_levels:
        padded_coeffs_for_ts.append(np.zeros(universal_max_len_d))
    
    d_coeffs_padded = np.stack(padded_coeffs_for_ts, axis=0)
    d_coeffs_tensor = torch.FloatTensor(d_coeffs_padded).unsqueeze(0)
    
    with torch.no_grad():
        reconstructed_d_coeffs_tensor = model(d_coeffs_tensor)
    
    reconstructed_d_coeffs_padded = reconstructed_d_coeffs_tensor.squeeze(0).cpu().numpy()
    
    # Unpad để về kích thước gốc
    reconstructed_coeffs_unpadded = []
    for i in range(len(d_coeffs)):
        original_len = len(d_coeffs[i])
        # Đảm bảo không vượt quá universal_max_len_d khi unpad
        actual_len = min(original_len, universal_max_len_d)
        reconstructed_coeffs_unpadded.append(reconstructed_d_coeffs_padded[i, :actual_len])
    
    # THÊM: Validate và fix coefficients trước khi inverse transform
    try:
        # Validate coefficients trước khi inverse transform
        fixed_coeffs = validate_and_fix_coeffs(reconstructed_coeffs_unpadded, model_config['wavelet_type'])
        reconstructed_signal = inverse_dwt_transform(fixed_coeffs, model_config['wavelet_type'])
    except Exception as e:
        print(f"  - Lỗi trong inverse transform, thử phương pháp dự phòng: {e}")
        # Phương pháp dự phòng: tạo signal đơn giản từ approximation coefficient
        if len(reconstructed_coeffs_unpadded) > 0:
            # Sử dụng approximation coefficient và upsample
            approx_coeff = reconstructed_coeffs_unpadded[0]
            # Upsample bằng cách lặp lại giá trị
            target_len = len(time_series_scaled)
            if len(approx_coeff) < target_len:
                repeat_factor = target_len // len(approx_coeff) + 1
                reconstructed_signal = np.tile(approx_coeff, repeat_factor)[:target_len]
            else:
                reconstructed_signal = approx_coeff[:target_len]
        else:
            # Nếu không có gì, trả về zeros
            reconstructed_signal = np.zeros_like(time_series_scaled)
    
    # Đảm bảo shape match chính xác với original signal
    if len(reconstructed_signal) > len(time_series_scaled):
        reconstructed_signal = reconstructed_signal[:len(time_series_scaled)]
    elif len(reconstructed_signal) < len(time_series_scaled):
        # Pad với giá trị cuối nếu signal ngắn hơn
        pad_size = len(time_series_scaled) - len(reconstructed_signal)
        reconstructed_signal = np.pad(reconstructed_signal, (0, pad_size), mode='edge')
    
    if truncated_count > 0:
        print(f"  - Truncated {truncated_count} coefficients để phù hợp với model")
    
    anomaly_scores = (time_series_scaled - reconstructed_signal)**2
    threshold = np.percentile(anomaly_scores, 98.5)
    predicted_anomalies = np.where(anomaly_scores > threshold)[0].tolist()
    
    # Tính các metric CompF1, AffF1, TD
    metrics = calculate_metrics(predicted_anomalies, ground_truth_anomalies, len(time_series))
    
    print(f"Place ID: {place_id}")
    print(f"  - Dự đoán: {len(predicted_anomalies)} điểm bất thường")
    print(f"  - Ground truth: {len(ground_truth_anomalies)} điểm bất thường")
    if not np.isnan(metrics.get('CompF1', np.nan)):
        print(f"  - CompF1: {metrics['CompF1']:.3f}")
    if not np.isnan(metrics.get('AffF1', np.nan)):
        print(f"  - AffF1: {metrics['AffF1']:.3f}")
    if not np.isnan(metrics.get('TD', np.nan)):
        print(f"  - TD: {metrics['TD']:.1f}")
    
    plot_inference_results(place_id, time_series, dates, anomaly_scores, 
                          predicted_anomalies, ground_truth_anomalies, threshold, metrics, FOLDER_PATH)
    
    return metrics

def run_inference_batch():
    # THAY ĐỔI: Đường dẫn đến model dynamic mới
    DATA_PATH = 'data/cleaned_data_no_zero_periods_filtered.csv'
    MODEL_PATH = 'saved_models_wavelet_kan/wavelet_kan_autoencoder_dynamic.pth'
    CONFIG_PATH = 'saved_models_wavelet_kan/model_config.txt'
    LABELS_FOLDER = 'new_labels_2'
    FOLDER_PATH = 'inference_results_wavelet_kan' # THAY ĐỔI: Thư mục output mới cho dynamic model
    
    os.makedirs(FOLDER_PATH, exist_ok=True)
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CONFIG_PATH):
        print(f"Lỗi: Không tìm thấy file model hoặc config trong thư mục '{os.path.dirname(MODEL_PATH)}'.")
        print("Vui lòng chạy lại file main.py để huấn luyện và tạo ra các file này trước.")
        return
    
    model_config = load_model_config(CONFIG_PATH)
    print(f"Model config loaded: {model_config}")
    
    model = load_wavelet_model(MODEL_PATH, model_config)
    
    df = pd.read_csv(DATA_PATH)
    unique_places = df['placeId'].unique()[:30]  # Test 30 địa điểm đầu tiên
    print(f"\nSẽ xử lý {len(unique_places)} địa điểm đầu tiên:")
    print(unique_places)
    
    # Danh sách để lưu metrics của từng địa điểm
    all_metrics = []
    
    for i, place_id in enumerate(unique_places, 1):
        print(f"\n--- Xử lý {i}/{len(unique_places)}: Place ID {place_id} ---")
        try:
            metrics = run_inference_single_place(place_id, df, model, model_config, LABELS_FOLDER, FOLDER_PATH)
            if metrics:
                all_metrics.append(metrics)
        except Exception as e:
            print(f"Lỗi khi xử lý place_id {place_id}: {e}")
            import traceback
            print(f"Chi tiết lỗi: {traceback.format_exc()}")
            # Vẫn tiếp tục với địa điểm tiếp theo
            continue
    
    # Tính trung bình các metric
    if all_metrics:
        valid_comp_f1 = [m['CompF1'] for m in all_metrics if not np.isnan(m.get('CompF1', np.nan))]
        valid_aff_f1 = [m['AffF1'] for m in all_metrics if not np.isnan(m.get('AffF1', np.nan))]
        valid_td = [m['TD'] for m in all_metrics if not np.isnan(m.get('TD', np.nan))]
        
        print(f"\n=== TỔNG KẾT METRICS ===")
        print(f"Số địa điểm xử lý thành công: {len(all_metrics)}")
        
        if valid_comp_f1:
            avg_comp_f1 = np.mean(valid_comp_f1)
            print(f"CompF1 trung bình: {avg_comp_f1:.3f} (trên {len(valid_comp_f1)} địa điểm)")
        else:
            print("CompF1: Không có dữ liệu hợp lệ")
            
        if valid_aff_f1:
            avg_aff_f1 = np.mean(valid_aff_f1)
            print(f"AffF1 trung bình: {avg_aff_f1:.3f} (trên {len(valid_aff_f1)} địa điểm)")
        else:
            print("AffF1: Không có dữ liệu hợp lệ")
            
        if valid_td:
            avg_td = np.mean(valid_td)
            print(f"TD trung bình: {avg_td:.1f} (trên {len(valid_td)} địa điểm)")
        else:
            print("TD: Không có dữ liệu hợp lệ")
    
    print(f"\nHoàn thành! Đã xuất kết quả vào thư mục: {FOLDER_PATH}")

if __name__ == "__main__":
    print("Bắt đầu inference với KAN wavelet autoencoder DYNAMIC cho 30 địa điểm đầu tiên...")
    run_inference_batch()