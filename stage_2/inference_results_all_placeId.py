import sys
import os
import pandas as pd
import torch
import numpy as np
import time
from sklearn.preprocessing import RobustScaler
from joblib import Parallel, delayed

# Thêm đường dẫn để import các module tùy chỉnh
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stage_1.STL import stl_decomposition
from stage_1.KAMA import kama_decomposition
from stage_2.AE import TimeSeriesAutoencoder

# --- Các hàm group_consecutive_indices, filter_anomaly_groups giữ nguyên ---
def group_consecutive_indices(indices, max_gap=1):
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

def filter_anomaly_groups(anomaly_indices, min_group_size=3):
    if len(anomaly_indices) == 0: return np.array([])
    anomaly_groups = group_consecutive_indices(anomaly_indices, max_gap=1)
    filtered_indices = []
    for start_idx, end_idx in anomaly_groups:
        if end_idx - start_idx + 1 >= min_group_size:
            for idx in range(start_idx, end_idx + 1):
                if idx in anomaly_indices: filtered_indices.append(idx)
    return np.array(filtered_indices)

# --- THAY ĐỔI 1: TẠO HÀM WORKER NHẬN CHUỖI DỮ LIỆU VÀ ĐƯỜNG DẪN MODEL ---
def process_chunk(chunk_of_data, model_path, trend_percentile, other_percentile):
    """
    Hàm này sẽ được thực thi bởi một tiến trình con (worker).
    Nó sẽ tải model một lần và xử lý một "chunk" dữ liệu.
    """
    # --- SỬA LỖI Ở ĐÂY ---
    # 1. Tải toàn bộ dictionary checkpoint với weights_only=False
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # 2. Lấy config và khởi tạo lại kiến trúc model
    config = checkpoint['model_config']
    model = TimeSeriesAutoencoder(
        input_dim=config['input_dim'],
        kan_out_features=config['kan_out_features'],
        num_experts=config['num_experts']
    )
    
    # 3. Tải các trọng số đã huấn luyện vào model
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # --- KẾT THÚC SỬA LỖI ---

    chunk_anomalies = []
    
    # Timing variables for this chunk
    total_data_prep_time = 0
    total_scaling_time = 0
    total_decomposition_time = 0
    total_loss_calc_time = 0
    total_anomaly_detection_time = 0
    
    for place_id, place_data in chunk_of_data:
        # Data Preparation
        prep_start = time.time()
        time_series = place_data['view'].values
        dates = place_data['date']
        prep_end = time.time()
        total_data_prep_time += (prep_end - prep_start)
        
        # Data Scaling
        scaling_start = time.time()
        scaler = RobustScaler()
        time_series_scaled = scaler.fit_transform(time_series.reshape(-1, 1)).flatten()
        scaling_end = time.time()
        total_scaling_time += (scaling_end - scaling_start)
        
        # Decomposition
        decomp_start = time.time()
        a_np, _ = stl_decomposition(time_series_scaled)
        _, d_np = kama_decomposition(time_series_scaled)
        decomp_end = time.time()
        total_decomposition_time += (decomp_end - decomp_start)
        
        # Loss Calculation
        loss_start = time.time()
        a_tensor = torch.FloatTensor(a_np.copy()).unsqueeze(0).unsqueeze(-1)
        d_tensor = torch.FloatTensor(d_np.copy()).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            reconstructed_a, reconstructed_d = model(a_tensor, d_tensor)
            loss_fn = torch.nn.HuberLoss(reduction='none')
            loss_a = loss_fn(reconstructed_a, a_tensor).squeeze().numpy()
            loss_d = loss_fn(reconstructed_d, d_tensor).squeeze().numpy()
        loss_end = time.time()
        total_loss_calc_time += (loss_end - loss_start)
            
        # Anomaly Detection
        anomaly_start = time.time()
        threshold_a = np.percentile(loss_a, trend_percentile)
        trend_anomaly_indices_raw = np.where(loss_a > threshold_a)[0]
        trend_anomaly_indices = filter_anomaly_groups(trend_anomaly_indices_raw, min_group_size=3)
        
        threshold_d = np.percentile(loss_d, other_percentile)
        other_anomaly_indices = np.where(loss_d >= threshold_d)[0]
        anomaly_end = time.time()
        total_anomaly_detection_time += (anomaly_end - anomaly_start)
        
        for idx in trend_anomaly_indices:
            chunk_anomalies.append({
                'placeId': place_id, 'date': dates.iloc[idx].strftime('%Y-%m-%d'),
                'view': time_series[idx], 'anomaly_type': 'trend_anomaly'
            })
        for idx in other_anomaly_indices:
            chunk_anomalies.append({
                'placeId': place_id, 'date': dates.iloc[idx].strftime('%Y-%m-%d'),
                'view': time_series[idx], 'anomaly_type': 'point_anomaly'
            })
    
    # Return anomalies and timing info
    timing_info = {
        'data_prep_time': total_data_prep_time,
        'scaling_time': total_scaling_time,
        'decomposition_time': total_decomposition_time,
        'loss_calc_time': total_loss_calc_time,
        'anomaly_detection_time': total_anomaly_detection_time,
        'num_places': len(chunk_of_data)
    }
            
    return chunk_anomalies, timing_info

# ---- HÀM CHÍNH ĐÃ ĐƯỢC TỐI ƯU HÓA ----
def inference_on_places_optimized(data_path, model_path):
    total_start_time = time.time()
    
    # --- Bước 1: Đọc và chuẩn bị dữ liệu MỘT LẦN ---
    prep_start = time.time()
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['placeId', 'date']).reset_index(drop=True)
    num_places = df['placeId'].nunique()
    
    print(f"Running optimized inference on all {num_places} places...")
    
    grouped_data = list(df.groupby('placeId'))
    prep_end = time.time()

    # --- THAY ĐỔI 2: CHIA DỮ LIỆU THÀNH CÁC "CHUNK" CHO TỪNG WORKER ---
    n_jobs = os.cpu_count() or 1 # Tự động lấy số nhân CPU
    chunk_size = int(np.ceil(len(grouped_data) / n_jobs))
    chunks = [grouped_data[i:i + chunk_size] for i in range(0, len(grouped_data), chunk_size)]
    
    print(f"Data split into {len(chunks)} chunks for {n_jobs} parallel workers.")

    # --- Xử lý song song ---
    processing_start_time = time.time()
    trend_percentile = 97.0
    other_percentile = 98.0
    
    # verbose=10 để in ra tiến trình xử lý
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_chunk)(chunk, model_path, trend_percentile, other_percentile)
        for chunk in chunks
    )
    
    # --- Gom kết quả và timing ---
    all_anomalies_list = []
    total_timing = {
        'data_prep_time': 0,
        'scaling_time': 0,
        'decomposition_time': 0,
        'loss_calc_time': 0,
        'anomaly_detection_time': 0,
        'total_places': 0
    }
    
    for chunk_anomalies, timing_info in results:
        all_anomalies_list.extend(chunk_anomalies)
        total_timing['data_prep_time'] += timing_info['data_prep_time']
        total_timing['scaling_time'] += timing_info['scaling_time']
        total_timing['decomposition_time'] += timing_info['decomposition_time']
        total_timing['loss_calc_time'] += timing_info['loss_calc_time']
        total_timing['anomaly_detection_time'] += timing_info['anomaly_detection_time']
        total_timing['total_places'] += timing_info['num_places']
    
    processing_end_time = time.time()
    
    print("\n--- Inference complete ---")
    total_execution_time = processing_end_time - total_start_time
    print(f"Total execution time for {num_places} places: {total_execution_time:.2f} seconds.")
    
    print("\n--- Timing Breakdown ---")
    print(f"  - Data Loading & Grouping: {prep_end - prep_start:.2f} seconds")
    print(f"  - Parallel Processing:     {processing_end_time - processing_start_time:.2f} seconds")
    
    print("\n--- Average timing per place (ms) ---")
    if total_timing['total_places'] > 0:
        avg_data_prep = (total_timing['data_prep_time'] / total_timing['total_places']) * 1000
        avg_scaling = (total_timing['scaling_time'] / total_timing['total_places']) * 1000
        avg_decomposition = (total_timing['decomposition_time'] / total_timing['total_places']) * 1000
        avg_loss_calc = (total_timing['loss_calc_time'] / total_timing['total_places']) * 1000
        avg_anomaly_detection = (total_timing['anomaly_detection_time'] / total_timing['total_places']) * 1000
        
        print(f"  - Data Preparation:    {avg_data_prep:.4f} ms")
        print(f"  - Data Scaling:        {avg_scaling:.4f} ms")
        print(f"  - Decomposition:       {avg_decomposition:.4f} ms")
        print(f"  - Loss Calculation:    {avg_loss_calc:.4f} ms")
        print(f"  - Anomaly Detection:   {avg_anomaly_detection:.4f} ms")
    
    # --- Lưu kết quả ---
    output_dir = 'inference_results_all_100000placeIds'
    os.makedirs(output_dir, exist_ok=True)
    if all_anomalies_list:
        anomalies_df = pd.DataFrame(all_anomalies_list)
        anomalies_df = anomalies_df.drop_duplicates().sort_values(['placeId', 'date'])
        results_path = os.path.join(output_dir, 'detected_anomalies_detailed.csv')
        anomalies_df.to_csv(results_path, index=False)
        print(f"\nSuccessfully saved {len(anomalies_df)} detected anomalies to: {results_path}")
    else:
        print("\nNo anomalies were detected.")

# ---- KHỐI CHẠY CHÍNH ----
if __name__ == "__main__":
    DATA_PATH = 'data/augmented_data_100000.csv'
    MODEL_PATH = 'saved_models/autoencoder_model.pth'
    
    # THAY ĐỔI 3: Di chuyển phần tải model ra khỏi hàm process_chunk
    # và chỉ tải 1 lần ở main process để truyền vào hàm
    # (Hàm load_model cũ của bạn đã được tích hợp vào trong process_chunk)
    
    for path in [DATA_PATH, MODEL_PATH]:
        if not os.path.exists(path):
            print(f"Error: Path not found at {path}")
            sys.exit(1)
    
    try:
        # Thay vì truyền model, chúng ta chỉ cần truyền model_path
        inference_on_places_optimized(DATA_PATH, MODEL_PATH)
        print("\nProcess finished successfully!")
    except Exception as e:
        print(f"\nAn error occurred during inference: {e}")
        import traceback
        traceback.print_exc()