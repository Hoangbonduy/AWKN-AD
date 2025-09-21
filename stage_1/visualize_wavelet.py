import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from sklearn.preprocessing import RobustScaler
from STL import stl_decomposition

def analyze_wavelet_for_place_id(data_path, place_id, wavelet='db4', period=7):
    """
    Hàm thực hiện toàn bộ pipeline phân tích và trực quan hóa wavelet cho một placeId.
    """
    # --- Bước 1: Tải và Lọc Dữ liệu ---
    try:
        df = pd.read_csv(data_path)
        place_df = df[df['placeId'] == place_id].copy()
        if place_df.empty:
            print(f"Lỗi: Không tìm thấy dữ liệu cho placeId: {place_id}")
            return
        time_series = place_df['view'].values
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file dữ liệu tại: {data_path}")
        return

    print(f"Đã tải thành công chuỗi thời gian cho placeId: {place_id}, độ dài: {len(time_series)}")

    # --- Bước 2: Tiền xử lý và Phân rã STL ---
    scaler = RobustScaler()
    time_series_scaled = scaler.fit_transform(time_series.reshape(-1, 1)).flatten()

    # --- Bước 3: Áp dụng Phép Biến Đổi Wavelet (DWT) ---
    max_level = pywt.dwt_max_level(len(time_series_scaled), pywt.Wavelet(wavelet).dec_len)
    level_to_use = min(4, max_level) # Giới hạn ở level 4 để dễ nhìn, hoặc mức tối đa cho phép

    coeffs = pywt.wavedec(time_series_scaled, wavelet, level=level_to_use)

    # cA4, cD4, cD3, cD2, cD1 (ví dụ với level=4)
    print(f"Đã áp dụng DWT với wavelet '{wavelet}' và {level_to_use} cấp độ.")
    print("Số lượng hệ số ở mỗi cấp độ:")
    for i, c in enumerate(coeffs):
        level_name = f"Approximation (cA{level_to_use})" if i == 0 else f"Detail (cD{level_to_use - i + 1})"
        print(f"  - {level_name}: {len(c)} hệ số")

    # --- Bước 4: Trực quan hóa Kết quả ---
    fig, axs = plt.subplots(len(coeffs) + 2, 1, figsize=(18, 14), sharex=False)
    fig.suptitle(f'Phân Tích Wavelet cho Tín Hiệu Residual (d) - PlaceId: {place_id}', fontsize=16)

    # Biểu đồ 1: Chuỗi thời gian gốc
    axs[0].plot(time_series, color='blue', label='Chuỗi Thời Gian Gốc')
    axs[0].set_title('Bước 1: Chuỗi Thời Gian Ban Đầu', fontsize=12)
    axs[0].set_ylabel('View Count')
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # Các biểu đồ tiếp theo: Các hệ số wavelet
    for i, c in enumerate(coeffs):
        ax = axs[i + 2]
        level_name = f"Approximation (cA{level_to_use})" if i == 0 else f"Detail (cD{level_to_use - i + 1})"
        ax.stem(c, linefmt='-', markerfmt=' ', basefmt=" ")
        ax.set_title(f'Bước 3: Hệ số Wavelet - Cấp độ {level_name}', fontsize=12)
        ax.set_ylabel('Wavelet Coefficients')
        ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Lưu và hiển thị
    output_filename = f'test_layers/wavelet_analysis_placeId_{place_id}.png'
    plt.savefig(output_filename)
    print(f"\nĐã lưu biểu đồ phân tích vào file: {output_filename}")
    plt.show()


if __name__ == "__main__":
    # --- THAY ĐỔI CÁC THAM SỐ NÀY ĐỂ PHÂN TÍCH ---
    DATA_PATH = 'data/cleaned_data_no_zero_periods_filtered.csv'
    
    # Lấy một placeId bất kỳ từ dữ liệu để phân tích
    # Bạn có thể thay bằng placeId cụ thể mà bạn muốn xem
    try:
        df = pd.read_csv(DATA_PATH)
        if not df.empty:
            PLACE_ID_TO_ANALYZE = df['placeId'].unique()[0] # Ví dụ: lấy địa điểm thứ 1
            analyze_wavelet_for_place_id(DATA_PATH, place_id=PLACE_ID_TO_ANALYZE)
        else:
            print("File dữ liệu trống.")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file dữ liệu tại: {DATA_PATH}")