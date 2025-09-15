import pandas as pd
import numpy as np
import pandas_ta as ta # Thư viện mới để tính KAMA
from sklearn.preprocessing import RobustScaler
from scipy.stats import zscore
from statsmodels.robust import mad

def clean_and_augment_d(d, method='mad', threshold=3.5, noise_scale=0.5):
    """
    Làm sạch ngoại lệ trong d và tăng cường dữ liệu.
    
    Args:
        d (np.ndarray): Mảng numpy của thành phần phần dư.
        method (str): Phương pháp phát hiện ngoại lệ. 'mad' (mặc định) hoặc 'zscore'.
        threshold (float): Ngưỡng để xác định một điểm là ngoại lệ.
                           Mặc định là 3.5 cho MAD.
        noise_scale (float): Tỷ lệ nhiễu để tăng cường dữ liệu.
        
    Returns:
        d_aug (np.ndarray): Mảng d sau khi đã làm sạch và tăng cường.
    """
    d_clean = d.copy()
    median_d = np.median(d)
    
    if method == 'mad':
        # Sử dụng Median Absolute Deviation (MAD) - Mạnh mẽ hơn Z-score
        # MAD tính toán độ lệch so với trung vị, ít bị ảnh hưởng bởi ngoại lệ.
        # hằng số 0.6745 giúp mad tương đương với độ lệch chuẩn của phân phối chuẩn.
        d_mad = mad(d, c=0.6745)
        
        # Tránh chia cho 0 nếu chuỗi dữ liệu không có biến động
        if d_mad > 1e-8:
            # Tính điểm số dựa trên MAD
            mad_score = np.abs(d - median_d) / d_mad
            outliers = mad_score > threshold
            d_clean = np.where(outliers, median_d, d)
            
    elif method == 'zscore':
        # Giữ lại phương pháp Z-score nếu bạn muốn so sánh
        z_d = zscore(d)
        outliers = np.abs(z_d) > threshold
        d_clean = np.where(outliers, median_d, d)

    # Augmentation: Thêm nhiễu dựa trên độ lệch chuẩn của dữ liệu đã làm sạch
    std_clean = np.std(d_clean)
    noise = np.random.normal(0, noise_scale * std_clean, len(d_clean))
    d_aug = d_clean + noise
    
    return d_aug

def kama_decomposition(time_series: np.ndarray, length=20, fast=2, slow=30):
    """
    Tầng 1: Phân rã chuỗi thời gian bằng Trung bình trượt Thích ứng Kaufman (KAMA).
    - a: Thành phần Xấp xỉ (xu hướng đã được làm mượt một cách thích ứng).
    - d: Thành phần Chi tiết (phần dư, chứa biến động và bất thường).
    
    Args:
        time_series (np.ndarray): Chuỗi thời gian đầu vào.
        length (int): Chu kỳ chính của KAMA.
        fast (int): Chu kỳ cho hằng số làm mượt nhanh nhất.
        slow (int): Chu kỳ cho hằng số làm mượt chậm nhất.
    """
    # Chuyển sang pandas Series để dùng thư viện pandas_ta
    series = pd.Series(time_series)

    # Tính length động nếu cần 
    # dynamic_length = int(get_dynamic_kama_length(series))

    # Tính 'a' (thành phần xấp xỉ) bằng KAMA với các tham số mặc định
    # Sử dụng trực tiếp hàm ta.kama
    a = ta.kama(series, length=length, fast=fast, slow=slow)

    # KAMA sẽ tạo ra giá trị NaN ở đầu chuỗi.
    # Ta điền các giá trị NaN này để đảm bảo chuỗi 'a' có cùng độ dài.
    a = a.bfill().ffill()
    
    # Tính 'd' là phần chênh lệch (phần dư)
    d = series - a
    
    return a.values, d.values # Trả về numpy array để nhất quán

def decompose_by_place_id(dataframe: pd.DataFrame):
    """
    Chạy phân rã bằng KAMA cho từng placeId trong DataFrame.
    """
    results = {}
    for place_id, group in dataframe.groupby('placeId'):
        time_series = group['view'].values # Lấy dữ liệu dưới dạng numpy array
        
        # Gọi hàm phân rã bằng KAMA mới
        a, d = kama_decomposition(time_series)

        d = clean_and_augment_d(d) # Clean và augment detail component

        results[place_id] = {
            'a (approximation)': a,
            'd (detail)': d
        }
    return results

# --- Chạy ví dụ ---
# Giả sử bạn có file 'place.csv' đã được tiền xử lý
if __name__ == "__main__":
    DATA_PATH = 'data/cleaned_data_after_idx30.csv'
    try:
        df = pd.read_csv(DATA_PATH)
        
        # Chỉ lấy placeId đầu tiên để làm ví dụ
        if not df.empty:
            first_place_id = df['placeId'].iloc[0]
            df_one_place = df[df['placeId'] == first_place_id]

            scaler = RobustScaler()
            df_one_place['view'] = scaler.fit_transform(df_one_place[['view']])

            # Chạy hàm phân rã bằng KAMA
            final_results = decompose_by_place_id(df_one_place)

            print("=== KẾT QUẢ CHO 1 ĐỊA ĐIỂM DUY NHẤT (SỬ DỤNG KAMA) ===")
            for place_id, data in final_results.items():
                # Chuyển đổi lại thành pandas Series để xem và lưu file
                a_series = pd.Series(data['a (approximation)'])
                d_series = pd.Series(data['d (detail)'])

                # Clean và augment detail component - trả về numpy array
                d_cleaned = clean_and_augment_d(d_series.values)  

                print(f"\n--- Kết quả cho placeId: {place_id} ---")
                print("Hệ số Xấp xỉ (a - Chuỗi xu hướng KAMA):")
                print(a_series.head())
                a_series.to_csv(f'kama_approximation_placeId_{place_id}.csv', index=False, header=['view'])
                
                print("\nHệ số Chi tiết (d - Chuỗi biến động KAMA):")
                print(d_cleaned[:5])  # In 5 phần tử đầu của numpy array
                
                # Chuyển numpy array thành pandas Series để lưu file
                d_cleaned_series = pd.Series(d_cleaned)
                d_cleaned_series.to_csv(f'kama_detail_coeffs_placeId_{place_id}.csv', index=False, header=['view'])
                
                print(f"\nĐã lưu kết quả vào file kama_approximation_placeId_{place_id}.csv và kama_detail_coeffs_placeId_{place_id}.csv")
                print("\n" + "="*50 + "\n")
        else:
            print(f"File {DATA_PATH} trống hoặc không đúng định dạng.")

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file tại đường dẫn '{DATA_PATH}'. Vui lòng kiểm tra lại.")