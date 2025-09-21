# Cần import thư viện này ở đầu file KAMA.py của bạn
from statsmodels.tsa.seasonal import STL
from statsmodels.robust import mad
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import RobustScaler

# --- CÁC HÀM CŨ CỦA BẠN (giữ nguyên) ---
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

def clean_d(d, method='mad', threshold=3.5):
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

    return d_clean

# --- HÀM MỚI SỬ DỤNG STL ---
def stl_decomposition(time_series: np.ndarray, period: int = 28, robust: bool = True):
    """
    Tầng 1: Phân rã chuỗi thời gian bằng phương pháp STL (Seasonal-Trend decomposition using Loess).
    - a: Thành phần Xu hướng (trend).
    - d: Thành phần Phần dư (residual), chứa biến động và bất thường.
    
    STL bền vững hơn với các điểm ngoại lệ và không tạo ra độ trễ như KAMA.
    
    Args:
        time_series (np.ndarray): Chuỗi thời gian đầu vào.
        period (int): Ước tính chu kỳ mùa vụ chính của chuỗi. 
                      Đối với dữ liệu không có tính mùa vụ rõ ràng, đây là tham số 
                      kiểm soát độ "mượt" của đường xu hướng. Giá trị lẻ thường 
                      được ưu tiên.
        robust (bool): Nếu True, sử dụng phiên bản STL mạnh mẽ, ít bị ảnh hưởng 
                       bởi các điểm bất thường.
    """
    # Chuyển sang pandas Series để dùng STL, vì nó xử lý index tốt hơn
    series = pd.Series(time_series)
    
    # Thực hiện phân rã STL
    # period là tham số quan trọng nhất của STL
    stl_result = STL(series, period=period, robust=robust).fit()
    
    # Lấy thành phần xu hướng (trend) và phần dư (residual)
    # a = stl_result.trend
    a = stl_result.trend
    d = stl_result.resid
    
    # STL có thể tạo ra NaN ở đầu và cuối, ta cần xử lý chúng
    # a = a.bfill().ffill()
    a = a.bfill().ffill()
    d = d.bfill().ffill()
    
    return a.values, d.values # Trả về numpy array để nhất quán

def stl_decomposition_2(time_series: np.ndarray, period: int = 29, robust: bool = True):
    # Chuyển sang pandas Series để dùng STL, vì nó xử lý index tốt hơn
    series = pd.Series(time_series)

    # Thực hiện phân rã STL
    # period là tham số quan trọng nhất của STL
    stl_result = STL(series, period=period, robust=robust).fit()

    a = stl_result.trend
    a = a.bfill().ffill()

    new_series = series - a
    new_series = new_series.bfill().ffill()

    return a.values, new_series.values # Trả về numpy array để nhất quán

def stl_decomposition_3(time_series: np.ndarray, period: int = 29, robust: bool = True):
    # Chuyển sang pandas Series để dùng STL, vì nó xử lý index tốt hơn
    series = pd.Series(time_series)

    # Thực hiện phân rã STL
    # period là tham số quan trọng nhất của STL
    stl_result = STL(series, period=period, robust=robust).fit()

    a = stl_result.trend + stl_result.resid
    a = a.bfill().ffill()

    return a.values # Trả về numpy array để nhất quán

def stl_decomposition_4(time_series: np.ndarray, period: int = 29, robust: bool = True):
    # Chuyển sang pandas Series để dùng STL, vì nó xử lý index tốt hơn
    series = pd.Series(time_series)

    # Thực hiện phân rã STL
    # period là tham số quan trọng nhất của STL
    stl_result = STL(series, period=period, robust=robust).fit()

    a = stl_result.trend + stl_result.seasonal
    a = a.bfill().ffill()

    return a.values # Trả về numpy array để nhất quán

# --- Chạy ví dụ ---
# Giả sử bạn có file 'place.csv' đã được tiền xử lý
if __name__ == "__main__":
    DATA_PATH = 'data/cleaned_data_no_zero_periods_filtered.csv'
    
    # Chỉ định placeId cụ thể thay vì dùng chỉ mục
    TARGET_PLACE_ID = 4621920615327109068  # Thay đổi placeId này theo nhu cầu
    
    try:
        df = pd.read_csv(DATA_PATH)
        
        # Lọc dữ liệu theo placeId cụ thể
        if not df.empty:
            print(f"Đang xử lý placeId: {TARGET_PLACE_ID}")
            
            # Kiểm tra xem placeId có tồn tại trong dữ liệu không
            if TARGET_PLACE_ID not in df['placeId'].values:
                print(f"Lỗi: PlaceId {TARGET_PLACE_ID} không tồn tại trong dữ liệu.")
                print(f"Các placeId có sẵn: {sorted(df['placeId'].unique())[:10]}...")  # Hiển thị 10 placeId đầu tiên
                exit()
            
            df_one_place = df[df['placeId'] == TARGET_PLACE_ID].copy()  # Tạo copy để tránh warning

            # Chỉ áp dụng log1p cho cột 'view' (numeric), không phải toàn bộ DataFrame
            df_one_place['view'] = np.log1p(df_one_place['view'].values)

            scaler = RobustScaler()
            df_one_place['view'] = scaler.fit_transform(df_one_place[['view']].values).flatten()

            # Lấy time series array từ DataFrame
            time_series = df_one_place['view'].values

            # Chạy hàm phân rã bằng STL
            # a_values, d_values = stl_decomposition_2(time_series)
            a_values, _ = stl_decomposition(time_series,period=7)

            _ , d_values = stl_decomposition(time_series,period=7)
            # d_values = clean_and_augment_d(d_values)  # Clean và augment detail component

            print("=== KẾT QUẢ CHO 1 ĐỊA ĐIỂM DUY NHẤT (SỬ DỤNG STL) ===")
            
            # Chuyển đổi lại thành pandas Series để xem và lưu file
            a_series = pd.Series(a_values)
            d_series = pd.Series(d_values)

            # a_values, new_series = stl_decomposition_2(time_series)

            print(f"\n--- Kết quả cho placeId: {TARGET_PLACE_ID} ---")
            print("Hệ số Xấp xỉ (a - Chuỗi xu hướng STL):")
            print(a_values[:5])  # In 5 phần tử đầu của numpy array
            
            # Chuyển đổi thành pandas Series để lưu file
            a_series = pd.Series(a_values)
            a_series.to_csv(f'STL_approximation_placeId_{TARGET_PLACE_ID}.csv', index=False, header=['view'])
            
            print("\nHệ số Chi tiết (d - Chuỗi biến động STL):")
            print(d_values[:5])  # In 5 phần tử đầu của numpy array
            
            # Chuyển đổi thành pandas Series để lưu file
            d_series = pd.Series(d_values)
            d_series.to_csv(f'STL_detail_coeffs_placeId_{TARGET_PLACE_ID}.csv', index=False, header=['view'])

            print(f"\nĐã lưu kết quả vào file STL_approximation_placeId_{TARGET_PLACE_ID}.csv và STL_detail_coeffs_placeId_{TARGET_PLACE_ID}.csv")
            print("\n" + "="*50 + "\n")
        else:
            print(f"File {DATA_PATH} trống hoặc không đúng định dạng.")

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file tại đường dẫn '{DATA_PATH}'. Vui lòng kiểm tra lại.")