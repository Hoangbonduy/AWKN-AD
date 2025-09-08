import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    # --- Tải dữ liệu ---
    # 1. Dữ liệu gốc
    df_raw = pd.read_csv('data/place.csv', header=None, usecols=[1, 2], names=['date', 'view'], parse_dates=['date'])
    df_raw.set_index('date', inplace=True)
    x = df_raw['view']

    # 2. Hệ số xấp xỉ và chi tiết từ file KAMA của bạn
    a = pd.read_csv('kama_approximation_placeId_4624474044569362538.csv', header=None).squeeze("columns")
    d = pd.read_csv('kama_detail_coeffs_placeId_4624474044569362538.csv', header=None).squeeze("columns")
    
    # Gán index thời gian để trực quan hóa cho đúng
    a.index = x.index
    d.index = x.index

    # --- Xử lý giá trị NaN (do KAMA cần thời gian khởi động) ---
    # Bỏ các giá trị NaN ở đầu chuỗi để các phép tính thống kê chính xác
    x_cleaned = x.dropna()
    a_cleaned = a.dropna()
    d_cleaned = d.dropna()
    
    # Căn chỉnh lại độ dài sau khi bỏ NaN
    common_index = x_cleaned.index.intersection(a_cleaned.index).intersection(d_cleaned.index)
    x_final = x_cleaned[common_index]
    a_final = a_cleaned[common_index]
    d_final = d_cleaned[common_index]

    # --- Tính toán các Metric Định lượng ---
    mean_d = d_final.mean()
    var_x = x_final.var()
    var_d = d_final.var()
    var_a = a_final.var()
    detail_energy_ratio = var_d / var_x if var_x > 0 else 0
    energy_preservation_ratio = (var_a + var_d) / var_x if var_x > 0 else 0

    # --- Trực quan hóa ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    plt.style.use('seaborn-v0_8-whitegrid')

    axes[0].plot(x_final, color='black', linewidth=1.5, label='Dữ liệu gốc `view` (x)')
    axes[0].set_title('1. Phân tích Phân rã bằng KAMA', fontsize=16, fontweight='bold')
    axes[0].set_ylabel('Lượt xem (Views)', fontsize=12)
    axes[0].legend(loc='upper left')

    axes[1].plot(a_final, color='royalblue', linewidth=1.8, label='Xấp xỉ `a` (từ KAMA)')
    axes[1].set_title('2. Thành phần Xấp xỉ `a` (Xu hướng Thích ứng)', fontsize=14)
    axes[1].set_ylabel('Giá trị', fontsize=12)
    axes[1].legend(loc='upper left')

    axes[2].plot(d_final, color='teal', linewidth=1.0, label='Chi tiết `d` (Phần dư)')
    axes[2].axhline(0, color='red', linestyle='--', linewidth=1.2)
    axes[2].set_title('3. Thành phần Chi tiết `d` (Biến động & Bất thường)', fontsize=14)
    axes[2].set_xlabel('Thời gian', fontsize=12)
    axes[2].set_ylabel('Giá trị', fontsize=12)
    axes[2].legend(loc='upper left')
    
    fig.autofmt_xdate()
    plt.tight_layout(pad=2.0)
    plot_filename = 'kama_final_analysis.png'
    plt.savefig(plot_filename)

    # --- In kết quả Metric ---
    print("="*55)
    print("      KẾT QUẢ ĐÁNH GIÁ ĐỊNH LƯỢNG (KAMA)")
    print("="*55)
    print("\n--- Tiêu chí cho Hệ số Chi tiết (d) ---")
    print(f"1. Trung bình của d: {mean_d:<25.6f} (Lý tưởng: ≈ 0)")
    print(f"2. Tỷ lệ Năng lượng Chi tiết (Var(d)/Var(x)): {detail_energy_ratio:<9.4f} (Lý tưởng: Càng nhỏ càng tốt)")
    print("\n--- Tiêu chí Toàn cục ---")
    print(f"3. Tỷ lệ Bảo toàn Năng lượng ((Var(a)+Var(d))/Var(x)): {energy_preservation_ratio:.4f} (Lý tưởng: ≈ 1.0)")
    print("="*55)
    print(f"\nBiểu đồ phân tích đã được lưu vào file: {plot_filename}")

except FileNotFoundError as e:
    print(f"Lỗi: Không tìm thấy file. Vui lòng đảm bảo các file cần thiết đều có mặt.")
except Exception as e:
    print(f"Đã có lỗi xảy ra: {e}")