import pandas as pd
import matplotlib.pyplot as plt

# --- Tải Dữ liệu ---
# Giữ nguyên phần tải dữ liệu của bạn
# Giả sử bạn có file 'place.csv' đã được tiền xử lý
DATA_PATH = 'data/cleaned_data_no_zero_periods_filtered.csv'
raw_df = pd.read_csv(DATA_PATH)


first_place_id = 4621920615327109068  # Thay đổi placeId này theo nhu cầu
raw_df_one_place = raw_df[raw_df['placeId'] == first_place_id].copy()
raw_df_one_place.set_index('date', inplace=True)


# Đọc hệ số xấp xỉ và chi tiết
approx = pd.read_csv('STL_approximation_placeId_4621920615327109068.csv')
detail = pd.read_csv('STL_detail_coeffs_placeId_4621920615327109068.csv')
original = raw_df_one_place['view']


# --- Trực quan hóa (Phiên bản Cải tiến) ---

# Thêm style để biểu đồ đẹp hơn
plt.style.use('seaborn-v0_8-whitegrid')

fig, axs = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

# 1. Biểu đồ Dữ liệu Gốc
axs[0].plot(original, color='black', linewidth=1.5, label='Dữ liệu Gốc (view)')
axs[0].set_title(f'Phân tích Phân rã cho placeId: {first_place_id}', fontsize=18, fontweight='bold')
axs[0].set_ylabel('Giá trị (Lượt xem)', fontsize=12)
axs[0].legend(loc='upper left')
axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)

# 2. Biểu đồ Hệ số Xấp xỉ (a)
axs[1].plot(approx, color='royalblue', linewidth=1.8, linestyle='-', label='Xấp xỉ `a` (Xu hướng)')
axs[1].set_title('Thành phần Xấp xỉ (s - seasonal)', fontsize=14)
axs[1].set_ylabel('Giá trị', fontsize=12)
axs[1].legend(loc='upper left')
axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)

# 3. Biểu đồ Hệ số Chi tiết (d)
axs[2].plot(detail, color='teal', linewidth=1, label='Chi tiết `d` (Phần dư)')
# Thêm đường zero-line để dễ so sánh
axs[2].axhline(0, color='red', linestyle='--', linewidth=1.2)
axs[2].set_title('Thành phần Chi tiết (d)', fontsize=14)
axs[2].set_xlabel('Thời gian', fontsize=12)
axs[2].set_ylabel('Giá trị', fontsize=12)
axs[2].legend(loc='upper left')
axs[2].grid(True, which='both', linestyle='--', linewidth=0.5)

# Tự động xoay nhãn ngày tháng cho dễ đọc
fig.autofmt_xdate()
plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# Lưu lại file ảnh
output_filename = f'STL_plot_decomposition_placeId_{first_place_id}.png'
plt.savefig(output_filename)

print(f"Đã lưu biểu đồ cải tiến vào file: {output_filename}")

# plt.show() # Bỏ dòng này khi chạy trong môi trường tự động