import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
import pywt

def get_max_wavelet_level(data_len, wavelet='db4'):
    """
    Tính toán cấp độ phân rã wavelet tối đa cho phép để tránh lỗi.
    """
    filt_len = pywt.Wavelet(wavelet).dec_len
    return pywt.dwt_max_level(data_len, filt_len)

def stl_decomposition(time_series: np.ndarray, period: int = 7, robust: bool = True):
    """
    Phân rã chuỗi thời gian bằng STL.
    """
    series = pd.Series(time_series)
    stl_result = STL(series, period=period, robust=robust).fit()
    resid = stl_result.resid.bfill().ffill()
    return resid.values

def dwt_transform(data: np.ndarray, wavelet='db4', level=None):
    """
    Áp dụng DWT. Tự động xác định level nếu không được cung cấp.
    """
    if level is None:
        level = get_max_wavelet_level(len(data), wavelet)
    
    # Đảm bảo level không quá lớn, ít nhất là 1
    level = max(1, level)
    
    coeffs = pywt.wavedec(data, wavelet, level=level)
    return coeffs

def inverse_dwt_transform(coeffs, wavelet='db4'):
    """
    Tái tạo tín hiệu từ các hệ số wavelet.
    """
    return pywt.waverec(coeffs, wavelet)

def pad_wavelet_coeffs_universal(all_coeffs, universal_max_len):
    """
    Đệm tất cả các hệ số wavelet về CÙNG MỘT độ dài tối đa phổ quát.
    """
    padded_coeffs_list = []
    for coeffs_per_ts in all_coeffs:
        padded_coeffs_for_ts = []
        for c in coeffs_per_ts:
            pad_length = universal_max_len - len(c)
            padded_c = np.pad(c, (0, pad_length), mode='constant', constant_values=0)
            padded_coeffs_for_ts.append(padded_c)
        padded_coeffs_list.append(padded_coeffs_for_ts)
    return padded_coeffs_list