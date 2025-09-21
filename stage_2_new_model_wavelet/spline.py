import torch
import torch.nn as nn
import torch.nn.functional as F

class BSplineBasis(nn.Module):
    """
    Tính các giá trị B-spline cơ sở dựa trên công thức đệ quy Cox-de Boor.
    Đây là phiên bản đã được sửa lỗi và hoàn thiện.
    """
    def __init__(self, grid, degree):
        super().__init__()
        self.register_buffer('grid', grid)
        self.degree = degree

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Tensor đầu vào, shape [batch_size, sequence_length]

        Returns:
            torch.Tensor: Tensor chứa các giá trị cơ sở, shape [batch_size, sequence_length, num_coeffs]
        """
        # Mở rộng chiều của x và grid để tính toán theo lô
        x = x.unsqueeze(-1)
        grid = self.grid.unsqueeze(0).unsqueeze(0)

        # Công thức đệ quy Cox-de Boor
        # Bắt đầu với spline bậc 0 (hàm hộp)
        basis = ((x >= grid[:, :, :-1]) & (x < grid[:, :, 1:])).float()

        # Tính toán đệ quy cho các bậc cao hơn
        for k in range(1, self.degree + 1):
            # Tính vế đầu tiên của công thức
            term1_num = (x - grid[:, :, :-(k + 1)]) * basis[:, :, :-1]
            term1_den = grid[:, :, k:-1] - grid[:, :, :-(k + 1)]
            # Xử lý trường hợp chia cho 0
            term1_den[term1_den == 0] = 1e-6
            term1 = term1_num / term1_den

            # Tính vế thứ hai của công thức
            term2_num = (grid[:, :, k + 1:] - x) * basis[:, :, 1:]
            term2_den = grid[:, :, k + 1:] - grid[:, :, 1:-k]
            # Xử lý trường hợp chia cho 0
            term2_den[term2_den == 0] = 1e-6
            term2 = term2_num / term2_den
            
            basis = term1 + term2

        return basis

class LearnableSplineExperts(nn.Module):
    """"Từ điển" gồm k chuyên gia B-spline có thể học được."""
    def __init__(self, num_experts=8, grid_size=10, degree=3):
        super().__init__()
        # Tạo lưới (grid) cho các hàm spline
        # Thêm các điểm đệm (padding) vào grid, đây là kỹ thuật chuẩn trong KAN gốc
        grid_edges = torch.linspace(-1, 1, grid_size + 1)
        h = (grid_edges[1] - grid_edges[0])
        grid = torch.cat([
            grid_edges[0] - h * torch.arange(degree, 0, -1),
            grid_edges,
            grid_edges[-1] + h * torch.arange(1, degree + 1)
        ])
        
        self.basis_generator = BSplineBasis(grid, degree)
        
        # Tham số có thể học: các hệ số cho k chuyên gia
        # Số coefficients phải khớp với số basis functions thực tế
        num_coeffs = len(grid) - degree - 1
        # Khởi tạo tốt hơn cho các hệ số
        self.expert_coeffs = nn.Parameter(torch.empty(num_experts, num_coeffs))
        nn.init.xavier_uniform_(self.expert_coeffs)


    def forward(self, x):
        # x shape: [batch_size, sequence_length]
        
        # 1. Tính các giá trị cơ sở B-spline
        # basis_vals shape: [batch_size, sequence_length, num_coeffs]
        basis_vals = self.basis_generator(x)
        
        # 2. Kết hợp với các hệ số chuyên gia để tính đầu ra
        # Dùng einsum để nhân và tính tổng hiệu quả
        # 'b s c, e c -> b s e'
        # b: batch, s: sequence, c: coefficients, e: experts
        expert_outputs = torch.einsum('bsc,ec->bse', basis_vals, self.expert_coeffs)
        
        return expert_outputs

# --- Thử nghiệm với phiên bản đã sửa lỗi ---
print("Chạy thử nghiệm với phiên bản B-spline đã sửa lỗi:")
experts = LearnableSplineExperts(num_experts=8, grid_size=10, degree=3)
input_tensor = torch.randn(128, 50) # Batch 128, chuỗi dài 50
expert_outputs = experts(input_tensor)

# Sẽ in ra: torch.Size([128, 50, 8])
print("Hình dạng đầu ra:", expert_outputs.shape)
# Kiểm tra giá trị đầu ra không phải là NaN
print("Tổng của đầu ra (để kiểm tra NaN):", torch.sum(expert_outputs).item())