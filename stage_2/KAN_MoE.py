import torch
import torch.nn as nn
from spline import LearnableSplineExperts

class KANMoELayer(nn.Module):
    def __init__(self, in_features, out_features, num_experts=8, grid_size=10, degree=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 1. "Từ điển" chuyên gia dùng chung cho cả lớp
        self.experts = LearnableSplineExperts(num_experts, grid_size, degree)
        
        # 2. "Trọng số Gating" cho mỗi cạnh
        # Mỗi cạnh (i, j) có một vector k trọng số
        self.gating_weights = nn.Parameter(
            torch.randn(in_features, out_features, num_experts)
        )
        # Chuẩn hóa bằng Softmax để ổn định việc học
        self.gating_activation = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: [batch_size, sequence_length, in_features]
        batch_size, seq_len, in_features = x.shape
        
        # Reshape x để áp dụng experts cho từng feature riêng biệt
        # x_reshaped shape: [batch_size * in_features, sequence_length]
        x_reshaped = x.transpose(1, 2).contiguous().view(batch_size * in_features, seq_len)
        
        # 1. Tính toán đầu ra của các chuyên gia
        # expert_outputs shape: [batch_size * in_features, sequence_length, num_experts]
        expert_outputs = self.experts(x_reshaped)
        
        # Reshape về [batch_size, in_features, sequence_length, num_experts]
        expert_outputs = expert_outputs.view(batch_size, in_features, seq_len, -1)
        
        # Transpose để có [batch_size, sequence_length, in_features, num_experts]
        expert_outputs = expert_outputs.transpose(1, 2)
        
        # 2. Áp dụng gating và tính tổng kết quả
        # gating_weights shape: [in_features, out_features, num_experts]
        activated_gates = self.gating_activation(self.gating_weights)
        output = torch.einsum('bsie,ioe->bso', expert_outputs, activated_gates)
        
        return output