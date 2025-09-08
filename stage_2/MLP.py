import torch
import torch.nn as nn

class DecisionHeadMLP(nn.Module):
    """
    Tầng 3: Đầu ra Quyết định (Decision Head).
    
    Đây là một mạng Multi-Layer Perceptron (MLP) nhỏ có nhiệm vụ biến đổi 
    các đặc trưng phức tạp đã được học từ Tầng 2 thành một điểm bất thường 
    (anomaly score) duy nhất cho mỗi điểm thời gian.
    """
    def __init__(self, input_dim=64, hidden_dim=32, output_dim=1):
        """
        Khởi tạo các tầng của MLP.
        
        Args:
            input_dim (int): Số chiều của vector đặc trưng đầu vào (từ Tầng 2). Mặc định là 64.
            hidden_dim (int): Số nơ-ron trong tầng ẩn. Mặc định là 32.
            output_dim (int): Số chiều đầu ra. Mặc định là 1 (điểm bất thường).
        """
        super(DecisionHeadMLP, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Định nghĩa luồng dữ liệu đi qua mạng.
        
        Args:
            x (torch.Tensor): Tensor đặc trưng đầu vào từ Tầng 2. 
                              Shape: [batch_size, sequence_length, input_dim]
                              
        Returns:
            torch.Tensor: Tensor chứa điểm bất thường cho mỗi điểm thời gian.
                          Shape: [batch_size, sequence_length, output_dim]
        """
        return self.network(x)