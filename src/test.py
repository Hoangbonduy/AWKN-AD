import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from sklearn.preprocessing import StandardScaler  # Optional normalize

# Manual KAMA (fix constant issue)
def kama_decomposition(time_series: np.ndarray, length=30, fast=5, slow=50):
    """Manual KAMA với initial SMA, tuned params cho N~640."""
    n = len(time_series)
    if n < length:
        length = n // 2 + 1
    kama = np.full(n, np.nan)
    
    # Initial SMA
    kama[:length-1] = time_series[:length-1]
    kama[length-1] = np.mean(time_series[:length])
    
    for i in range(length, n):
        change = abs(time_series[i] - time_series[i - length])
        volatility = sum(abs(time_series[j] - time_series[j-1]) for j in range(i - length + 1, i + 1))
        er = change / volatility if volatility > 0 else 0.0
        
        sc_fast = 2.0 / (fast + 1)
        sc_slow = 2.0 / (slow + 1)
        sc = (er * (sc_fast - sc_slow) + sc_slow) ** 2
        
        kama[i] = kama[i-1] + sc * (time_series[i] - kama[i-1])
    
    # No ffill, use forward fill only if needed (rare NaN now)
    kama = pd.Series(kama).ffill().bfill().values
    d = time_series - kama
    return kama, d

def decompose_all_places(df: pd.DataFrame):
    """Batch decompose all places."""
    grouped = df.groupby('placeId')['view'].apply(np.array).values
    max_n = max(len(ts) for ts in grouped)
    a_list, d_list = [], []
    place_ids = df['placeId'].unique()
    
    for ts in grouped:
        a, d = kama_decomposition(ts)
        a_pad = np.pad(a, (0, max_n - len(a)), 'constant')
        d_pad = np.pad(d, (0, max_n - len(d)), 'constant')
        a_list.append(a_pad)
        d_list.append(d_pad)
    
    a_batch = torch.FloatTensor(np.array(a_list)).unsqueeze(-1)  # [num_places, max_n, 1]
    d_batch = torch.FloatTensor(np.array(d_list)).unsqueeze(-1)
    return a_batch, d_batch, place_ids, max_n, grouped  # grouped for trim later

# Fourier KAN-MoE proxy (nhanh, thay B-spline; learnable freq)
class FourierKANMoE(nn.Module):
    def __init__(self, in_features=1, out_features=16, num_experts=4):  # Reduce for speed
        super().__init__()
        self.num_experts = num_experts
        # Learnable freq/phase for experts
        self.freqs = nn.Parameter(torch.randn(num_experts, out_features))
        self.phases = nn.Parameter(torch.randn(num_experts, out_features))
        self.ampls = nn.Parameter(torch.ones(num_experts, out_features))
        # Gating
        self.gating = nn.Parameter(torch.randn(in_features, out_features, num_experts))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):  # x [B, seq, in_f=1]
        B, seq, _ = x.shape
        x_flat = x.squeeze(-1)  # [B, seq]
        
        # Create time indices normalized to [0, 1] for Fourier basis
        time_indices = torch.linspace(0, 1, seq, device=x.device).unsqueeze(0).repeat(B, 1)  # [B, seq]
        
        # Compute Fourier basis for each expert
        # self.freqs: [num_experts, out_features]
        # time_indices: [B, seq] -> [B, seq, 1, 1] for broadcasting
        time_expanded = time_indices.unsqueeze(-1).unsqueeze(-1)  # [B, seq, 1, 1]
        freqs_expanded = self.freqs.unsqueeze(0).unsqueeze(0)  # [1, 1, num_experts, out_features]
        phases_expanded = self.phases.unsqueeze(0).unsqueeze(0)  # [1, 1, num_experts, out_features]
        ampls_expanded = self.ampls.unsqueeze(0).unsqueeze(0)  # [1, 1, num_experts, out_features]
        
        # Compute angles: [B, seq, num_experts, out_features]
        angles = 2 * np.pi * freqs_expanded * time_expanded + phases_expanded
        expert_out = ampls_expanded * torch.sin(angles)  # [B, seq, num_experts, out_features]
        
        # Average over experts dimension: [B, seq, out_features]
        expert_out = expert_out.mean(dim=2)
        
        # Gating: [in_features=1, out_features, num_experts] -> simplified for in_features=1
        gates = self.softmax(self.gating.squeeze(0))  # [out_features, num_experts]
        
        # Since we averaged experts, just return expert_out directly
        return expert_out

# MLP same
class DecisionHeadMLP(nn.Module):
    def __init__(self, input_dim=32):  # Reduced 64→32
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)

# Full model with recon decoder
class AWKNAD(nn.Module):
    def __init__(self, out_f=16, num_exp=4):
        super().__init__()
        self.trend_kan = FourierKANMoE(out_features=out_f, num_experts=num_exp)
        self.detail_kan = FourierKANMoE(out_features=out_f, num_experts=num_exp)
        self.mlp = DecisionHeadMLP(out_f * 2)
        # Recon decoder for train
        self.decoder = nn.Sequential(
            nn.Linear(out_f * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, a, d, reconstruct=False):
        feat_a = self.trend_kan(a)
        feat_d = self.detail_kan(d)
        combined = torch.cat([feat_a, feat_d], dim=-1)
        scores = self.mlp(combined)
        if reconstruct:
            recon = self.decoder(combined)
            return scores, recon
        return scores

# Train function (unsupervised recon on normal; assume first 80% normal)
def train_model(model, a_batch, d_batch, num_places=1000, epochs=20, lr=0.001, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    a_batch, d_batch = a_batch.to(device), d_batch.to(device)
    
    # Train on subset for speed
    subset = min(num_places, a_batch.shape[0])
    a_sub = a_batch[:subset]
    d_sub = d_batch[:subset]
    
    start = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        _, recon = model(a_sub, d_sub, reconstruct=True)
        # Recon target: original x = a + d (padded)
        x_target = (a_sub.squeeze(-1) + d_sub.squeeze(-1)).unsqueeze(-1)
        loss = criterion(recon, x_target)
        # + L1 on d for sparsity (anomalies large)
        reg = 0.01 * torch.mean(torch.abs(d_sub.squeeze(-1)))
        total_loss = loss + reg
        total_loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss.item():.4f}')
    
    train_time = time.time() - start
    print(f'Train time: {train_time:.2f}s')
    return model

# Main
if __name__ == "__main__":
    DATA_PATH = 'data/cleaned_data_after_idx30.csv'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    df = pd.read_csv(DATA_PATH)
    num_places = len(df['placeId'].unique())
    print(f'Total places: {num_places}')
    
    # Decompose all (fast O(N))
    start_pre = time.time()
    a_batch, d_batch, place_ids, max_n, orig_lengths = decompose_all_places(df)
    pre_time = time.time() - start_pre
    print(f'Preprocess time: {pre_time:.2f}s')
    
    out_f = 16
    model = AWKNAD(out_f=out_f)
    
    # Train on subset (e.g., 1000 places)
    model = train_model(model, a_batch, d_batch, num_places=1000)
    
    # Inference all (no_grad, batch)
    inf_start = time.time()
    model.eval()
    with torch.no_grad():
        scores_batch = model(a_batch, d_batch)
    inf_time = time.time() - inf_start
    print(f'Inf time for {num_places}: {inf_time:.2f}s')
    
    # Save per place (trim pad)
    results = {}
    for i, pid in enumerate(place_ids):
        len_orig = len(orig_lengths[i])
        scores_trim = scores_batch[i, :len_orig, 0].cpu().numpy()  # Anomaly scores
        results[pid] = scores_trim
        # Threshold example: > mean + 3*std = anomaly
        mean_score = np.mean(scores_trim)
        std_score = np.std(scores_trim)
        anomalies = np.where(scores_trim > mean_score + 3 * std_score)[0]
        print(f'Place {pid}: Anomalies at indices {anomalies[:5]}...')  # Top 5
    
    # Total time
    total_time = pre_time + inf_time  # + train
    print(f'Total time: {total_time:.2f}s ({total_time/60:.2f} min)')
    
    # Save full scores
    scores_df = pd.DataFrame(results).T
    scores_df.to_csv('anomaly_scores_all_places.csv')