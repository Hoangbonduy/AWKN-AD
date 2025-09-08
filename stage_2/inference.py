import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import joblib  # Để load scaler
from stage_1.KAMA import kama_decomposition
from sklearn.preprocessing import RobustScaler
from AE import TimeSeriesAutoencoder

def load_model(model_path):
    """Load trained autoencoder model"""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Extract model config
    config = checkpoint['model_config']
    
    # Initialize model with saved config
    model = TimeSeriesAutoencoder(
        input_dim=config['input_dim'],
        kan_out_features=config['kan_out_features'],
        num_experts=config['num_experts'],
        decoder_hidden_dim=config['decoder_hidden_dim']
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Training stats: {checkpoint['training_stats']}")
    
    return model, checkpoint['training_stats']

def load_ground_truth_labels(place_id, labels_dir):
    """Load ground truth anomaly labels for a place"""
    label_file = os.path.join(labels_dir, f'label_{place_id}.csv')
    
    if os.path.exists(label_file):
        labels_df = pd.read_csv(label_file)
        # The CSV has columns 'date', 'view', 'label' where label=1 means anomaly
        labels_df['is_anomaly'] = labels_df['label']
        return labels_df
    else:
        print(f"Warning: No ground truth labels found for place {place_id}")
        return None

def rolling_zscore_anomaly_detection(reconstruction_errors, window_size=30, zscore_threshold=3.0):
    """
    Detect anomalies using rolling Z-score on reconstruction errors
    
    Args:
        reconstruction_errors: Array of reconstruction errors per time step
        window_size: Size of rolling window for computing mean and std
        zscore_threshold: Z-score threshold for anomaly detection
    
    Returns:
        anomaly_scores: Z-scores for each time step
        anomaly_indices: Indices where anomaly score > threshold
        rolling_stats: Dict with rolling means and stds for analysis
    """
    errors = np.array(reconstruction_errors)
    n_points = len(errors)
    
    anomaly_scores = np.zeros(n_points)
    rolling_means = np.zeros(n_points)
    rolling_stds = np.zeros(n_points)
    
    for i in range(n_points):
        # Define window bounds
        start_idx = max(0, i - window_size + 1)
        end_idx = i + 1
        
        # Get window data
        window_errors = errors[start_idx:end_idx]
        
        # Compute rolling statistics
        rolling_mean = np.mean(window_errors)
        rolling_std = np.std(window_errors)
        
        rolling_means[i] = rolling_mean
        rolling_stds[i] = rolling_std
        
        # Compute Z-score
        if rolling_std > 1e-8:  # Avoid division by zero
            anomaly_scores[i] = (errors[i] - rolling_mean) / rolling_std
        else:
            anomaly_scores[i] = 0.0
    
    # Find anomaly indices
    anomaly_indices = np.where(np.abs(anomaly_scores) > zscore_threshold)[0]
    
    rolling_stats = {
        'rolling_means': rolling_means,
        'rolling_stds': rolling_stds,
        'window_size': window_size,
        'threshold': zscore_threshold
    }
    
    return anomaly_scores, anomaly_indices, rolling_stats

def detect_anomalies(reconstruction_loss, window_size=100, zscore_threshold=4.0):
    """Detect anomalies based on reconstruction loss using rolling Z-score"""
    anomaly_scores, anomaly_indices, rolling_stats = rolling_zscore_anomaly_detection(
        reconstruction_loss, window_size, zscore_threshold
    )
    
    # Return compatible format with existing code
    # Threshold is now the Z-score threshold
    return anomaly_indices, zscore_threshold

def plot_results(place_id, time_series, dates, anomaly_indices, ground_truth_labels=None, save_path=None):
    """Plot time series with predicted and ground truth anomalies"""
    plt.figure(figsize=(20, 8))
    
    # Plot original time series
    plt.plot(dates, time_series, 'b-', linewidth=1, alpha=0.7, label='Original Time Series')
    
    # Plot ground truth anomalies (yellow background)
    if ground_truth_labels is not None:
        gt_anomaly_dates = ground_truth_labels[ground_truth_labels['is_anomaly'] == 1]['date']
        for gt_date in gt_anomaly_dates:
            if gt_date in dates.values:
                idx = np.where(dates == gt_date)[0]
                if len(idx) > 0:
                    idx = idx[0]
                    plt.axvspan(dates.iloc[max(0, idx-1)], dates.iloc[min(len(dates)-1, idx+1)], 
                               alpha=0.3, color='yellow', label='Ground Truth Anomaly' if gt_date == gt_anomaly_dates.iloc[0] else "")
    
    # Plot predicted anomalies (red dots)
    if len(anomaly_indices) > 0:
        plt.scatter([dates.iloc[i] for i in anomaly_indices], 
                   [time_series[i] for i in anomaly_indices], 
                   color='red', s=100, zorder=5, label='Ensemble Prediction')
    
    plt.title(f'Test - PlaceId: {place_id}', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('View Count', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    
    plt.close()  # Close the figure to free memory and avoid showing

def inference_on_places(data_path, labels_dir, model_path, num_places=30):
    """Run inference on first num_places from the dataset"""
    
    # Load data
    df = pd.read_csv(data_path)
    place_ids = df['placeId'].unique()[:num_places]
    
    print(f"Running inference on {len(place_ids)} places...")
    
    # Load trained model
    model, training_stats = load_model(model_path)
    
    # Create output directory
    output_dir = 'inference_results'
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    # Timing variables
    total_kama_time = 0
    total_forward_time = 0
    total_detect_time = 0
    total_preprocess_time = 0
    
    for i, place_id in enumerate(place_ids):
        print(f"\nProcessing place {i+1}/{len(place_ids)}: {place_id}")
        
        # Get data for this place
        place_data = df[df['placeId'] == place_id].copy()
        place_data = place_data.sort_values('date').reset_index(drop=True)
        
        time_series = place_data['view'].values
        dates = pd.to_datetime(place_data['date'])
        
        # Load ground truth labels
        ground_truth = load_ground_truth_labels(place_id, labels_dir)
        if ground_truth is not None:
            ground_truth['date'] = pd.to_datetime(ground_truth['date'])
        
        start_time = time.time()
        # Preprocess data using local scaler (như cũ)
        scaler = RobustScaler()
        time_series_scaled = scaler.fit_transform(time_series.reshape(-1, 1)).flatten()
        preprocess_time = time.time() - start_time
        total_preprocess_time += preprocess_time
        
        # KAMA decomposition (TIMED)
        start_time = time.time()
        a_np, d_np = kama_decomposition(time_series_scaled)
        kama_time = time.time() - start_time
        total_kama_time += kama_time
        
        start_time = time.time()
        # Convert to tensors
        a_tensor = torch.FloatTensor(a_np.copy()).unsqueeze(0).unsqueeze(-1)
        d_tensor = torch.FloatTensor(d_np.copy()).unsqueeze(0).unsqueeze(-1)
        
        with torch.no_grad():
            # Reconstruction
            reconstructed_a, reconstructed_d = model(a_tensor, d_tensor)
            
            # Calculate reconstruction loss per time step
            loss_fn = torch.nn.MSELoss(reduction='none')
            loss_a = loss_fn(reconstructed_a, a_tensor).squeeze().numpy()
            loss_d = loss_fn(reconstructed_d, d_tensor).squeeze().numpy()
            total_loss_per_step = loss_a + loss_d
        forward_time = time.time() - start_time
        total_forward_time += forward_time
        
        # Detect anomalies using Rolling Z-score (TIMED)
        start_time = time.time()
        
        # Sử dụng adaptive Z-score threshold dựa trên độ dài chuỗi
        adaptive_threshold = 4.0
        
        # Phát hiện anomalies với threshold động
        anomaly_indices, zscore_threshold = detect_anomalies(
            total_loss_per_step,
            window_size=100,
            zscore_threshold=adaptive_threshold
        )
        
        detect_time = time.time() - start_time
        total_detect_time += detect_time

        print(f"Preprocess: {preprocess_time:.4f}s | KAMA: {kama_time:.4f}s | Forward: {forward_time:.4f}s | Z-score: {detect_time:.4f}s")
        print(f"  Reconstruction errors: min={total_loss_per_step.min():.6f}, max={total_loss_per_step.max():.6f}, mean={total_loss_per_step.mean():.6f}")
        print(f"  Adaptive Z-score threshold: {adaptive_threshold:.2f}")
        print(f"  Anomalies detected: {len(anomaly_indices)} points ({len(anomaly_indices)/len(total_loss_per_step)*100:.2f}%)")

        # Plot results
        plot_path = os.path.join(output_dir, f'test_{i+1}_{place_id}.png')
        plot_results(place_id, time_series, dates, anomaly_indices, ground_truth, plot_path)
        
        # Store results
        results.append({
            'place_id': place_id,
            'num_anomalies_predicted': len(anomaly_indices),
            'anomaly_indices': anomaly_indices,
            'adaptive_threshold': adaptive_threshold,
            'avg_reconstruction_loss': np.mean(total_loss_per_step),
            'anomaly_threshold': zscore_threshold,
            'has_ground_truth': ground_truth is not None,
            'num_ground_truth_anomalies': len(ground_truth[ground_truth['is_anomaly'] == 1]) if ground_truth is not None else 0,
            'kama_time': kama_time,
            'forward_time': forward_time,
            'detect_time': detect_time,
            'preprocess_time': preprocess_time,
            'series_length': len(total_loss_per_step)
        })

    
    # Save results summary
    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, 'inference_summary.csv')
    results_df.to_csv(results_path, index=False)
    
    print(f"\\n=== INFERENCE SUMMARY ===")
    print(f"Total places processed: {len(place_ids)}")
    print(f"Average anomalies per place: {results_df['num_anomalies_predicted'].mean():.2f}")
    print(f"Places with ground truth: {results_df['has_ground_truth'].sum()}")
    print(f"Results saved to: {results_path}")
    
    print(f"\\n=== TIMING SUMMARY ===")
    print(f"Total KAMA time: {total_kama_time:.4f}s | Avg: {total_kama_time/len(place_ids):.4f}s per place")
    print(f"Total Forward time: {total_forward_time:.4f}s | Avg: {total_forward_time/len(place_ids):.4f}s per place")
    print(f"Total Z-score time: {total_detect_time:.4f}s | Avg: {total_detect_time/len(place_ids):.4f}s per place")
    print(f"Total Preprocess time: {total_preprocess_time:.4f}s | Avg: {total_preprocess_time/len(place_ids):.4f}s per place")
    print(f"Total time: {total_kama_time + total_forward_time + total_detect_time + total_preprocess_time:.4f}s")
    
    print(f"\\n=== ANOMALY DETECTION SUMMARY ===")
    print(f"Detection Method: Rolling Z-score with Adaptive Threshold")
    print(f"Adaptive Thresholds Used:")
    for _, row in results_df.iterrows():
        print(f"  Place {row['place_id']}: Length={row['series_length']}, Threshold={row['adaptive_threshold']:.1f}")
    
    print(f"\\nAnomaly Rate Distribution:")
    anomaly_rates = results_df['num_anomalies_predicted'] / results_df['series_length'] * 100
    print(f"  Min: {anomaly_rates.min():.2f}%, Max: {anomaly_rates.max():.2f}%, Mean: {anomaly_rates.mean():.2f}%")

    return results_df

if __name__ == "__main__":
    # Paths
    DATA_PATH = 'data/cleaned_data_no_zero_periods_filtered.csv'
    LABELS_DIR = 'new_labels_2'
    MODEL_PATH = 'saved_models/autoencoder_model.pth'
    
    # Check if files exist
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        exit(1)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        exit(1)
    
    if not os.path.exists(LABELS_DIR):
        print(f"Error: Labels directory not found at {LABELS_DIR}")
        exit(1)
    
    # Run inference
    try:
        results = inference_on_places(DATA_PATH, LABELS_DIR, MODEL_PATH, num_places=30)
        print("Inference completed successfully!")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
