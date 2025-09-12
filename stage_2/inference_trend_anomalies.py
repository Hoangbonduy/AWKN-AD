import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from stage_1.STL import stl_decomposition
from sklearn.preprocessing import RobustScaler
from stage_2.AE import TimeSeriesAutoencoder

def load_model(model_path):
    """Load trained autoencoder model"""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Extract model config
    config = checkpoint['model_config']
    
    # Initialize model with saved config
    model = TimeSeriesAutoencoder(
        input_dim=config['input_dim'],
        kan_out_features=config['kan_out_features'],
        num_experts=config['num_experts']
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

def percentile_anomaly_detection(reconstruction_errors, percentile=98):
    """
    Detect anomalies using percentile threshold on reconstruction errors
    
    Args:
        reconstruction_errors: Array of reconstruction errors per time step
        percentile: Percentile threshold for anomaly detection (default 98%)
    
    Returns:
        threshold: The percentile threshold value
        anomaly_indices: Indices where reconstruction error > threshold
        stats: Dict with statistics for analysis
    """
    errors = np.array(reconstruction_errors)
    
    # Calculate percentile threshold
    threshold = np.percentile(errors, percentile)
    
    # Find anomaly indices
    anomaly_indices = np.where(errors > threshold)[0]
    
    stats = {
        'threshold': threshold,
        'percentile': percentile,
        'min_error': np.min(errors),
        'max_error': np.max(errors),
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'num_anomalies': len(anomaly_indices),
        'anomaly_rate': len(anomaly_indices) / len(errors) * 100
    }
    
    return threshold, anomaly_indices, stats

def detect_anomalies(reconstruction_loss, percentile=98, min_group_size=3):
    """
    Detect anomalies based on reconstruction loss using percentile threshold,
    filtering out groups with less than min_group_size points
    """
    threshold, raw_anomaly_indices, stats = percentile_anomaly_detection(
        reconstruction_loss, percentile
    )
    
    # Group consecutive anomalies
    anomaly_groups = group_consecutive_indices(raw_anomaly_indices, max_gap=1)
    
    # Filter out groups that are too small
    filtered_indices = []
    for start_idx, end_idx in anomaly_groups:
        group_size = end_idx - start_idx + 1
        if group_size >= min_group_size:
            # Add all indices in this group
            for idx in range(start_idx, end_idx + 1):
                if idx in raw_anomaly_indices:  # Make sure it was originally an anomaly
                    filtered_indices.append(idx)
    
    filtered_indices = np.array(filtered_indices)
    
    return filtered_indices, threshold

def group_consecutive_indices(indices, max_gap=1):
    """
    Group consecutive indices into clusters
    
    Args:
        indices: Array of indices
        max_gap: Maximum gap between indices to be considered in the same group
    
    Returns:
        List of tuples (start_idx, end_idx) for each group
    """
    if len(indices) == 0:
        return []
    
    groups = []
    start = indices[0]
    end = indices[0]
    
    for i in range(1, len(indices)):
        if indices[i] - end <= max_gap + 1:  # Consecutive or within gap
            end = indices[i]
        else:
            groups.append((start, end))
            start = indices[i]
            end = indices[i]
    
    groups.append((start, end))
    return groups

def plot_results(place_id, time_series, dates, anomaly_indices, ground_truth_labels=None, loss_a=None, save_path=None):
    """Plot time series with predicted and ground truth anomalies, and reconstruction loss"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), sharex=True)
    
    # Plot original time series
    ax1.plot(dates, time_series, 'b-', linewidth=1, alpha=0.7, label='Original Time Series')
    
    # Plot predicted anomalies (orange background spans)
    if len(anomaly_indices) > 0:
        anomaly_groups = group_consecutive_indices(anomaly_indices, max_gap=2)
        for i, (start_idx, end_idx) in enumerate(anomaly_groups):
            # Extend the span slightly for better visibility
            start_date = dates.iloc[max(0, start_idx-1)]
            end_date = dates.iloc[min(len(dates)-1, end_idx+1)]
            
            ax1.axvspan(start_date, end_date, 
                       alpha=0.4, color='orange', 
                       label='Predicted Anomaly' if i == 0 else "")
    
    ax1.set_title(f'Time Series - PlaceId: {place_id}', fontsize=16)
    ax1.set_ylabel('View Count', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot reconstruction loss_a
    if loss_a is not None:
        ax2.plot(dates, loss_a, 'g-', linewidth=1, alpha=0.7, label='Reconstruction Loss (A component)')
        
        # Plot predicted anomalies on loss plot (orange background spans)
        if len(anomaly_indices) > 0:
            anomaly_groups = group_consecutive_indices(anomaly_indices, max_gap=2)
            for i, (start_idx, end_idx) in enumerate(anomaly_groups):
                start_date = dates.iloc[max(0, start_idx-1)]
                end_date = dates.iloc[min(len(dates)-1, end_idx+1)]
                
                ax2.axvspan(start_date, end_date, 
                           alpha=0.4, color='orange', 
                           label='Predicted Anomaly' if i == 0 else "")
        
        ax2.set_title(f'Reconstruction Loss (A component) - PlaceId: {place_id}', fontsize=16)
        ax2.set_ylabel('Loss Value', fontsize=12)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
    
    # Set common x-label
    ax2.set_xlabel('Date', fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
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
    output_dir = 'inference_results_trend_anomalies'
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    # Timing variables
    total_kama_time = 0
    total_forward_time = 0
    total_detect_time = 0
    total_preprocess_time = 0

    # Sử dụng 96th percentile threshold
    percentile_threshold = 96.0
    
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

        # STL decomposition (TIMED)
        start_time = time.time()
        a_np, d_np = stl_decomposition(time_series_scaled)
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
            loss_fn = torch.nn.HuberLoss(reduction='none')
            loss_a = loss_fn(reconstructed_a, a_tensor).squeeze().numpy()
            # loss_d = loss_fn(reconstructed_d, d_tensor).squeeze().numpy()
            
            # Ensure loss_a is 1D array with correct length
            if loss_a.ndim == 0:  # scalar case
                loss_a = np.array([loss_a])
            if len(loss_a) == 1 and len(time_series) > 1:
                # If loss_a is single value but time series is longer, repeat it
                loss_a = np.full(len(time_series), loss_a[0])
            # total_loss_per_step = loss_a + loss_d
        forward_time = time.time() - start_time
        total_forward_time += forward_time
        
        # Detect anomalies using 98th percentile threshold (TIMED)
        start_time = time.time()
        
        # Phát hiện anomalies với percentile threshold, loại bỏ nhóm < 3 điểm
        anomaly_indices, threshold_value = detect_anomalies(
            loss_a,
            percentile=percentile_threshold,
            min_group_size=3  # Chỉ giữ lại nhóm có ít nhất 3 điểm liên tiếp
        )
        
        detect_time = time.time() - start_time
        total_detect_time += detect_time

        print(f"Preprocess: {preprocess_time:.4f}s | KAMA: {kama_time:.4f}s | Forward: {forward_time:.4f}s | Percentile: {detect_time:.4f}s")
        print(f"  Reconstruction errors: min={loss_a.min():.6f}, max={loss_a.max():.6f}, mean={loss_a.mean():.6f}")
        print(f"  {percentile_threshold}th percentile threshold: {threshold_value:.6f}")
        print(f"  Anomalies detected (after filtering groups < 3 points): {len(anomaly_indices)} points ({len(anomaly_indices)/len(loss_a)*100:.2f}%)")

        # Plot results
        plot_path = os.path.join(output_dir, f'test_{i+1}_{place_id}.png')
        plot_results(place_id, time_series, dates, anomaly_indices, ground_truth, loss_a, plot_path)
        
        # Store results
        results.append({
            'place_id': place_id,
            'num_anomalies_predicted': len(anomaly_indices),
            'anomaly_indices': anomaly_indices,
            'percentile_threshold': percentile_threshold,
            'threshold_value': threshold_value,
            'avg_reconstruction_loss': np.mean(loss_a),
            'has_ground_truth': ground_truth is not None,
            'num_ground_truth_anomalies': len(ground_truth[ground_truth['is_anomaly'] == 1]) if ground_truth is not None else 0,
            'kama_time': kama_time,
            'forward_time': forward_time,
            'detect_time': detect_time,
            'preprocess_time': preprocess_time,
            'series_length': len(loss_a)
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
    print(f"Total Percentile time: {total_detect_time:.4f}s | Avg: {total_detect_time/len(place_ids):.4f}s per place")
    print(f"Total Preprocess time: {total_preprocess_time:.4f}s | Avg: {total_preprocess_time/len(place_ids):.4f}s per place")
    print(f"Total time: {total_kama_time + total_forward_time + total_detect_time + total_preprocess_time:.4f}s")
    
    print(f"\\n=== ANOMALY DETECTION SUMMARY ===")
    print(f"Detection Method: {percentile_threshold}th Percentile Threshold on Reconstruction Errors")
    print(f"Filtering: Only keep anomaly groups with ≥3 consecutive points")
    print(f"Percentile Thresholds Used:")
    for _, row in results_df.iterrows():
        print(f"  Place {row['place_id']}: Length={row['series_length']}, Threshold={row['threshold_value']:.6f}")
    
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