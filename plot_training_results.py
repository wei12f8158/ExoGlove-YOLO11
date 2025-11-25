#!/usr/bin/env python3
"""
Plot training results from results.csv file
Creates a 2x5 grid of plots similar to YOLOv8 training visualization
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

def smooth_curve(y, window_size=5):
    """Smooth the curve using moving average"""
    if len(y) < window_size:
        return y
    # Simple moving average
    smoothed = np.zeros_like(y)
    half_window = window_size // 2
    for i in range(len(y)):
        start = max(0, i - half_window)
        end = min(len(y), i + half_window + 1)
        smoothed[i] = np.mean(y[start:end])
    return smoothed

def plot_training_results(csv_path, output_path=None, smooth_window=5):
    """
    Plot training results from CSV file
    
    Args:
        csv_path: Path to results.csv file
        output_path: Path to save the output image (default: same directory as CSV)
        smooth_window: Window size for smoothing (default: 5)
    """
    # Read CSV file
    data = {}
    epochs = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            for key, value in row.items():
                if key != 'epoch' and key != 'time':
                    if key not in data:
                        data[key] = []
                    try:
                        data[key].append(float(value))
                    except (ValueError, TypeError):
                        data[key].append(np.nan)
    
    epochs = np.array(epochs)
    
    # Define the metrics to plot in 2x5 grid
    # Using exact labels as specified by user
    metrics = [
        # Row 1: Training losses and metrics
        ('train/box_loss', 'train/box_loss', 'Loss'),
        ('train/cls_loss', 'train/cls_loss', 'Loss'),
        ('train/dfl_loss', 'train/dfl_loss', 'Loss'),
        ('metrics/precision(B)', 'metrics/precision (B)', 'Score'),
        ('metrics/recall(B)', 'metrics/recall(B)', 'Score'),
        # Row 2: Validation losses and mAP metrics
        ('val/box_loss', 'val/box_loss', 'Loss'),
        ('val/cls_loss', 'val/cls_loss', 'Loss'),  # Note: user wrote "cIs" but CSV has "cls"
        ('val/dfl_loss', 'val/dfl_loss', 'Loss'),
        ('metrics/mAP50(B)', 'metrics/mAP50(B)', 'Score'),
        ('metrics/mAP50-95(B)', 'metrics/mAP50-95(B)', 'Score'),
    ]
    
    # Create figure with 2 rows and 5 columns
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Training Results', fontsize=16, fontweight='bold')
    
    # Plot each metric
    for idx, (metric, title, ylabel) in enumerate(metrics):
        row = idx // 5
        col = idx % 5
        ax = axes[row, col]
        
        # Get data
        if metric in data:
            y_data = np.array(data[metric])
            
            # Plot original data (blue with circular markers)
            ax.plot(epochs, y_data, 'o-', color='#1f77b4', markersize=3, 
                   linewidth=1.5, label='results', alpha=0.7)
            
            # Plot smoothed data (orange dotted line)
            y_smooth = smooth_curve(y_data, window_size=smooth_window)
            ax.plot(epochs, y_smooth, '--', color='#ff7f0e', linewidth=2, 
                   label='smooth', alpha=0.8)
            
            # Set labels and title
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=8)
            
            # Set y-axis limits with some padding
            y_min, y_max = y_data.min(), y_data.max()
            y_range = y_max - y_min
            if y_range > 0:
                ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
        else:
            ax.text(0.5, 0.5, f'Metric not found:\n{metric}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=11)
    
    plt.tight_layout()
    
    # Save the figure
    if output_path is None:
        csv_file = Path(csv_path)
        output_path = csv_file.parent / 'training_results_plot.png'
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_path}")
    
    # Also show the plot
    plt.show()
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Plot training results from CSV')
    parser.add_argument('--csv', type=str, 
                       default='~/Downloads/train11/results.csv',
                       help='Path to results.csv file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for the plot (default: same directory as CSV)')
    parser.add_argument('--smooth', type=int, default=5,
                       help='Smoothing window size (default: 5)')
    
    args = parser.parse_args()
    
    # Expand user path
    csv_path = Path(args.csv).expanduser()
    
    if not csv_path.exists():
        print(f"❌ Error: CSV file not found: {csv_path}")
        return
    
    print(f"📊 Reading results from: {csv_path}")
    plot_training_results(csv_path, args.output, args.smooth)

if __name__ == '__main__':
    main()

