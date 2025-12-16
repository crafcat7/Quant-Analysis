# -*- coding: utf-8 -*-
"""
Charts module for generating visualization for quantization analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

def _set_plot_style():
    """Helper to set plot fonts suitable for English."""
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

def plot_accuracy_fit(original_data, quantized_data, output_dir='output', save=False, show=True):
    """
    Generate accuracy fit line plot.

    Args:
        original_data: Original data array
        quantized_data: Quantized data array
        output_dir: Directory to save output plots
        save: Save image
        show: Show image

    Returns:
        dict: Dictionary containing fit statistics
    """
    if original_data is None or quantized_data is None:
        raise ValueError("Please provide data")

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Ensure data is flat
    original_flat = original_data.flatten()
    quantized_flat = quantized_data.flatten()

    # Calculate linear regression parameters
    slope, intercept, r_value, p_value, std_err = stats.linregress(original_flat, quantized_flat)
    r_squared = r_value ** 2

    # Generate fit line data
    x_range = np.array([np.min(original_flat), np.max(original_flat)])
    y_fit = slope * x_range + intercept

    _set_plot_style()
    fig = plt.figure(figsize=(10, 8))

    # Plot scatter (sample if data is too large)
    if len(original_flat) > 10000:
        sample_idx = np.random.choice(len(original_flat), 10000, replace=False)
        plt.scatter(original_flat[sample_idx], quantized_flat[sample_idx],
                    alpha=0.5, s=1, color='blue', label='Sampled Data')
    else:
        plt.scatter(original_flat, quantized_flat,
                    alpha=0.5, s=1, color='blue', label='Data Points')

    # Plot ideal line (y=x)
    plt.plot(x_range, x_range, 'r--', linewidth=2, label='Ideal Fit (y=x)')

    # Plot actual fit line
    plt.plot(x_range, y_fit, 'g-', linewidth=2,
             label=f'Fit: y = {slope:.4f}x + {intercept:.4f}')

    # Labels and Title
    plt.xlabel('Original Value')
    plt.ylabel('Quantized Value')
    plt.title(f'Accuracy Fit (R² = {r_squared:.6f})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Calculate metrics for stats box
    from quantize.analyzer.metrics import calculate_error_metrics
    metrics = calculate_error_metrics(original_data, quantized_data)
    snr = metrics.get('snr', float('inf'))
    psnr = metrics.get('psnr', float('inf'))

    stats_text = (f'SNR: {snr:.2f} dB\n'
                  f'PSNR: {psnr:.2f} dB\n'
                  f'Slope: {slope:.6f}\n'
                  f'Intercept: {intercept:.6f}\n'
                  f'R²: {r_squared:.6f}')

    fig.tight_layout(rect=(0, 0.18, 1, 0.98))
    plt.figtext(0.02, 0.03, stats_text, bbox=dict(facecolor='white', alpha=0.8))

    if save:
        save_path = os.path.join(output_dir, 'accuracy_fit.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'snr': snr,
        'psnr': psnr
    }

def plot_histogram_comparison(original_data, quantized_data, output_dir='output', bins=50, save=False, show=True):
    """
    Generate histogram comparison between original and quantized data.

    Args:
        original_data: Original data array
        quantized_data: Quantized data array
        output_dir: Directory to save output plots
        bins: Number of histogram bins
        save: Save image
        show: Show image
    """
    if original_data is None or quantized_data is None:
        raise ValueError("Please provide data")

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    original_flat = original_data.flatten()
    quantized_flat = quantized_data.flatten()

    _set_plot_style()
    plt.figure(figsize=(12, 6))

    # Determine common bin range
    data_min = min(np.min(original_flat), np.min(quantized_flat))
    data_max = max(np.max(original_flat), np.max(quantized_flat))

    # Plot Original
    plt.subplot(1, 2, 1)
    plt.hist(original_flat, bins=bins, range=(data_min, data_max),
             alpha=0.7, color='blue', edgecolor='black')
    plt.title('Original')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)

    orig_mean = np.mean(original_flat)
    orig_std = np.std(original_flat)
    plt.text(0.05, 0.95, f'Mean: {orig_mean:.4f}\nStd Dev: {orig_std:.4f}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))

    # Plot Quantized
    plt.subplot(1, 2, 2)
    plt.hist(quantized_flat, bins=bins, range=(data_min, data_max),
             alpha=0.7, color='green', edgecolor='black')
    plt.title('Quantized')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)

    quant_mean = np.mean(quantized_flat)
    quant_std = np.std(quantized_flat)
    plt.text(0.05, 0.95, f'Mean: {quant_mean:.4f}\nStd Dev: {quant_std:.4f}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))

    plt.suptitle('Histogram Comparison')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    if save:
        save_path = os.path.join(output_dir, 'histogram_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

def plot_scatter_comparison(original_data, quantized_data, output_dir='output', save=False, show=True, sample_size=None):
    """
    Generate scatter plot comparison.

    Args:
        original_data: Original data array
        quantized_data: Quantized data array
        output_dir: Directory to save output plots
        save: Save image
        show: Show image
        sample_size: Number of samples to use (None for all)
    """
    if original_data is None or quantized_data is None:
        raise ValueError("Please provide data")

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    original_flat = original_data.flatten()
    quantized_flat = quantized_data.flatten()

    # Sampling
    if sample_size is not None and len(original_flat) > sample_size:
        sample_idx = np.random.choice(len(original_flat), sample_size, replace=False)
        original_flat = original_flat[sample_idx]
        quantized_flat = quantized_flat[sample_idx]
    elif len(original_flat) > 50000:  # Default sampling for large datasets
        sample_size = 50000
        sample_idx = np.random.choice(len(original_flat), sample_size, replace=False)
        original_flat = original_flat[sample_idx]
        quantized_flat = quantized_flat[sample_idx]

    _set_plot_style()
    fig = plt.figure(figsize=(10, 8))

    plt.scatter(original_flat, quantized_flat, alpha=0.5, s=1, color='blue')

    min_val = min(np.min(original_flat), np.min(quantized_flat))
    max_val = max(np.max(original_flat), np.max(quantized_flat))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

    data_count = f" (Sampled {len(original_flat)} points)" if sample_size is not None else ""
    plt.xlabel('Original Value')
    plt.ylabel('Quantized Value')
    plt.title(f'Scatter Plot Comparison{data_count}')
    plt.grid(True, linestyle='--', alpha=0.7)

    if save:
        save_path = os.path.join(output_dir, 'scatter_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.tight_layout()
        plt.show()
    else:
        plt.close()

def plot_error_distribution(original_data, quantized_data, output_dir='output', bins=50, save=False, show=True):
    """
    Generate error distribution plot.

    Args:
        original_data: Original data array
        quantized_data: Quantized data array
        output_dir: Directory to save output plots
        bins: Number of histogram bins
        save: Save image
        show: Show image

    Returns:
        dict: Dictionary containing error statistics
    """
    if original_data is None or quantized_data is None:
        raise ValueError("Please provide data")

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    error = original_data - quantized_data
    error_flat = error.flatten()

    # Statistics
    error_mean = np.mean(error_flat)
    error_std = np.std(error_flat)
    error_min = np.min(error_flat)
    error_max = np.max(error_flat)
    error_rms = np.sqrt(np.mean(error_flat ** 2))

    _set_plot_style()
    fig = plt.figure(figsize=(10, 8))

    n, bin_edges, patches = plt.hist(error_flat, bins=bins,
                                    alpha=0.7, color='red', edgecolor='black')

    # Normal distribution fit
    x = np.linspace(error_min, error_max, 1000)
    p = stats.norm.pdf(x, error_mean, error_std)
    plt.plot(x, p * len(error_flat) * (bin_edges[1] - bin_edges[0]),
             'k--', linewidth=2, label='Normal Fit')

    # Lines for Mean and Std Dev
    plt.axvline(x=error_mean, color='green', linestyle='-', linewidth=2,
                label=f'Mean: {error_mean:.6f}')
    plt.axvline(x=error_mean + error_std, color='orange', linestyle='--', linewidth=1.5,
                label=f'+1σ: {error_mean + error_std:.6f}')
    plt.axvline(x=error_mean - error_std, color='orange', linestyle='--', linewidth=1.5,
                label=f'-1σ: {error_mean - error_std:.6f}')

    plt.xlabel('Error (Original - Quantized)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    stats_text = (
        f'Mean Error: {error_mean:.6f}\n'
        f'Std Dev Error: {error_std:.6f}\n'
        f'Min Error: {error_min:.6f}\n'
        f'Max Error: {error_max:.6f}\n'
        f'RMSE: {error_rms:.6f}'
    )

    fig.tight_layout(rect=(0, 0.18, 1, 0.98))
    plt.figtext(0.02, 0.03, stats_text, bbox=dict(facecolor='white', alpha=0.8))

    if save:
        save_path = os.path.join(output_dir, 'error_distribution.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        'mean_error': error_mean,
        'std_error': error_std,
        'min_error': error_min,
        'max_error': error_max,
        'rmse': error_rms
    }

def plot_axis_metrics(original_data, quantized_data, output_dir='output', save=False, show=True, sample_limit=512):
    if original_data is None or quantized_data is None:
        raise ValueError("Please provide data")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    diff = original_data - quantized_data
    nd = diff.ndim
    results = []
    _set_plot_style()
    for i in range(nd):
        x = np.moveaxis(diff, i, 0)
        s0 = x.shape[0]
        x2 = x.reshape(s0, -1)
        abs_x2 = np.abs(x2)
        mae_axis = np.mean(abs_x2, axis=1)
        mse_axis = np.mean(x2 ** 2, axis=1)
        rmse_axis = np.sqrt(mse_axis)
        if s0 > sample_limit:
            step = int(np.ceil(s0 / sample_limit))
            idx = np.arange(0, s0, step)
        else:
            idx = np.arange(s0)
        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.plot(idx, mae_axis[idx], color='tab:blue')
        ax1.set_ylabel('MAE')
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax2 = fig.add_subplot(3, 1, 2)
        ax2.plot(idx, mse_axis[idx], color='tab:orange')
        ax2.set_ylabel('MSE')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax3 = fig.add_subplot(3, 1, 3)
        ax3.plot(idx, rmse_axis[idx], color='tab:green')
        ax3.set_ylabel('RMSE')
        ax3.set_xlabel(f'Axis {i} Index')
        ax3.grid(True, linestyle='--', alpha=0.6)
        fig.suptitle(f'Axis {i} Metrics (size {s0})')
        mae_mean = float(np.mean(mae_axis))
        mae_min = float(np.min(mae_axis))
        mae_max = float(np.max(mae_axis))
        mse_mean = float(np.mean(mse_axis))
        mse_min = float(np.min(mse_axis))
        mse_max = float(np.max(mse_axis))
        rmse_mean = float(np.mean(rmse_axis))
        rmse_min = float(np.min(rmse_axis))
        rmse_max = float(np.max(rmse_axis))
        stats_text = (
            f"Axis {i} size: {s0}\n"
            f"MAE mean {mae_mean:.6f}, min {mae_min:.6f}, max {mae_max:.6f}\n"
            f"MSE mean {mse_mean:.10f}, min {mse_min:.10f}, max {mse_max:.10f}\n"
            f"RMSE mean {rmse_mean:.6f}, min {rmse_min:.6f}, max {rmse_max:.6f}"
        )
        fig.tight_layout(rect=(0, 0.18, 1, 0.98))
        plt.figtext(0.02, 0.03, stats_text, bbox=dict(facecolor='white', alpha=0.8))
        if save:
            save_path = os.path.join(output_dir, f'axis_metrics_axis_{i}.png')
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)
        results.append({
            'axis': i,
            'size': int(s0),
            'mae': mae_axis,
            'mse': mse_axis,
            'rmse': rmse_axis
        })
    return {'axis_metrics': results}
