# -*- coding: utf-8 -*-
"""
Utility functions for quantization analysis.
"""

import numpy as np
import os
import logging
from typing import List, Tuple, Dict, Any
import csv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_single_data(data):
    """
    Helper function to load data from various sources.

    Args:
        data: Data source (array or file path)

    Returns:
        np.ndarray: Loaded data as numpy array
    """
    if data is None:
        return None

    # If string, try loading as file path
    if isinstance(data, str):
        if os.path.exists(data):
            ext = os.path.splitext(data)[1].lower()
            if ext == '.npy':
                return np.load(data, allow_pickle=False)
            elif ext == '.npz':
                z = np.load(data, allow_pickle=False)
                if 'arr_0' in z:
                    return z['arr_0']
                keys = list(z.keys())
                return z[keys[0]] if keys else np.array([])
            elif ext == '.csv':
                try:
                    return np.loadtxt(data, delimiter=',', dtype=np.float64)
                except:
                    return np.genfromtxt(data, delimiter=',', dtype=np.float64)
            elif ext == '.txt':
                try:
                    return np.loadtxt(data, dtype=np.float64)
                except:
                    return np.genfromtxt(data, dtype=np.float64)
            elif ext in ['.pt', '.pth']:
                try:
                    import torch
                    obj = torch.load(data, map_location='cpu')
                    if isinstance(obj, torch.Tensor):
                        return obj.detach().cpu().numpy()
                    return np.array(obj)
                except Exception as e:
                    raise ValueError(f"Unsupported torch file content: {e}")
            else:
                raise ValueError(f"Unsupported file format: {ext}")
        else:
            raise FileNotFoundError(f"File not found: {data}")

    # If array-like, convert to numpy array
    try:
        import torch
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
    except Exception:
        pass
    try:
        if isinstance(data, np.ndarray):
            return data
        return np.array(data)
    except:
        raise ValueError("Cannot convert input to numpy array")

def load_data(original=None, quantized=None):
    """
    Load and validate original and quantized data.

    Args:
        original: Original data (array or file path)
        quantized: Quantized data (array or file path)

    Returns:
        tuple: (original_data, quantized_data)
    """
    # Load data
    original_data = load_single_data(original)
    quantized_data = load_single_data(quantized)

    # Verify shapes match
    if original_data is not None and quantized_data is not None:
        if original_data.shape != quantized_data.shape:
            raise ValueError(f"Shape mismatch: {original_data.shape} vs {quantized_data.shape}")

    logger.info("Data loaded successfully")
    if original_data is not None:
        logger.info(f"Original: shape {original_data.shape}, dtype {original_data.dtype}")
    if quantized_data is not None:
        logger.info(f"Quantized: shape {quantized_data.shape}, dtype {quantized_data.dtype}")

    return original_data, quantized_data

def run_complete_analysis(original, quantized, output_dir='output'):
    """
    Run the complete analysis workflow.

    Args:
        original: Original data (array or file path)
        quantized: Quantized data (array or file path)
        output_dir: Directory to save output plots

    Returns:
        dict: Dictionary containing analysis results
    """
    # Import here to avoid circular imports
    from quantize.analyzer.main import QuantizationAnalyzer

    analyzer = QuantizationAnalyzer(output_dir)
    return analyzer.run_complete_analysis(original, quantized)

def export_analysis_csv(original_stats: Dict[str, Any], metrics: Dict[str, Any], output_dir: str):
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        combined_path = os.path.join(output_dir, 'analysis_report.csv')
        with open(combined_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['Tensor Analysis'])
            w.writerow(['Metric', 'Value'])
            w.writerow(['Shape', str(original_stats.get('shape'))])
            w.writerow(['Mean', f"{original_stats.get('mean', 0):.6f}"])
            w.writerow(['Median', f"{original_stats.get('median', 0):.6f}"])
            w.writerow(['Standard Deviation', f"{original_stats.get('std', 0):.6f}"])
            w.writerow(['Min', f"{original_stats.get('min', 0):.6f}"])
            w.writerow(['Max', f"{original_stats.get('max', 0):.6f}"])
            w.writerow(['Skewness', f"{original_stats.get('skewness', 0):.6f}"])
            w.writerow(['Kurtosis', f"{original_stats.get('kurtosis', 0):.6f}"])
            w.writerow(['25th Percentile', f"{original_stats.get('percentile_25', 0):.6f}"])
            w.writerow(['75th Percentile', f"{original_stats.get('percentile_75', 0):.6f}"])
            w.writerow(['Interquartile Range (IQR)', f"{original_stats.get('iqr', 0):.6f}"])
            w.writerow(['Zero Crossing Rate', f"{original_stats.get('zero_crossing_rate', 0):.6f}"])
            w.writerow(['Non-zero Elements', f"{original_stats.get('non_zero_count', 0)} ({original_stats.get('non_zero_ratio', 0):.2%})"])
            w.writerow([])
            w.writerow(['Dequantization'])
            w.writerow(['Metric', 'Value'])
            s = metrics.get('snr')
            if s is not None:
                w.writerow(['SNR(dB)', '∞' if s == float('inf') else f"{s:.2f} dB"])
            p = metrics.get('psnr')
            if p is not None:
                w.writerow(['PSNR(dB)', '∞' if p == float('inf') else f"{p:.2f} dB"])
            q = metrics.get('sqnr')
            w.writerow(['SQNR(dB)', '∞' if q == float('inf') else f"{q:.2f} dB"])
            w.writerow(['CosineSimilarity', f"{metrics.get('cosine_similarity', 0):.6f}"])
            w.writerow(['MeanRelativeError', f"{metrics.get('mean_relative_error', 0):.6f}"])
            w.writerow(['R2', f"{metrics.get('r2', 0):.6f}"])
            w.writerow(['MAE', f"{metrics.get('mae', 0):.10f}"])
            w.writerow(['MSE', f"{metrics.get('mse', 0):.10f}"])
            w.writerow(['RMSE', f"{metrics.get('rmse', 0):.10f}"])
            w.writerow(['MaxAbsError', f"{metrics.get('max_abs_error', 0):.10f}"])
            w.writerow(['MinAbsError', f"{metrics.get('min_abs_error', 0):.10f}"])
            w.writerow(['StdAbsError', f"{metrics.get('std_abs_error', 0):.10f}"])
            w.writerow(['WeightElements', f"{metrics.get('weight_elements', 0)}"])
    except Exception:
        pass
def parse_tensor_shape(shape):
    if shape is None:
        return None
    if isinstance(shape, (list, tuple)):
        return tuple(int(x) for x in shape)
    s = str(shape).strip().lower()
    for sep in ['x', '*', ',', ' ']:
        if sep in s:
            parts = [p for p in s.replace('*', 'x').replace(',', 'x').replace(' ', 'x').split('x') if p]
            return tuple(int(p) for p in parts)
    return (int(s),)
