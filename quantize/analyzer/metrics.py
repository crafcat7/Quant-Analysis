# -*- coding: utf-8 -*-
"""
Metrics module for quantization analysis.
"""

import numpy as np
from scipy import stats

def analyze_original_tensor(original_data):
    """
    Analyze the original tensor data, calculating various statistical metrics.

    Args:
        original_data: Original data array

    Returns:
        dict: Dictionary containing statistical metrics
    """
    if original_data is None:
        raise ValueError("Please provide original data")

    data_flat = original_data.flatten()

    mean_value = np.mean(data_flat)
    median_value = np.median(data_flat)
    std_value = np.std(data_flat)
    min_value = np.min(data_flat)
    max_value = np.max(data_flat)

    skewness = stats.skew(data_flat)
    kurtosis = stats.kurtosis(data_flat)
    percentile_25 = np.percentile(data_flat, 25)
    percentile_75 = np.percentile(data_flat, 75)
    iqr = percentile_75 - percentile_25

    zero_crossings = np.where(np.diff(np.signbit(data_flat)))[0]
    zero_crossing_rate = len(zero_crossings) / len(data_flat) if len(data_flat) > 0 else 0

    non_zero_count = np.count_nonzero(data_flat)
    non_zero_ratio = non_zero_count / len(data_flat) if len(data_flat) > 0 else 0

    axis_stats = []
    try:
        nd = original_data.ndim
        for i in range(nd):
            x = np.moveaxis(original_data, i, 0)
            s0 = x.shape[0]
            x2 = x.reshape(s0, -1)
            m = np.mean(x2, axis=1)
            sd = np.std(x2, axis=1)
            mi = np.min(x2, axis=1)
            ma = np.max(x2, axis=1)
            nzc = np.count_nonzero(x2, axis=1)
            nzr = np.divide(nzc, x2.shape[1]) if x2.shape[1] > 0 else np.zeros(s0)
            try:
                sk = stats.skew(x2, axis=1)
                ku = stats.kurtosis(x2, axis=1)
            except Exception:
                sk = np.zeros(s0)
                ku = np.zeros(s0)
            zcr = []
            for k in range(s0):
                v = x2[k]
                if v.size == 0:
                    zcr.append(0.0)
                else:
                    z = np.where(np.diff(np.signbit(v)))[0]
                    zcr.append(len(z) / len(v))
            axis_stats.append({
                'axis': i,
                'size': int(s0),
                'mean': m,
                'std': sd,
                'min': mi,
                'max': ma,
                'skewness': sk,
                'kurtosis': ku,
                'non_zero_count': nzc,
                'non_zero_ratio': nzr,
                'zero_crossing_rate': np.array(zcr)
            })
    except Exception:
        axis_stats = []
    return {
        'shape': original_data.shape,
        'mean': mean_value,
        'median': median_value,
        'std': std_value,
        'min': min_value,
        'max': max_value,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'percentile_25': percentile_25,
        'percentile_75': percentile_75,
        'iqr': iqr,
        'zero_crossing_rate': zero_crossing_rate,
        'non_zero_count': non_zero_count,
        'non_zero_ratio': non_zero_ratio,
        'axis_stats': axis_stats
    }

def analyze_distribution(stats, threshold=0.05):
    """
    Analyze the distribution of the original data based on thresholds using pre-calculated statistics.

    Args:
        stats: Dictionary containing statistical metrics (from analyze_original_tensor)
        threshold: Threshold for determining distribution characteristics

    Returns:
        dict: Distribution analysis results
    """
    if not stats:
        return {'distribution_type': 'Unknown (No Data)'}

    skewness = stats.get('skewness', 0)
    kurtosis = stats.get('kurtosis', 0)
    non_zero_ratio = stats.get('non_zero_ratio', 0)

    distribution_type = "Unknown"
    confidence = "Low"

    # Check for Sparsity
    if non_zero_ratio < threshold:
        distribution_type = "Sparse"
        confidence = "High"
    elif non_zero_ratio < 0.3:
        distribution_type = "Sparse-like"
        confidence = "Medium"

    # Check for Gaussian (Normal)
    # Normal: Skewness ~ 0, Kurtosis ~ 0 (Fisher)
    elif abs(skewness) < threshold and abs(kurtosis) < threshold:
        distribution_type = "Gaussian"
        confidence = "High"
    elif abs(skewness) < threshold * 5 and abs(kurtosis) < threshold * 5:
         distribution_type = "Gaussian-like"
         confidence = "Medium"

    # Check for Uniform
    # Uniform: Skewness ~ 0, Kurtosis ~ -1.2
    elif abs(skewness) < threshold and abs(kurtosis + 1.2) < threshold:
        distribution_type = "Uniform"
        confidence = "High"
    elif abs(skewness) < threshold * 5 and abs(kurtosis + 1.2) < threshold * 5:
        distribution_type = "Uniform-like"
        confidence = "Medium"

    # Check for Symmetric
    elif abs(skewness) < threshold:
         distribution_type = "Symmetric"
         confidence = "Medium"

    return {
        'distribution_type': distribution_type,
        'confidence': confidence,
        'metrics': {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'non_zero_ratio': non_zero_ratio
        }
    }


def calculate_error_metrics(original_data, quantized_data):
    """
    Calculate various error metrics between original and quantized data.
    Includes SNR, PSNR, SQNR, MSE, MAE, etc.

    Args:
        original_data: Original data array
        quantized_data: Quantized data array

    Returns:
        dict: Dictionary containing all calculated metrics
    """
    if original_data is None or quantized_data is None:
        raise ValueError("Please provide original and quantized data")

    # Ensure numpy arrays
    original = np.asarray(original_data)
    quantized = np.asarray(quantized_data)

    # Basic error calculations
    diff = original - quantized
    abs_errors = np.abs(diff)
    mse = np.mean(diff ** 2)

    # Calculate metrics
    metrics = {
        'mae': np.mean(abs_errors),
        'mse': mse,
        'rmse': np.sqrt(mse),
        'max_abs_error': np.max(abs_errors),
        'min_abs_error': np.min(abs_errors),
        'std_abs_error': np.std(abs_errors),
    }

    # Calculate SQNR (Signal-to-Quantization Noise Ratio)
    # SQNR = 10 * log10(Signal_Power / Noise_Power)
    # Here Signal Power is mean(original^2) and Noise Power is MSE
    signal_power_sqnr = np.mean(original ** 2)
    if mse > 0:
        metrics['sqnr'] = 10 * np.log10(signal_power_sqnr / mse)
    else:
        metrics['sqnr'] = float('inf')

    # Calculate PSNR (Peak Signal-to-Noise Ratio)
    # PSNR = 10 * log10(Max_Value^2 / MSE)
    max_val = np.max(np.abs(original))
    if mse == 0:
        metrics['psnr'] = float('inf')
    elif max_val == 0:
        metrics['psnr'] = float('nan')
    else:
        metrics['psnr'] = 10 * np.log10((max_val ** 2) / mse)

    # Calculate SNR (Signal-to-Noise Ratio)
    # SNR = 10 * log10(Var(original) / Var(noise))
    # Note: calculate_snr function uses variance, SQNR uses mean square.
    # Reusing the logic from calculate_snr but inlined or called.
    # Let's call calculate_snr to maintain exact consistency if logic differs slightly,
    # but efficient implementation would use variance.
    # calculate_snr uses np.var(original) and np.var(noise)
    signal_var = np.var(original)
    noise_var = np.var(diff)
    if noise_var == 0:
        metrics['snr'] = float('inf')
    else:
        metrics['snr'] = 10 * np.log10(signal_var / noise_var)

    # Calculate Cosine Similarity
    original_flat = original.flatten()
    quantized_flat = quantized.flatten()

    norm_original = np.linalg.norm(original_flat)
    norm_quantized = np.linalg.norm(quantized_flat)

    if norm_original > 0 and norm_quantized > 0:
        metrics['cosine_similarity'] = np.dot(original_flat, quantized_flat) / (norm_original * norm_quantized)
    else:
        metrics['cosine_similarity'] = 0.0

    # Calculate Mean Relative Error
    non_zero_mask = original != 0
    if np.any(non_zero_mask):
        relative_errors = np.abs(diff[non_zero_mask]) / np.abs(original[non_zero_mask])
        metrics['mean_relative_error'] = np.mean(relative_errors)
    else:
        metrics['mean_relative_error'] = 0.0

    # Calculate R-squared (Coefficient of Determination)
    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((original - np.mean(original)) ** 2)
    if ss_tot > 0:
        metrics['r2'] = 1 - (ss_res / ss_tot)
    else:
        metrics['r2'] = 1.0

    # Calculate weight elements count
    metrics['weight_elements'] = original.size

    return metrics
