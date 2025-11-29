# -*- coding: utf-8 -*-
"""
Metrics module for quantization analysis.
"""

import numpy as np
from scipy import stats
from quantize.analyzer.formatters import format_table, print_section

def calculate_snr(original_data, quantized_data):
    """
    Calculate SNR (Signal-to-Noise Ratio).

    Args:
        original_data: Original data array
        quantized_data: Quantized data array

    Returns:
        snr_value: SNR in dB
    """
    if original_data is None or quantized_data is None:
        raise ValueError("Please provide original and quantized data")

    # Calculate signal power (variance of original data)
    signal_power = np.var(original_data)

    # Calculate noise power (variance of difference)
    noise = original_data - quantized_data
    noise_power = np.var(noise)

    # Avoid division by zero
    if noise_power == 0:
        print("Warning: Noise power is 0, SNR is infinite")
        return float('inf')

    # Calculate SNR (dB)
    snr_value = 10 * np.log10(signal_power / noise_power)

    return snr_value

def calculate_psnr(original_data, quantized_data):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio).

    Args:
        original_data: Original data array
        quantized_data: Quantized data array

    Returns:
        psnr_value: PSNR in dB
    """
    if original_data is None or quantized_data is None:
        raise ValueError("Please provide original and quantized data")

    # Calculate Mean Squared Error (MSE)
    mse = np.mean((original_data - quantized_data) ** 2)

    # Avoid division by zero
    if mse == 0:
        print("Warning: MSE is 0, PSNR is infinite")
        return float('inf')

    # Determine max signal value automatically
    max_value = np.max(np.abs(original_data))

    # Avoid invalid max value
    if max_value == 0:
        print("Warning: Max signal is 0, cannot calculate PSNR")
        return float('nan')

    # Calculate PSNR (dB)
    psnr_value = 10 * np.log10((max_value ** 2) / mse)

    return psnr_value

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

    skewness = stats.skew(data_flat)  # 偏度
    kurtosis = stats.kurtosis(data_flat)  # 峰度
    percentile_25 = np.percentile(data_flat, 25)
    percentile_75 = np.percentile(data_flat, 75)
    iqr = percentile_75 - percentile_25  # 四分位距

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
