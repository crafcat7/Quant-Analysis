# -*- coding: utf-8 -*-
"""
Main module for quantization analysis.
Contains the primary QuantizationAnalyzer class.
"""

import os
import numpy as np
import torch
from quantize.analyzer.utils import load_single_data, export_analysis_csv, parse_tensor_shape
from quantize.analyzer.formatters import format_table, print_section, format_quantization_table, format_combined_sections, format_rich_sections
from quantize.analyzer.metrics import calculate_snr as calc_snr, calculate_psnr as calc_psnr, analyze_original_tensor
from quantize.analyzer.charts import (
    plot_accuracy_fit as plot_accuracy,
    plot_histogram_comparison as plot_histogram,
    plot_scatter_comparison as plot_scatter,
    plot_error_distribution as plot_error
)

class QuantizationAnalyzer:
    """
    Class for analyzing quantized data.
    This is the main interface to the quantization analysis tools.
    """

    def __init__(self, output_dir='output'):
        """
        Initialize the analyzer.

        Args:
            output_dir: Directory to save output plots
        """
        self.original_data = None
        self.quantized_data = None
        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_data(self, original=None, quantized=None, tensor_shape=None):
        """
        Load data.

        Args:
            original: Original data (array or file path)
            quantized: Dequantized data (array or file path)

        Returns:
            self: For method chaining
        """
        # Load original and dequantized data
        self.original_data = load_single_data(original)
        self.quantized_data = load_single_data(quantized)

        if tensor_shape is not None:
            ts = parse_tensor_shape(tensor_shape)
            prod = int(np.prod(ts))
            if self.original_data is not None and self.original_data.size == prod:
                self.original_data = self.original_data.reshape(ts)
            if self.quantized_data is not None and self.quantized_data.size == prod:
                self.quantized_data = self.quantized_data.reshape(ts)

        # Ensure data aligned for comparison
        if self.original_data is not None and self.quantized_data is not None:
            if self.original_data.shape != self.quantized_data.shape:
                of = np.array(self.original_data).flatten()
                qf = np.array(self.quantized_data).flatten()
                min_len = min(of.size, qf.size)
                if min_len == 0:
                    raise ValueError("No overlapping elements to compare after alignment")
                self.original_data = of[:min_len]
                self.quantized_data = qf[:min_len]

        # Suppress Data Loading table output

        return self

    def calculate_snr(self):
        """
        Calculate SNR (Signal-to-Noise Ratio).

        Returns:
            snr_value: SNR in dB
        """
        return calc_snr(self.original_data, self.quantized_data)

    def calculate_psnr(self):
        """
        Calculate PSNR (Peak Signal-to-Noise Ratio).

        Returns:
            psnr_value: PSNR in dB
        """
        return calc_psnr(self.original_data, self.quantized_data)

    def calculate_error_metrics(self):
        """
        Calculate various error metrics between original and quantized data.

        Returns:
            dict: Dictionary containing all calculated metrics
        """
        # Convert to numpy arrays if they're PyTorch tensors
        if isinstance(self.original_data, torch.Tensor):
            original = self.original_data.cpu().numpy()
        else:
            original = np.array(self.original_data)

        if isinstance(self.quantized_data, torch.Tensor):
            quantized = self.quantized_data.cpu().numpy()
        else:
            quantized = np.array(self.quantized_data)

        abs_errors = np.abs(original - quantized)

        # Calculate metrics
        metrics = {
            'mae': np.mean(abs_errors),
            'mse': np.mean((original - quantized) ** 2),
            'rmse': np.sqrt(np.mean((original - quantized) ** 2)),
            'max_abs_error': np.max(abs_errors),
            'min_abs_error': np.min(abs_errors),
            'std_abs_error': np.std(abs_errors),
        }

        # Calculate SQNR (Signal-to-Quantization Noise Ratio)
        signal_power = np.mean(original ** 2)
        noise_power = metrics['mse']
        if noise_power > 0:
            metrics['sqnr'] = 10 * np.log10(signal_power / noise_power)
        else:
            metrics['sqnr'] = float('inf')

        # Calculate Cosine Similarity
        original_flat = original.flatten()
        quantized_flat = quantized.flatten()

        # Avoid division by zero
        if np.linalg.norm(original_flat) > 0 and np.linalg.norm(quantized_flat) > 0:
            metrics['cosine_similarity'] = np.dot(original_flat, quantized_flat) / \
                                         (np.linalg.norm(original_flat) * np.linalg.norm(quantized_flat))
        else:
            metrics['cosine_similarity'] = 0.0

        # Calculate Mean Relative Error
        non_zero_original = original != 0
        if np.any(non_zero_original):
            relative_errors = np.abs(original[non_zero_original] - quantized[non_zero_original]) / np.abs(original[non_zero_original])
            metrics['mean_relative_error'] = np.mean(relative_errors)
        else:
            metrics['mean_relative_error'] = 0.0

        # Calculate R-squared (Coefficient of Determination)
        ss_res = np.sum((original - quantized) ** 2)
        ss_tot = np.sum((original - np.mean(original)) ** 2)
        if ss_tot > 0:
            metrics['r2'] = 1 - (ss_res / ss_tot)
        else:
            metrics['r2'] = 1.0

        # Calculate weight elements count
        metrics['weight_elements'] = original.size

        return metrics

    def plot_accuracy_fit(self, save=False, show=True):
        """
        Generate accuracy fit line plot.

        Args:
            save: Save image
            show: Show image

        Returns:
            dict: Dictionary containing fit statistics
        """
        return plot_accuracy(
            self.original_data,
            self.quantized_data,
            self.output_dir,
            save,
            show
        )

    def plot_histogram_comparison(self, bins=50, save=False, show=True):
        """
        Generate histogram comparison between original and quantized data.

        Args:
            bins: Number of histogram bins
            save: Save image
            show: Show image
        """
        plot_histogram(
            self.original_data,
            self.quantized_data,
            self.output_dir,
            bins,
            save,
            show
        )

    def plot_scatter_comparison(self, save=False, show=True, sample_size=None):
        """
        Generate scatter plot comparison.

        Args:
            save: Save image
            show: Show image
            sample_size: Number of samples to use (None for all)
        """
        plot_scatter(
            self.original_data,
            self.quantized_data,
            self.output_dir,
            save,
            show,
            sample_size
        )

    def plot_error_distribution(self, bins=50, save=False, show=True):
        """
        Generate error distribution plot.

        Args:
            bins: Number of histogram bins
            save: Save image
            show: Show image

        Returns:
            dict: Dictionary containing error statistics
        """
        return plot_error(
            self.original_data,
            self.quantized_data,
            self.output_dir,
            bins,
            save,
            show
        )

    def run_complete_analysis(self, original=None, quantized=None, tensor_shape=None):
        """
        Run the complete analysis workflow.

        Args:
            original: Original data (array or file path)
            quantized: Quantized data (array or file path)

        Returns:
            dict: Dictionary containing analysis results
        """
        print("Starting complete analysis...")

        # Load data if provided
        if original is not None or quantized is not None:
            self.load_data(original, quantized, tensor_shape)

        original_stats = analyze_original_tensor(self.original_data)

        # Calculate metrics
        snr_value = self.calculate_snr()
        psnr_value = self.calculate_psnr()
        error_metrics = self.calculate_error_metrics()
        error_metrics['snr'] = snr_value
        error_metrics['psnr'] = psnr_value

        rich_output = format_rich_sections(original_stats, error_metrics)
        if rich_output is not None:
            print(rich_output)
        else:
            print(format_combined_sections(original_stats, error_metrics))
        export_analysis_csv(original_stats, error_metrics, self.output_dir)



        # Generate all charts
        accuracy_stats = self.plot_accuracy_fit(save=True, show=False)
        self.plot_histogram_comparison(save=True, show=False)
        self.plot_scatter_comparison(save=True, show=False)
        error_stats = self.plot_error_distribution(save=True, show=False)



        # Return combined results
        results = {
            'snr': snr_value,
            'psnr': psnr_value,
            'error_metrics': error_metrics,
            'output_dir': self.output_dir,
            'accuracy_stats': accuracy_stats,
            'error_stats': error_stats,
            'original_stats': original_stats
        }

        return results
