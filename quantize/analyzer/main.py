# -*- coding: utf-8 -*-
"""
Main module for quantization analysis.
Contains the primary QuantizationAnalyzer class.
"""

import os
import numpy as np
from quantize.analyzer.utils import load_single_data, export_analysis_csv
from quantize.analyzer.formatters import format_table, print_section, format_quantization_table, format_combined_sections, format_rich_sections
from quantize.analyzer.metrics import analyze_original_tensor, analyze_distribution, calculate_error_metrics
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

    def load_data(self, original=None, quantized=None):
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

    def calculate_error_metrics(self):
        """
        Calculate various error metrics between original and quantized data.

        Returns:
            dict: Dictionary containing all calculated metrics
        """
        return calculate_error_metrics(self.original_data, self.quantized_data)

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

    def analyze_distribution(self, threshold=0.05, stats=None):
        """
        Analyze the distribution of the original data based on thresholds.

        Args:
            threshold: Threshold for determining distribution characteristics
            stats: Optional pre-calculated statistics

        Returns:
            dict: Distribution analysis results
        """
        if self.original_data is None:
            return {'distribution_type': 'Unknown (No Data)'}

        if stats is None:
            stats = analyze_original_tensor(self.original_data)

        return analyze_distribution(stats, threshold)

    def run_complete_analysis(self, original=None, quantized=None, threshold=0.05):
        """
        Run the complete analysis workflow.

        Args:
            original: Original data (array or file path)
            quantized: Quantized data (array or file path)
            threshold: Threshold for distribution analysis

        Returns:
            dict: Dictionary containing analysis results
        """
        print("Starting complete analysis...")

        # Load data if provided
        if original is not None or quantized is not None:
            self.load_data(original, quantized)

        original_stats = analyze_original_tensor(self.original_data)

        # Analyze distribution
        dist_analysis = self.analyze_distribution(threshold, stats=original_stats)
        original_stats['distribution_analysis'] = dist_analysis

        # Calculate metrics
        error_metrics = self.calculate_error_metrics()
        snr_value = error_metrics.get('snr', float('inf'))
        psnr_value = error_metrics.get('psnr', float('inf'))

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
