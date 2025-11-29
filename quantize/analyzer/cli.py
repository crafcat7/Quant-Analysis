# -*- coding: utf-8 -*-
"""
Command Line Interface for quantization analysis tool.
"""

import argparse
import numpy as np
from .main import QuantizationAnalyzer

def main():
    """
    Main function providing CLI.

    Examples:
    # Analyze specific files
    python -m quantize.analyzer.cli --original original.npy --quantized quantized.npy
    """

    parser = argparse.ArgumentParser(description='Quantized Data Analysis Tool')

    parser.add_argument('--original', '-o', type=str, required=True, help='Path to original data file')
    parser.add_argument('--quantized', '-q', type=str, required=True, help='Path to quantized data file')
    parser.add_argument('--shape', '-s', type=str, required=False, help='Tensor shape, e.g., 4x64x1x1')

    args = parser.parse_args()

    try:
        analyzer = QuantizationAnalyzer()

        # Always run complete analysis using the analyzer instance
        analyzer.run_complete_analysis(args.original, args.quantized, tensor_shape=args.shape)

        return 0
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
