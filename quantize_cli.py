# -*- coding: utf-8 -*-
"""
Main entry point for quantization analysis tool.
This script serves as a wrapper to avoid module import order warnings.
"""

from quantize.analyzer.cli import main

if __name__ == "__main__":
    exit(main())