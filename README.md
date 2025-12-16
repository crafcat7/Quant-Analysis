# Quantized Data Analysis Tool

A Python tool for analyzing quantized data, comparing with original data, and generating various visualizations and statistical metrics.

## Features

- **Signal Metrics Calculation**
  - Signal-to-Noise Ratio (SNR)
  - Peak Signal-to-Noise Ratio (PSNR)
  - Distribution Analysis (Type identification, Confidence score)

- **Visualizations**
  - Accuracy fit line plots with linear regression analysis
  - Histogram comparisons between original and quantized data
  - Scatter plot comparisons
  - Error distribution plots with statistical analysis

- **Data Support**
  - Multiple file formats: NPY, NPZ, CSV, TXT, PT, PTH
  - Direct numpy array and PyTorch tensor input support
  - Automatic data sampling for large datasets

## Installation

### From Source

```bash
# Clone the repository
git clone <repository-url>
cd quantize

# Install the package
pip install -e .
```

### Dependencies

- numpy>=1.20.0
- matplotlib>=3.5.0
- scipy>=1.7.0
- pandas>=1.3.0
- seaborn>=0.11.0
- scikit-learn>=1.0.0
- rich>=10.0.0

## Usage

### Command Line Interface

```bash
# Use the new standalone entry point (recommended, avoids import warnings)
python quantize_cli.py --original original.npy --quantized quantized.npy

# Analyze specific files (always runs complete analysis)
python -m quantize.analyzer.cli --original original.npy --quantized quantized.npy

# Or use the installed command
quant-analyze --original original.npy --quantized quantized.npy
```

### Python API

```python
from quantize import QuantizationAnalyzer

# Create analyzer instance
analyzer = QuantizationAnalyzer()

# Load data
analyzer.load_data(original='data.npy', quantized='quantized_data.npy')

# Calculate metrics
print(f"SNR: {analyzer.calculate_snr():.2f} dB")
print(f"PSNR: {analyzer.calculate_psnr():.2f} dB")

# Generate individual plots
analyzer.plot_accuracy_fit(save=True, show=False)
analyzer.plot_histogram_comparison(save=True, show=False)

# Run complete analysis
results = analyzer.run_complete_analysis()
print(f"Analysis complete. Results: {results}")

# Alternatively, use the utility function
try:
    from quantize import run_complete_analysis
    results = run_complete_analysis('data.npy', 'quantized_data.npy')
except ImportError:
    from quantize.analyzer.utils import run_complete_analysis
    results = run_complete_analysis('data.npy', 'quantized_data.npy')
```

## Project Structure

```
quantize/
├── __init__.py          # Package initialization
├── analyzer/            # Core analysis modules
│   ├── __init__.py      # Module initialization
│   ├── main.py          # Main QuantizationAnalyzer class
│   ├── metrics.py       # SNR and PSNR calculations
│   ├── charts.py        # Visualization functions
│   ├── utils.py         # Utility functions
│   └── cli.py           # Command line interface
├── quant_analysis.py    # Entry point
├── setup.py             # Package setup
└── README.md            # This file
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.