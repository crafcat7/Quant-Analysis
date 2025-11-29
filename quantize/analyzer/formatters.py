# -*- coding: utf-8 -*-
"""
Formatting utilities for terminal output, including table formatting and section headers.
"""

import os

# ANSI color codes for terminal output
COLORS = {
    'red': '\033[91m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'blue': '\033[94m',
    'cyan': '\033[96m',
    'purple': '\033[95m',
    'reset': '\033[0m'
}

def format_table(headers, rows, title=None):
    """
    Format data as a table with proper alignment and borders.

    Args:
        headers: List of column headers
        rows: List of lists containing row data
        title: Optional table title

    Returns:
        Formatted table as a string
    """
    # Calculate terminal width for responsive table
    try:
        terminal_width = os.get_terminal_size().columns
    except:
        terminal_width = 80  # Default fallback

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(cell)))

    # Calculate total table width (including borders and padding)
    total_width = sum(col_widths) + len(headers) + 3

    # Create table
    result = []

    # Add title if provided
    if title:
        title_line = f"{title.center(total_width - 2)}"
        result.append(f"┌{title_line}┐")

    # Add header separator
    separator = "┬".join(["─" * (w + 2) for w in col_widths])
    result.append(f"┌{separator}┐")

    # Add headers
    header_line = "│".join([f" {h:<{w}} " for h, w in zip(headers, col_widths)])
    result.append(f"│{header_line}│")

    # Add separator between header and rows
    separator = "┼".join(["─" * (w + 2) for w in col_widths])
    result.append(f"├{separator}┤")

    # Add rows
    for row in rows:
        row_line = "│".join([f" {str(cell):<{w}} " for cell, w in zip(row, col_widths)])
        result.append(f"│{row_line}│")

    # Add bottom border
    separator = "┴".join(["─" * (w + 2) for w in col_widths])
    result.append(f"└{separator}┘")

    return "\n".join(result)


def print_section(title, color='blue'):
    """
    Print a section header with colored text.

    Args:
        title: Section title text
        color: Color name from COLORS dictionary
    """
    # Check if color is supported
    use_color = os.name != 'nt'  # Windows typically doesn't support ANSI colors in command prompt

    try:
        terminal_width = os.get_terminal_size().columns
    except:
        terminal_width = 80  # Default fallback

    # Create header with border
    padding = (terminal_width - len(title) - 4) // 2
    border = "=" * (padding + 2)

    if use_color and color in COLORS:
        print(f"\n{COLORS[color]}{border} {title} {border}{COLORS['reset']}")
    else:
        print(f"\n{border} {title} {border}")


def format_quantization_table(metrics):
    """
    Format quantization metrics into a comprehensive table.

    Args:
        metrics: Dictionary containing quantization metrics

    Returns:
        str: Formatted table as a string
    """
    # Create the metrics table
    all_metrics = []

    # Format SNR and PSNR (if available)
    snr = metrics.get('snr')
    if snr is not None:
        snr_str = "∞" if snr == float('inf') else f"{snr:.2f} dB"
        all_metrics.append(["SNR(dB)", snr_str])

    psnr = metrics.get('psnr')
    if psnr is not None:
        psnr_str = "∞" if psnr == float('inf') else f"{psnr:.2f} dB"
        all_metrics.append(["PSNR(dB)", psnr_str])

    # Format SQNR
    sqnr = metrics.get('sqnr', 0)
    if sqnr == float('inf'):
        sqnr_str = "∞"
    else:
        sqnr_str = f"{sqnr:.2f} dB"
    all_metrics.append(["SQNR(dB)", f"{sqnr_str}"])

    # Format Cosine Similarity
    cosine = metrics.get('cosine_similarity', 0)
    all_metrics.append(["CosineSimilarity", f"{cosine:.6f}"])

    # Format Mean Relative Error
    relative = metrics.get('mean_relative_error', 0)
    all_metrics.append(["MeanRelativeError", f"{relative:.6f}"])

    # Format R-squared
    r2 = metrics.get('r2', 0)
    all_metrics.append(["R2", f"{r2:.6f}"])

    # Format MAE
    mae = metrics.get('mae', 0)
    all_metrics.append(["MAE", f"{mae:.10f}"])

    # Format MSE
    mse = metrics.get('mse', 0)
    all_metrics.append(["MSE", f"{mse:.10f}"])

    # Format RMSE
    rmse = metrics.get('rmse', 0)
    all_metrics.append(["RMSE", f"{rmse:.10f}"])

    # Format Max Absolute Error
    max_abs = metrics.get('max_abs_error', 0)
    all_metrics.append(["MaxAbsError", f"{max_abs:.10f}"])

    # Format Min Absolute Error
    min_abs = metrics.get('min_abs_error', 0)
    all_metrics.append(["MinAbsError", f"{min_abs:.10f}"])

    # Format Std of Absolute Error
    std_abs = metrics.get('std_abs_error', 0)
    all_metrics.append(["StdAbsError", f"{std_abs:.10f}"])

    # Format Weight Elements
    weight_elements = metrics.get('weight_elements', 0)
    # 添加逗号分隔符来提高可读性，特别是对于大数字
    weight_elements_str = f"{weight_elements:,}"
    all_metrics.append(["WeightElements", f"{weight_elements_str:>18}"])

    headers = ["Metric", "Value"]
    col0_width = max(len(headers[0]), max(len(r[0]) for r in all_metrics))
    col1_width = max(len(headers[1]), max(len(r[1]) for r in all_metrics))

    top = f"┌{'─' * (col0_width + 2)}┬{'─' * (col1_width + 2)}┐"
    header = f"│ {headers[0]:<{col0_width}} │ {headers[1]:<{col1_width}} │"
    sep = f"├{'─' * (col0_width + 2)}┼{'─' * (col1_width + 2)}┤"
    bottom = f"└{'─' * (col0_width + 2)}┴{'─' * (col1_width + 2)}┘"

    result = [top, header, sep]
    for m, v in all_metrics:
        result.append(f"│ {m:<{col0_width}} │ {v:>{col1_width}} │")
    result.append(bottom)
    return "\n".join(result)

def format_combined_analysis(original_stats, metrics):
    headers = ["Category", "Metric", "Value"]
    rows = []
    rows.append(["Original", "Shape", str(original_stats.get('shape'))])
    rows.append(["Original", "Mean", f"{original_stats.get('mean', 0):.6f}"])
    rows.append(["Original", "Median", f"{original_stats.get('median', 0):.6f}"])
    rows.append(["Original", "Standard Deviation", f"{original_stats.get('std', 0):.6f}"])
    rows.append(["Original", "Min", f"{original_stats.get('min', 0):.6f}"])
    rows.append(["Original", "Max", f"{original_stats.get('max', 0):.6f}"])
    rows.append(["Original", "Skewness", f"{original_stats.get('skewness', 0):.6f}"])
    rows.append(["Original", "Kurtosis", f"{original_stats.get('kurtosis', 0):.6f}"])
    rows.append(["Original", "25th Percentile", f"{original_stats.get('percentile_25', 0):.6f}"])
    rows.append(["Original", "75th Percentile", f"{original_stats.get('percentile_75', 0):.6f}"])
    rows.append(["Original", "Interquartile Range (IQR)", f"{original_stats.get('iqr', 0):.6f}"])
    rows.append(["Original", "Zero Crossing Rate", f"{original_stats.get('zero_crossing_rate', 0):.6f}"])
    nz = original_stats.get('non_zero_count', 0)
    nzr = original_stats.get('non_zero_ratio', 0)
    rows.append(["Original", "Non-zero Elements", f"{nz} ({nzr:.2%})"])

    snr = metrics.get('snr')
    if snr is not None:
        rows.append(["Summary", "SNR(dB)", "∞" if snr == float('inf') else f"{snr:.2f} dB"])
    psnr = metrics.get('psnr')
    if psnr is not None:
        rows.append(["Summary", "PSNR(dB)", "∞" if psnr == float('inf') else f"{psnr:.2f} dB"])
    sqnr = metrics.get('sqnr')
    rows.append(["Summary", "SQNR(dB)", "∞" if sqnr == float('inf') else f"{sqnr:.2f} dB"])
    rows.append(["Summary", "CosineSimilarity", f"{metrics.get('cosine_similarity', 0):.6f}"])
    rows.append(["Summary", "MeanRelativeError", f"{metrics.get('mean_relative_error', 0):.6f}"])
    rows.append(["Summary", "R2", f"{metrics.get('r2', 0):.6f}"])
    rows.append(["Summary", "MAE", f"{metrics.get('mae', 0):.10f}"])
    rows.append(["Summary", "MSE", f"{metrics.get('mse', 0):.10f}"])
    rows.append(["Summary", "RMSE", f"{metrics.get('rmse', 0):.10f}"])
    rows.append(["Summary", "MaxAbsError", f"{metrics.get('max_abs_error', 0):.10f}"])
    rows.append(["Summary", "MinAbsError", f"{metrics.get('min_abs_error', 0):.10f}"])
    rows.append(["Summary", "StdAbsError", f"{metrics.get('std_abs_error', 0):.10f}"])
    rows.append(["Summary", "WeightElements", f"{metrics.get('weight_elements', 0):,}"])

    return format_table(headers, rows, "Combined Analysis")

def format_combined_sections(original_stats, metrics):
    original_rows = [
        ("Shape", str(original_stats.get('shape'))),
        ("Mean", f"{original_stats.get('mean', 0):.6f}"),
        ("Median", f"{original_stats.get('median', 0):.6f}"),
        ("Standard Deviation", f"{original_stats.get('std', 0):.6f}"),
        ("Min", f"{original_stats.get('min', 0):.6f}"),
        ("Max", f"{original_stats.get('max', 0):.6f}"),
        ("Skewness", f"{original_stats.get('skewness', 0):.6f}"),
        ("Kurtosis", f"{original_stats.get('kurtosis', 0):.6f}"),
        ("25th Percentile", f"{original_stats.get('percentile_25', 0):.6f}"),
        ("75th Percentile", f"{original_stats.get('percentile_75', 0):.6f}"),
        ("Interquartile Range (IQR)", f"{original_stats.get('iqr', 0):.6f}"),
        ("Zero Crossing Rate", f"{original_stats.get('zero_crossing_rate', 0):.6f}"),
        ("Non-zero Elements", f"{original_stats.get('non_zero_count', 0)} ({original_stats.get('non_zero_ratio', 0):.2%})")
    ]

    summary_rows = []
    snr = metrics.get('snr')
    if snr is not None:
        summary_rows.append(("SNR(dB)", "∞" if snr == float('inf') else f"{snr:.2f} dB"))
    psnr = metrics.get('psnr')
    if psnr is not None:
        summary_rows.append(("PSNR(dB)", "∞" if psnr == float('inf') else f"{psnr:.2f} dB"))
    sqnr = metrics.get('sqnr')
    summary_rows.append(("SQNR(dB)", "∞" if sqnr == float('inf') else f"{sqnr:.2f} dB"))
    summary_rows.append(("CosineSimilarity", f"{metrics.get('cosine_similarity', 0):.6f}"))
    summary_rows.append(("MeanRelativeError", f"{metrics.get('mean_relative_error', 0):.6f}"))
    summary_rows.append(("R2", f"{metrics.get('r2', 0):.6f}"))
    summary_rows.append(("MAE", f"{metrics.get('mae', 0):.10f}"))
    summary_rows.append(("MSE", f"{metrics.get('mse', 0):.10f}"))
    summary_rows.append(("RMSE", f"{metrics.get('rmse', 0):.10f}"))
    summary_rows.append(("MaxAbsError", f"{metrics.get('max_abs_error', 0):.10f}"))
    summary_rows.append(("MinAbsError", f"{metrics.get('min_abs_error', 0):.10f}"))
    summary_rows.append(("StdAbsError", f"{metrics.get('std_abs_error', 0):.10f}"))
    summary_rows.append(("WeightElements", f"{metrics.get('weight_elements', 0):,}"))

    headers = ["Metric", "Value"]
    col0_width = max(len(headers[0]), max(len(n) for n, _ in original_rows + summary_rows))
    col1_width = max(len(headers[1]), max(len(v) for _, v in original_rows + summary_rows))

    col0_seg = col0_width + 2
    col1_seg = col1_width + 2
    inner_width = col0_seg + 1 + col1_seg

    def section_box(title, rows):
        lines = []
        lines.append(f"┌{'─' * inner_width}┐")
        lines.append(f"│{title.center(inner_width)}│")
        lines.append(f"├{'─' * col0_seg}┬{'─' * col1_seg}┤")
        lines.append(f"│ {headers[0]:<{col0_width}} │ {headers[1]:<{col1_width}} │")
        lines.append(f"├{'─' * col0_seg}┼{'─' * col1_seg}┤")
        for n, v in rows:
            lines.append(f"│ {n:<{col0_width}} │ {v:>{col1_width}} │")
        lines.append(f"└{'─' * col0_seg}┴{'─' * col1_seg}┘")
        return lines

    result = []
    result.extend(section_box("Tensor Analysis", original_rows))
    result.extend(section_box("Dequantization", summary_rows))
    return "\n".join(result)

def format_rich_sections(original_stats, metrics):
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
    except Exception:
        return None
    console = Console(record=True)
    def build_rows(stats):
        return [
            ("Shape", str(stats.get('shape'))),
            ("Mean", f"{stats.get('mean', 0):.6f}"),
            ("Median", f"{stats.get('median', 0):.6f}"),
            ("Standard Deviation", f"{stats.get('std', 0):.6f}"),
            ("Min", f"{stats.get('min', 0):.6f}"),
            ("Max", f"{stats.get('max', 0):.6f}"),
            ("Skewness", f"{stats.get('skewness', 0):.6f}"),
            ("Kurtosis", f"{stats.get('kurtosis', 0):.6f}"),
            ("25th Percentile", f"{stats.get('percentile_25', 0):.6f}"),
            ("75th Percentile", f"{stats.get('percentile_75', 0):.6f}"),
            ("Interquartile Range (IQR)", f"{stats.get('iqr', 0):.6f}"),
            ("Zero Crossing Rate", f"{stats.get('zero_crossing_rate', 0):.6f}"),
            ("Non-zero Elements", f"{stats.get('non_zero_count', 0)} ({stats.get('non_zero_ratio', 0):.2%})")
        ]
    def build_summary_rows(m):
        rows = []
        s = m.get('snr')
        if s is not None:
            rows.append(("SNR(dB)", "∞" if s == float('inf') else f"{s:.2f} dB"))
        p = m.get('psnr')
        if p is not None:
            rows.append(("PSNR(dB)", "∞" if p == float('inf') else f"{p:.2f} dB"))
        q = m.get('sqnr')
        rows.append(("SQNR(dB)", "∞" if q == float('inf') else f"{q:.2f} dB"))
        rows.append(("CosineSimilarity", f"{m.get('cosine_similarity', 0):.6f}"))
        rows.append(("MeanRelativeError", f"{m.get('mean_relative_error', 0):.6f}"))
        rows.append(("R2", f"{m.get('r2', 0):.6f}"))
        rows.append(("MAE", f"{m.get('mae', 0):.10f}"))
        rows.append(("MSE", f"{m.get('mse', 0):.10f}"))
        rows.append(("RMSE", f"{m.get('rmse', 0):.10f}"))
        rows.append(("MaxAbsError", f"{m.get('max_abs_error', 0):.10f}"))
        rows.append(("MinAbsError", f"{m.get('min_abs_error', 0):.10f}"))
        rows.append(("StdAbsError", f"{m.get('std_abs_error', 0):.10f}"))
        rows.append(("WeightElements", f"{m.get('weight_elements', 0):,}"))
        return rows
    def build_table(title, rows):
        t = Table(title=title, box=box.SQUARE, show_lines=False, pad_edge=False)
        t.add_column("Metric", justify="left")
        t.add_column("Value", justify="right")
        for n, v in rows:
            t.add_row(n, v)
        return t
    console.print(build_table("Tensor Analysis", build_rows(original_stats)))
    console.print(build_table("Dequantization", build_summary_rows(metrics)))
    return console.export_text()
