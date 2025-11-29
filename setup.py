# -*- coding: utf-8 -*-
"""
Setup script for quantize package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="quantize-analysis",
    version="1.0.0",
    description="Quantized Data Analysis Tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Quantize Team",
    author_email="",
    url="",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0"
    ],
    entry_points={
        'console_scripts': [
            'quant-analyze=quantize.analyzer.cli:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)