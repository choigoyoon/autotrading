# Comprehensive Quantitative Trading Suite

This repository contains a comprehensive suite of tools for developing, backtesting, and deploying quantitative trading strategies. It is designed with a focus on modularity, enabling robust research and execution pipelines.

## Project Goal

The primary goal of this project is to implement "Enhanced System 2.0", a sophisticated trading system featuring a 6-step verification process and a flow-first logic for signal validation and grading.

## Key Features

- **Data-Driven Analysis**: Tools for in-depth analysis of market data and strategy performance.
- **Advanced Labeling**: Sophisticated labeling techniques for training supervised learning models.
- **ML/DL Model Integration**: Supports various machine learning and deep learning models for signal prediction.
- **Robust Backtesting**: A comprehensive backtesting engine to validate strategies against historical data.
- **Modular Architecture**: A clean, modular structure that separates concerns and enhances reusability.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Pip and a virtual environment tool (`venv`)

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd quant-trading-suite
    ```

2.  Create and activate a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  Install the project in editable mode along with its dependencies:
    ```bash
    pip install -e .
    ```

4.  To install development tools (like `pytest` and `pyright`), run:
    ```bash
    pip install -e .[dev]
    ```

## Usage

The `scripts/` directory contains various scripts to run different parts of the pipeline, such as data processing, model training, and backtesting.

Example:
```bash
python scripts/generate_sequences.py
```

## Project Structure

- `src/`: Contains the core source code, organized by functionality.
- `scripts/`: Executable scripts for running various tasks.
- `tools/`: Utility and helper scripts for analysis and maintenance.
- `configs/`: Configuration files for different environments and models.
- `data/`: Raw and processed data (ignored by Git).
- `results/`: Backtesting results, logs, and artifacts (ignored by Git). 