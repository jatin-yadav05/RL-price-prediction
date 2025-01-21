"""Utility functions for the pricing RL project."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import logging
import os

def setup_logging(log_file: str) -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def plot_training_history(
    prices: List[float],
    sales: List[float],
    rewards: List[float],
    save_path: str
) -> None:
    """Plot training metrics history."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # Plot prices
    axes[0].plot(prices)
    axes[0].set_title('Price History')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Price ($)')
    
    # Plot sales
    axes[1].plot(sales)
    axes[1].set_title('Sales History')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Units Sold')
    
    # Plot rewards
    axes[2].plot(rewards)
    axes[2].set_title('Reward History')
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Reward')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def calculate_metrics(
    df: pd.DataFrame,
    predictions: np.ndarray
) -> Dict[str, float]:
    """Calculate evaluation metrics for the model predictions."""
    metrics = {
        'mean_price': float(np.mean(predictions)),
        'std_price': float(np.std(predictions)),
        'median_price': float(np.median(predictions)),
        'historical_mean_price': float(df['Product Price'].mean()),
        'historical_median_price': float(df['Product Price'].median()),
        'price_increase': float(
            (np.mean(predictions) - df['Product Price'].mean()) /
            df['Product Price'].mean() * 100
        )
    }
    return metrics

def validate_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate the input data format and contents."""
    required_columns = [
        'Report Date',
        'Product Price',
        'Organic Conversion Percentage',
        'Ad Conversion Percentage',
        'Total Profit',
        'Total Sales',
        'Predicted Sales'
    ]
    
    issues = []
    
    # Check for required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {', '.join(missing_cols)}")
    
    # Check for date format
    if 'Report Date' in df.columns:
        try:
            pd.to_datetime(df['Report Date'])
        except:
            issues.append("Invalid date format in Report Date column")
    
    # Check for numeric columns
    numeric_cols = [col for col in required_columns if col != 'Report Date']
    for col in numeric_cols:
        if col in df.columns and not pd.to_numeric(df[col], errors='coerce').notnull().all():
            issues.append(f"Non-numeric values found in {col}")
    
    return len(issues) == 0, issues

def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """Create a directory for the experiment with timestamp."""
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir 