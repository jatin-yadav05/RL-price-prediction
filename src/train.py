import argparse
import os
import logging
import numpy as np
import pandas as pd
from src.data_preprocessing import DataPreprocessor
from src.environment import PricingEnvironment
from src.model import PricingModel, TensorboardCallback
from src.utils import validate_data

def augment_data(df: pd.DataFrame, min_required: int = 100) -> pd.DataFrame:
    """Augment the dataset if it's too small by interpolating between points."""
    if len(df) >= min_required:
        return df
    
    # Sort by date
    df = df.sort_values('Report Date')
    
    # Create new dates between existing dates
    dates = pd.date_range(start=df['Report Date'].min(), end=df['Report Date'].max(), periods=min_required)
    
    # Create interpolated dataframe
    new_df = pd.DataFrame({'Report Date': dates})
    
    # Interpolate numeric columns
    numeric_cols = ['Product Price', 'Organic Conversion Percentage', 'Ad Conversion Percentage', 
                   'Total Profit', 'Total Sales', 'Predicted Sales']
    
    for col in numeric_cols:
        if col in df.columns:
            # Create interpolation function
            values = df[col].values
            old_indices = np.linspace(0, 1, len(df))
            new_indices = np.linspace(0, 1, min_required)
            
            # Interpolate values
            interpolated = np.interp(new_indices, old_indices, values)
            
            # Add small random noise to avoid identical values
            noise = np.random.normal(0, 0.01 * np.std(values), size=len(interpolated))
            interpolated = np.maximum(0, interpolated + noise)  # Ensure no negative values
            
            new_df[col] = interpolated
    
    return new_df

def train_model(data_path: str, model_save_path: str, log_dir: str = 'logs'):
    """Train the pricing model on historical data."""
    logger = logging.getLogger(__name__)
    
    try:
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        preprocessor = DataPreprocessor()
        df = preprocessor.load_data(data_path)
        
        # Validate data
        is_valid, issues = validate_data(df)
        if not is_valid:
            for issue in issues:
                logger.error(f"Data validation error: {issue}")
            raise ValueError("Data validation failed")
        
        # Augment data if needed
        original_length = len(df)
        df = augment_data(df, min_required=100)
        if len(df) > original_length:
            logger.info(f"Augmented data from {original_length} to {len(df)} points")
        
        # Preprocess data
        data, stats = preprocessor.preprocess_data(df)
        
        logger.info(f"Using {len(data['states'])} data points")
        logger.info(f"Price range: ${stats['median_price']:.2f} (median), ${stats['max_price']:.2f} (max)")
        
        # Create environment
        env = PricingEnvironment(data, stats)
        
        try:
            # Try to import tensorboard
            import tensorboard
            use_tensorboard = True
            callback = TensorboardCallback()
        except ImportError:
            logger.warning("TensorBoard not available. Training will proceed without logging.")
            use_tensorboard = False
            callback = None
        
        # Initialize model with conditional tensorboard logging
        model = PricingModel(
            env,
            log_dir=log_dir if use_tensorboard else None
        )
        
        # Train the model with fewer timesteps initially
        logger.info(f"Training model on {data_path}...")
        initial_timesteps = 50000  # Reduced number of steps for smaller dataset
        
        try:
            model.train(
                total_timesteps=initial_timesteps,
                callback=callback if use_tensorboard else None
            )
            logger.info("Initial training completed successfully")
            
            # Save the trained model
            model.save(model_save_path)
            logger.info(f"Model saved to {model_save_path}")
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Train the pricing RL model')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the historical data CSV file')
    parser.add_argument('--model_save_path', type=str, required=True,
                      help='Path to save the trained model')
    parser.add_argument('--log_dir', type=str, default='logs',
                      help='Directory to save training logs')
    
    args = parser.parse_args()
    train_model(args.data_path, args.model_save_path, args.log_dir)

if __name__ == '__main__':
    main() 