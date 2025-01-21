import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict

class DataPreprocessor:
    def __init__(self):
        self.price_scaler = MinMaxScaler()
        self.sales_scaler = MinMaxScaler()
        self.conversion_scaler = MinMaxScaler()
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and clean the historical data."""
        df = pd.read_csv(file_path)
        df['Report Date'] = pd.to_datetime(df['Report Date'])
        df = df.sort_values('Report Date')
        
        # Forward fill missing values in Product Price
        df['Product Price'] = df['Product Price'].ffill()
        
        # Fill missing values with 0 for other numeric columns
        numeric_cols = ['Organic Conversion Percentage', 'Ad Conversion Percentage', 
                       'Total Profit', 'Total Sales', 'Predicted Sales']
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Remove rows where essential columns are still missing
        essential_cols = ['Product Price', 'Total Sales']
        df = df.dropna(subset=essential_cols)
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """Preprocess the data for training."""
        # Ensure data is in correct format
        df = df.copy()
        df['Product Price'] = pd.to_numeric(df['Product Price'], errors='coerce')
        df['Total Sales'] = pd.to_numeric(df['Total Sales'], errors='coerce')
        df['Organic Conversion Percentage'] = pd.to_numeric(df['Organic Conversion Percentage'], errors='coerce')
        df['Ad Conversion Percentage'] = pd.to_numeric(df['Ad Conversion Percentage'], errors='coerce')
        
        # Remove any rows with NaN after conversion
        df = df.dropna(subset=['Product Price', 'Total Sales'])
        
        # Scale the features
        prices = self.price_scaler.fit_transform(df[['Product Price']].values)
        sales = self.sales_scaler.fit_transform(df[['Total Sales']].values)
        conversions = self.conversion_scaler.fit_transform(
            df[['Organic Conversion Percentage', 'Ad Conversion Percentage']].fillna(0).values
        )
        
        # Ensure all arrays are float32
        prices = prices.astype(np.float32)
        sales = sales.astype(np.float32)
        conversions = conversions.astype(np.float32)
        
        # Create state features
        states = np.concatenate([
            prices,
            sales,
            conversions
        ], axis=1)
        
        # Calculate statistics for reward scaling
        stats = {
            'median_price': float(df['Product Price'].median()),
            'max_price': float(df['Product Price'].max()),
            'median_sales': float(df['Total Sales'].median()),
            'max_sales': float(df['Total Sales'].max()),
            'price_scale': float(self.price_scaler.scale_[0]),
            'sales_scale': float(self.sales_scaler.scale_[0])
        }
        
        # Prepare training data
        data = {
            'states': states,
            'prices': prices,
            'sales': sales,
            'conversions': conversions,
            'raw_prices': df['Product Price'].values,
            'raw_sales': df['Total Sales'].values,
            'predicted_sales': df['Predicted Sales'].values
        }
        
        return data, stats
    
    def inverse_transform_price(self, scaled_price: np.ndarray) -> np.ndarray:
        """Convert scaled price back to original scale."""
        return self.price_scaler.inverse_transform(scaled_price.reshape(-1, 1))
    
    def inverse_transform_sales(self, scaled_sales: np.ndarray) -> np.ndarray:
        """Convert scaled sales back to original scale."""
        return self.sales_scaler.inverse_transform(scaled_sales.reshape(-1, 1)) 