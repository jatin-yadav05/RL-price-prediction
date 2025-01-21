import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.train import train_model
from src.utils import setup_logging

# Page config
st.set_page_config(
    page_title="Model Training - Price Optimization RL",
    page_icon="🎯",
    layout="wide"
)

def create_training_log_plot(log_file):
    """Create a plot from training logs."""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        rewards = []
        timestamps = []
        
        for line in lines:
            if "mean_reward" in line:
                try:
                    timestamp = datetime.strptime(line.split(' - ')[0], '%Y-%m-%d %H:%M:%S,%f')
                    reward = float(line.split('mean_reward=')[1].split()[0])
                    rewards.append(reward)
                    timestamps.append(timestamp)
                except:
                    continue
        
        if rewards:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=rewards,
                mode='lines',
                name='Mean Reward'
            ))
            fig.update_layout(
                title='Training Progress',
                xaxis_title='Time',
                yaxis_title='Mean Reward',
                template='plotly_white'
            )
            return fig
    except Exception as e:
        st.error(f"Error creating training plot: {str(e)}")
        return None

def main():
    st.title("🎯 Model Training")
    
    # Sidebar
    st.sidebar.header("Training Settings")
    
    product = st.sidebar.selectbox(
        "Select Product",
        ["Soapnuts", "Wool Balls"],
        help="Choose the product to train the model for"
    )
    
    timesteps = st.sidebar.number_input(
        "Training Timesteps",
        min_value=10000,
        max_value=1000000,
        value=100000,
        step=10000,
        help="Number of timesteps to train the model"
    )
    
    # Main content
    st.markdown("""
    This page allows you to train the reinforcement learning model for price optimization.
    The model will learn from historical data to suggest optimal prices.
    """)
    
    # Training section
    st.header("Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Source")
        data_path = f"data/{product.lower().replace(' ', '')}history.csv"
        st.text(f"Training data: {data_path}")
        
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            st.success(f"Found {len(df)} data points")
        else:
            st.error("Training data not found!")
            return
    
    with col2:
        st.subheader("Model Output")
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = f"{model_dir}/pricing_model_{product.lower().replace(' ', '')}.zip"
        st.text(f"Model will be saved to: {model_path}")
    
    # Training button
    if st.button("Start Training", type="primary"):
        try:
            # Setup logging
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"training_{product.lower().replace(' ', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            setup_logging(log_file)
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Training
            status_text.text("Training in progress...")
            train_model(data_path, model_path, log_dir)
            
            progress_bar.progress(100)
            status_text.text("Training completed!")
            
            # Show training plot
            st.subheader("Training Progress")
            fig = create_training_log_plot(log_file)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"Model has been trained and saved to {model_path}")
            
        except Exception as e:
            st.error(f"Error during training: {str(e)}")

if __name__ == "__main__":
    main() 