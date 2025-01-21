import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_preprocessing import DataPreprocessor
from src.environment import PricingEnvironment
from src.model import PricingModel

# Page config
st.set_page_config(
    page_title="Model Evaluation - Price Optimization RL",
    page_icon="📊",
    layout="wide"
)

def evaluate_model(data_path: str, model_path: str, num_episodes: int = 10):
    """Evaluate the trained model and return metrics."""
    # Load and preprocess data
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(data_path)
    data, stats = preprocessor.preprocess_data(df)
    
    # Create environment and load model
    env = PricingEnvironment(data, stats)
    model = PricingModel(env)
    model.load(model_path)
    
    # Pre-allocate arrays for better performance
    episode_rewards = np.zeros(num_episodes)
    price_trajectories = []
    sales_trajectories = []
    
    # Progress placeholder
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        prices = []
        sales = []
        
        # Reduced max steps per episode
        max_steps = min(50, len(data['states']) - 1)  # Limit to 50 steps max
        
        for step in range(max_steps):
            # Get model prediction
            action = model.predict(state)
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Record metrics
            episode_reward += reward
            prices.append(preprocessor.inverse_transform_price(info['price'])[0][0])
            sales.append(preprocessor.inverse_transform_sales(info['sales'])[0][0])
            
            if terminated or truncated:
                break
                
            state = next_state
        
        # Update progress
        progress = (episode + 1) / num_episodes
        progress_bar.progress(progress)
        status_text.text(f"Evaluating episode {episode + 1}/{num_episodes}")
        
        episode_rewards[episode] = episode_reward
        price_trajectories.append(prices)
        sales_trajectories.append(sales)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Calculate metrics efficiently using numpy
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_price = np.mean([np.mean(p) for p in price_trajectories])
    mean_sales = np.mean([np.mean(s) for s in sales_trajectories])
    
    # Calculate changes from historical
    historical_mean_price = df['Product Price'].mean()
    historical_mean_sales = df['Total Sales'].mean()
    price_change = ((mean_price - historical_mean_price) / historical_mean_price) * 100
    sales_change = ((mean_sales - historical_mean_sales) / historical_mean_sales) * 100
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_price': mean_price,
        'mean_sales': mean_sales,
        'price_change': price_change,
        'sales_change': sales_change,
        'price_trajectories': price_trajectories,
        'sales_trajectories': sales_trajectories,
        'historical_mean_price': historical_mean_price,
        'historical_mean_sales': historical_mean_sales
    }

def plot_trajectories(trajectories, mean_value, title, y_label):
    """Create a plot for trajectories with mean and historical reference."""
    fig = go.Figure()
    
    # Plot only a subset of individual trajectories for better performance
    max_trajectories = min(5, len(trajectories))  # Show max 5 trajectories
    for i in range(max_trajectories):
        fig.add_trace(go.Scatter(
            y=trajectories[i],
            mode='lines',
            name=f'Episode {i+1}',
            opacity=0.3,
            showlegend=False,
            line=dict(color='lightblue')
        ))
    
    # Plot mean trajectory
    mean_traj = np.mean(trajectories, axis=0)
    fig.add_trace(go.Scatter(
        y=mean_traj,
        mode='lines',
        name='Mean',
        line=dict(color='blue', width=2)
    ))
    
    # Add historical reference
    fig.add_hline(
        y=mean_value,
        line_dash="dash",
        line_color="red",
        annotation_text="Historical Mean",
        annotation_position="bottom right"
    )
    
    fig.update_layout(
        title=title,
        xaxis_title='Step',
        yaxis_title=y_label,
        template='plotly_white',
        height=400
    )
    
    return fig

def main():
    st.title("📊 Model Evaluation")
    
    # Sidebar
    st.sidebar.header("Evaluation Settings")
    
    product = st.sidebar.selectbox(
        "Select Product",
        ["Soapnuts", "Wool Balls"],
        help="Choose the product to evaluate"
    )
    
    num_episodes = st.sidebar.number_input(
        "Number of Episodes",
        min_value=1,
        max_value=20,  # Reduced max episodes
        value=5,  # Reduced default value
        help="Number of episodes to evaluate"
    )
    
    # Main content
    st.markdown("""
    This page evaluates the trained model's performance by running multiple episodes
    and analyzing the results. The evaluation shows how well the model balances
    price optimization with sales maintenance.
    """)
    
    # Get file paths
    data_path = f"data/{product.lower().replace(' ', '')}history.csv"
    model_path = f"models/pricing_model_{product.lower().replace(' ', '')}.zip"
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.error("Model not found! Please train the model first.")
        return
    
    # Evaluation button
    if st.button("Start Evaluation", type="primary"):
        try:
            with st.spinner("Running evaluation..."):
                # Run evaluation with progress tracking
                results = evaluate_model(data_path, model_path, num_episodes)
                
                # Display metrics
                st.header("Evaluation Results")
                
                # Metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Mean Price",
                        f"${results['mean_price']:.2f}",
                        f"{results['price_change']:.1f}%"
                    )
                
                with col2:
                    st.metric(
                        "Mean Sales",
                        f"{results['mean_sales']:.1f}",
                        f"{results['sales_change']:.1f}%"
                    )
                
                with col3:
                    st.metric(
                        "Mean Reward",
                        f"{results['mean_reward']:.2f}",
                        f"±{results['std_reward']:.2f}"
                    )
                
                with col4:
                    success_rate = np.mean(
                        [1 if p > results['historical_mean_price'] and s >= 0.9 * results['historical_mean_sales']
                         else 0 for p, s in zip(
                             np.mean(results['price_trajectories'], axis=1),
                             np.mean(results['sales_trajectories'], axis=1)
                         )]
                    ) * 100
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                
                # Plots
                st.subheader("Performance Visualization")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    price_fig = plot_trajectories(
                        results['price_trajectories'],
                        results['historical_mean_price'],
                        'Price Trajectories',
                        'Price ($)'
                    )
                    st.plotly_chart(price_fig, use_container_width=True)
                
                with col2:
                    sales_fig = plot_trajectories(
                        results['sales_trajectories'],
                        results['historical_mean_sales'],
                        'Sales Trajectories',
                        'Sales (units)'
                    )
                    st.plotly_chart(sales_fig, use_container_width=True)
                
                # Analysis
                st.subheader("Analysis")
                
                if results['price_change'] > 0 and results['sales_change'] >= -10:
                    st.success(
                        "The model has successfully learned to optimize prices while maintaining sales. "
                        f"It suggests a {results['price_change']:.1f}% price increase with only a "
                        f"{abs(results['sales_change']):.1f}% impact on sales volume."
                    )
                elif results['price_change'] > 0 and results['sales_change'] < -10:
                    st.warning(
                        "The model suggests higher prices but with a significant impact on sales. "
                        "Consider retraining with adjusted rewards to better balance price increases with sales maintenance."
                    )
                else:
                    st.info(
                        "The model suggests maintaining or lowering prices to optimize overall revenue. "
                        "This might be appropriate if it leads to significantly higher sales volumes."
                    )
                
        except Exception as e:
            st.error(f"Error during evaluation: {str(e)}")

if __name__ == "__main__":
    main() 