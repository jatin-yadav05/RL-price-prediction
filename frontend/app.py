import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import DataPreprocessor
from src.environment import PricingEnvironment
from src.model import PricingModel

# Page config
st.set_page_config(
    page_title="Price Optimization RL",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    /* Capitalize sidebar navigation text */
    [data-testid="stSidebarNavLink"] span {
        text-transform: capitalize !important;
    }
    </style>
    """, unsafe_allow_html=True)

def load_data(file_path):
    """Load and preprocess data."""
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(file_path)
    data, stats = preprocessor.preprocess_data(df)
    return df, data, stats, preprocessor

def plot_price_history(df):
    """Plot historical price data."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Report Date'],
        y=df['Product Price'],
        mode='lines+markers',
        name='Historical Price'
    ))
    fig.update_layout(
        title='Historical Price Trends',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_white'
    )
    return fig

def plot_sales_vs_price(df):
    """Plot sales vs price scatter plot."""
    fig = px.scatter(
        df,
        x='Product Price',
        y='Total Sales',
        color='Organic Conversion Percentage',
        title='Sales vs Price',
        labels={
            'Product Price': 'Price ($)',
            'Total Sales': 'Total Sales (units)',
            'Organic Conversion Percentage': 'Conversion Rate (%)'
        }
    )
    return fig

def predict_price(model, state, preprocessor, historical_median_price):
    """Get price prediction from the model with exploration tendency."""
    # Multiple predictions with different noise levels
    num_predictions = 5
    prices = []
    
    for _ in range(num_predictions):
        action = model.predict(state)
        price = preprocessor.inverse_transform_price(action)[0][0]
        # Add increasing noise for higher prices to encourage exploration
        if price > historical_median_price:
            noise = np.random.uniform(0, 0.05)  # Positive noise for higher prices
        else:
            noise = np.random.uniform(-0.02, 0.02)  # Less noise for lower prices
        prices.append(price * (1 + noise))
    
    # Bias towards higher prices by taking the 75th percentile
    return np.percentile(prices, 75)

def main():
    st.title("💰 Price Optimization with Reinforcement Learning")
    
    # Sidebar
    st.sidebar.header("APP Settings")
    product = st.sidebar.selectbox(
        "Select Product",
        ["Soapnuts", "Wool Balls"],
        help="Choose the product to analyze"
    )
    
    # Load data based on selection
    data_path = f"data/{product.lower().replace(' ', '')}history.csv"
    model_path = f"models/pricing_model_{product.lower().replace(' ', '')}.zip"
    
    try:
        df, data, stats, preprocessor = load_data(data_path)
        
        # Main content
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Historical Data Analysis")
            st.plotly_chart(plot_price_history(df), use_container_width=True)
            
            # Key metrics
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            with metrics_col1:
                st.metric("Median Price", f"${stats['median_price']:.2f}")
            with metrics_col2:
                st.metric("Max Price", f"${stats['max_price']:.2f}")
            with metrics_col3:
                st.metric("Median Sales", f"{stats['median_sales']:.0f} units")
        
        with col2:
            st.subheader("📈 Sales vs Price Analysis")
            st.plotly_chart(plot_sales_vs_price(df), use_container_width=True)
        
        # Price Prediction Section
        st.header("🎯 Price Prediction")
        
        if os.path.exists(model_path):
            env = PricingEnvironment(data, stats)
            model = PricingModel(env)
            model.load(model_path)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Current State")
                current_price = st.number_input(
                    "Current Price ($)",
                    min_value=float(df['Product Price'].min()),
                    max_value=float(df['Product Price'].max()),
                    value=float(stats['median_price'])
                )
                
                current_sales = st.number_input(
                    "Current Sales (units)",
                    min_value=0,
                    max_value=int(df['Total Sales'].max()),
                    value=int(stats['median_sales'])
                )
                
                # Add conversion rate inputs
                organic_conversion = st.number_input(
                    "Organic Conversion Rate (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(df['Organic Conversion Percentage'].median()),
                    step=0.1
                )
                
                ad_conversion = st.number_input(
                    "Ad Conversion Rate (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(df['Ad Conversion Percentage'].median()),
                    step=0.1
                )
                
                # Add predicted sales
                predicted_sales = st.number_input(
                    "Predicted Sales (units)",
                    min_value=0,
                    max_value=int(df['Total Sales'].max() * 1.5),
                    value=int(df['Predicted Sales'].median()),
                    help="Predicted sales for the next period"
                )
            
            with col2:
                st.subheader("Model Prediction")
                if st.button("Get Price Recommendation"):
                    # Prepare state with all features
                    state = np.zeros(4, dtype=np.float32)
                    state[0] = preprocessor.price_scaler.transform([[current_price]])[0]
                    state[1] = preprocessor.sales_scaler.transform([[current_sales]])[0]
                    state[2:4] = preprocessor.conversion_scaler.transform([[organic_conversion, ad_conversion]])[0]
                    
                    # Get prediction
                    recommended_price = predict_price(model, state, preprocessor, stats['median_price'])
                    
                    # Display recommendation
                    price_change = ((recommended_price - current_price) / current_price) * 100
                    price_vs_median = ((recommended_price - stats['median_price']) / stats['median_price']) * 100
                    
                    # Show multiple metrics
                    st.metric(
                        "Recommended Price",
                        f"${recommended_price:.2f}",
                        f"{price_change:+.1f}% from current"
                    )
                    
                    st.metric(
                        "vs Historical Median",
                        f"${stats['median_price']:.2f}",
                        f"{price_vs_median:+.1f}%"
                    )
                    
                    # Enhanced recommendation explanation
                    if price_change > 0:
                        message = (
                            "The model suggests increasing the price to optimize revenue. "
                            f"This recommendation is based on:\n"
                            f"- Current sales of {current_sales} units (Predicted: {predicted_sales} units)\n"
                            f"- Organic conversion rate of {organic_conversion:.1f}%\n"
                            f"- Ad conversion rate of {ad_conversion:.1f}%\n\n"
                            f"This price is {abs(price_vs_median):.1f}% {'above' if price_vs_median > 0 else 'below'} "
                            f"the historical median price of ${stats['median_price']:.2f}"
                        )
                        st.success(message)
                    else:
                        message = (
                            "The model suggests decreasing the price to optimize revenue. "
                            f"This recommendation is based on:\n"
                            f"- Current sales of {current_sales} units (Predicted: {predicted_sales} units)\n"
                            f"- Organic conversion rate of {organic_conversion:.1f}%\n"
                            f"- Ad conversion rate of {ad_conversion:.1f}%\n\n"
                            f"This price is {abs(price_vs_median):.1f}% {'above' if price_vs_median > 0 else 'below'} "
                            f"the historical median price of ${stats['median_price']:.2f}"
                        )
                        st.info(message)
                    
                    # Add exploration vs exploitation indicator
                    exploration_score = min(100, max(0, price_vs_median + 50))
                    st.subheader("Exploration vs Exploitation")
                    st.progress(exploration_score / 100)
                    st.caption(
                        f"{'Exploring new price points' if exploration_score > 50 else 'Exploiting known good prices'} "
                        f"(Score: {exploration_score:.0f}/100)"
                    )
                    
                    # Show predicted impact
                    st.subheader("Predicted Impact")
                    impact_col1, impact_col2 = st.columns(2)
                    with impact_col1:
                        sales_impact = (predicted_sales - current_sales) / current_sales * 100
                        st.metric(
                            "Expected Sales Impact",
                            f"{predicted_sales:.0f} units",
                            f"{sales_impact:+.1f}%"
                        )
                    with impact_col2:
                        revenue_current = current_price * current_sales
                        revenue_predicted = recommended_price * predicted_sales
                        revenue_change = (revenue_predicted - revenue_current) / revenue_current * 100
                        st.metric(
                            "Expected Revenue Impact",
                            f"${revenue_predicted:.2f}",
                            f"{revenue_change:+.1f}%"
                        )
        else:
            st.warning("Model not found. Please train the model first.")
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

if __name__ == "__main__":
    main() 