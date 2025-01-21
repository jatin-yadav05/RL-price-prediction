import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_preprocessing import DataPreprocessor
from environment import PricingEnvironment
from model import PricingModel

def evaluate_model(data_path: str, model_path: str, num_episodes: int = 10):
    """Evaluate the trained model on historical data."""
    # Load and preprocess data
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(data_path)
    data, stats = preprocessor.preprocess_data(df)
    
    # Create environment and load model
    env = PricingEnvironment(data, stats)
    model = PricingModel(env)
    model.load(model_path)
    
    # Evaluation metrics
    episode_rewards = []
    price_trajectories = []
    sales_trajectories = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        prices = []
        sales = []
        
        while True:
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
        
        episode_rewards.append(episode_reward)
        price_trajectories.append(prices)
        sales_trajectories.append(sales)
    
    # Calculate and print metrics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_price = np.mean([np.mean(p) for p in price_trajectories])
    mean_sales = np.mean([np.mean(s) for s in sales_trajectories])
    
    print(f"\nEvaluation Results (over {num_episodes} episodes):")
    print(f"Mean Episode Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean Suggested Price: ${mean_price:.2f}")
    print(f"Mean Expected Sales: {mean_sales:.2f} units")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    # Plot price trajectories
    plt.subplot(1, 2, 1)
    for prices in price_trajectories:
        plt.plot(prices, alpha=0.3)
    plt.plot(np.mean(price_trajectories, axis=0), 'r-', label='Mean Price')
    plt.axhline(y=stats['median_price'], color='g', linestyle='--', label='Historical Median')
    plt.title('Price Trajectories')
    plt.xlabel('Step')
    plt.ylabel('Price ($)')
    plt.legend()
    
    # Plot sales trajectories
    plt.subplot(1, 2, 2)
    for sales in sales_trajectories:
        plt.plot(sales, alpha=0.3)
    plt.plot(np.mean(sales_trajectories, axis=0), 'r-', label='Mean Sales')
    plt.axhline(y=stats['median_sales'], color='g', linestyle='--', label='Historical Median')
    plt.title('Sales Trajectories')
    plt.xlabel('Step')
    plt.ylabel('Sales (units)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate the trained pricing RL model')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the historical data CSV file')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model file')
    parser.add_argument('--num_episodes', type=int, default=10,
                      help='Number of evaluation episodes')
    
    args = parser.parse_args()
    evaluate_model(args.data_path, args.model_path, args.num_episodes)

if __name__ == '__main__':
    main() 