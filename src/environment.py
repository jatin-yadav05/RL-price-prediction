import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Any
from gymnasium import spaces

class PricingEnvironment(gym.Env):
    def __init__(self, data: Dict[str, np.ndarray], stats: Dict[str, float]):
        super().__init__()
        
        self.data = data
        self.stats = stats
        
        # Define action space (normalized price adjustments)
        # Increased range to allow for more exploration of higher prices
        self.action_space = spaces.Box(
            low=-0.5,  # Allow larger price decreases
            high=1.0,  # Allow larger price increases
            shape=(1,),
            dtype=np.float32
        )
        
        # Define observation space (price, sales, organic conversion, ad conversion)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(4,),
            dtype=np.float32
        )
        
        self.current_step = 0
        self.max_steps = len(self.data['states']) - 1
        self.historical_median_price = stats['median_price']
        self.historical_median_sales = stats['median_sales']
        
    def reset(self, seed=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Start from a random point in the historical data
        self.current_step = np.random.randint(0, self.max_steps)
        self.current_state = self.data['states'][self.current_step].astype(np.float32)
        
        return self.current_state, {}
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        # Convert normalized action to price adjustment
        # Allow for larger price adjustments (up to 20%)
        price_adjustment = action[0] * 0.2
        current_price = self.data['prices'][self.current_step][0]
        new_price = np.clip(current_price + price_adjustment, 0.1, 2.0)  # Allow up to 2x scaling
        
        # Get next state from historical data
        self.current_step = (self.current_step + 1) % self.max_steps
        next_state = self.data['states'][self.current_step].copy().astype(np.float32)
        next_state[0] = new_price  # Update price in state
        
        # Calculate reward components
        historical_price = max(0.1, self.data['prices'][self.current_step][0])
        price_ratio = new_price / historical_price
        
        current_sales = self.data['sales'][self.current_step][0]
        predicted_sales = self.data['predicted_sales'][self.current_step]
        sales_ratio = current_sales / max(0.1, predicted_sales)
        
        conversion_rates = self.data['conversions'][self.current_step]
        organic_conversion = conversion_rates[0]
        ad_conversion = conversion_rates[1]
        conversion_bonus = np.clip(np.mean([organic_conversion, ad_conversion]), 0, 1)
        
        # Enhanced reward function that:
        # 1. Rewards higher prices (especially above median)
        # 2. Rewards maintaining/increasing sales
        # 3. Rewards good conversion rates
        # 4. Punishes sales below predictions
        
        # Price component
        price_reward = 0.0
        if new_price > self.historical_median_price:
            # Extra reward for prices above historical median
            price_reward = 0.4 * (new_price / self.historical_median_price - 1.0)
        else:
            price_reward = 0.2 * (price_ratio - 1.0)
            
        # Sales component with prediction penalty
        sales_reward = 0.0
        if current_sales >= predicted_sales * 0.9:  # Within 10% of prediction
            sales_reward = 0.4 * (sales_ratio - 0.9)
        else:
            # Penalty for sales significantly below prediction
            sales_reward = -0.5 * (1.0 - sales_ratio)
            
        # Conversion rate component
        conversion_reward = 0.2 * conversion_bonus
        
        # Exploration bonus for trying new price points
        exploration_bonus = 0.0
        if new_price > self.historical_median_price * 1.1:  # 10% above median
            exploration_bonus = 0.1
            
        # Combine rewards
        reward = (
            price_reward +
            sales_reward +
            conversion_reward +
            exploration_bonus
        )
        
        # Clip final reward
        reward = np.clip(reward, -1.0, 1.0)
        
        # Update current state
        self.current_state = next_state
        
        # Check if episode should end
        done = False
        truncated = False
        
        return next_state, float(reward), done, truncated, {
            'price': new_price,
            'sales': current_sales,
            'predicted_sales': predicted_sales,
            'price_ratio': price_ratio,
            'sales_ratio': sales_ratio,
            'organic_conversion': organic_conversion,
            'ad_conversion': ad_conversion
        }
        
    def render(self):
        """Render the environment."""
        pass 