from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from typing import Dict
import os
import torch

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []
        self.prices = []
        self.sales = []
        
    def _on_step(self) -> bool:
        """Called after each step of the environment."""
        info = self.locals['infos'][0]
        self.rewards.append(self.locals['rewards'][0])
        self.prices.append(info['price'])
        self.sales.append(info['sales'])
        
        # Log metrics every 1000 steps
        if len(self.rewards) >= 1000:
            mean_reward = np.mean(self.rewards)
            mean_price = np.mean(self.prices)
            mean_sales = np.mean(self.sales)
            
            self.logger.record('rollout/mean_reward', mean_reward)
            self.logger.record('rollout/mean_price', mean_price)
            self.logger.record('rollout/mean_sales', mean_sales)
            
            self.rewards = []
            self.prices = []
            self.sales = []
        
        return True

class PricingModel:
    def __init__(self, env, log_dir: str = 'logs'):
        """Initialize the pricing model."""
        policy_kwargs = dict(
            net_arch=[64, 64],  # Smaller network architecture
            activation_fn=torch.nn.ReLU
        )
        
        self.model = PPO(
            'MlpPolicy',
            env,
            learning_rate=1e-4,  # Lower learning rate
            n_steps=1024,
            batch_size=64,
            n_epochs=5,
            gamma=0.95,
            gae_lambda=0.9,
            clip_range=0.1,  # Lower clip range
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.005,  # Lower entropy coefficient
            vf_coef=0.5,
            max_grad_norm=0.3,  # Lower gradient clipping
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=0.015,  # Add target KL divergence
            tensorboard_log=log_dir,
            policy_kwargs=policy_kwargs,
            verbose=1
        )
        
    def train(self, total_timesteps: int, callback: BaseCallback = None) -> None:
        """Train the model."""
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                progress_bar=True
            )
        except Exception as e:
            print(f"Training failed with error: {str(e)}")
            raise
        
    def save(self, path: str) -> None:
        """Save the trained model."""
        self.model.save(path)
        
    def load(self, path: str) -> None:
        """Load a trained model."""
        self.model = PPO.load(path)
        
    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict the next action given a state."""
        action, _ = self.model.predict(state, deterministic=True)
        return action 