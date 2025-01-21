"""Script to run complete pricing optimization experiments."""

import argparse
import os
from train import train_model
from evaluate import evaluate_model
from utils import setup_logging, create_experiment_dir
import logging

def run_experiment(
    data_path: str,
    experiment_name: str,
    base_dir: str = 'experiments',
    num_eval_episodes: int = 10
):
    """Run a complete experiment including training and evaluation."""
    # Create experiment directory
    experiment_dir = create_experiment_dir(base_dir, experiment_name)
    
    # Setup logging
    log_file = os.path.join(experiment_dir, 'experiment.log')
    setup_logging(log_file)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Experiment directory: {experiment_dir}")
    
    try:
        # Define paths
        model_path = os.path.join(experiment_dir, 'model.zip')
        log_dir = os.path.join(experiment_dir, 'logs')
        
        # Train model
        logger.info("Starting model training...")
        train_model(data_path, model_path, log_dir)
        logger.info("Model training completed")
        
        # Evaluate model
        logger.info("Starting model evaluation...")
        evaluate_model(data_path, model_path, num_eval_episodes)
        logger.info("Model evaluation completed")
        
        logger.info(f"Experiment completed successfully. Results saved in {experiment_dir}")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Run pricing optimization experiment')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the historical data CSV file')
    parser.add_argument('--experiment_name', type=str, required=True,
                      help='Name of the experiment')
    parser.add_argument('--base_dir', type=str, default='experiments',
                      help='Base directory for experiments')
    parser.add_argument('--num_eval_episodes', type=int, default=10,
                      help='Number of evaluation episodes')
    
    args = parser.parse_args()
    run_experiment(
        args.data_path,
        args.experiment_name,
        args.base_dir,
        args.num_eval_episodes
    )

if __name__ == '__main__':
    main() 