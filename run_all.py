"""Script to run experiments for both soapnuts and wool balls."""

import os
import subprocess
import logging
from datetime import datetime

def setup_logging():
    """Set up logging configuration."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f'run_all_{timestamp}.log')),
            logging.StreamHandler()
        ]
    )

def run_experiments():
    """Run experiments for both products."""
    logger = logging.getLogger(__name__)
    
    # Create necessary directories
    os.makedirs('experiments', exist_ok=True)
    
    try:
        # Run soapnuts experiment
        logger.info("Starting soapnuts experiment...")
        subprocess.run([
            'python', 'src/run_experiment.py',
            '--data_path', 'data/soapnutshistory.csv',
            '--experiment_name', 'soapnuts',
            '--base_dir', 'experiments',
            '--num_eval_episodes', '10'
        ], check=True)
        logger.info("Soapnuts experiment completed successfully")
        
        # Run wool balls experiment
        logger.info("Starting wool balls experiment...")
        subprocess.run([
            'python', 'src/run_experiment.py',
            '--data_path', 'data/woolballhistory.csv',
            '--experiment_name', 'woolballs',
            '--base_dir', 'experiments',
            '--num_eval_episodes', '10'
        ], check=True)
        logger.info("Wool balls experiment completed successfully")
        
        logger.info("All experiments completed successfully")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Experiment failed with error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise

if __name__ == '__main__':
    setup_logging()
    run_experiments() 