import logging
import yaml
import os
import random
import numpy as np

def setup_logger(name=__name__, level=logging.INFO, log_file=None):
    """
    Setup and return a logger.
    If log_file is specified, logs will be written there too.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding multiple handlers in interactive environments
    if not logger.hasHandlers():
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger


def load_config(config_path):
    """
    Load YAML config file and return dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def set_seed(seed=42):
    """
    Set seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)


def create_dir_if_not_exists(dir_path):
    """
    Create directory if it does not exist.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
