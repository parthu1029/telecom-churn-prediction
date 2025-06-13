from sklearn.base import ClassifierMixin
from typing import Dict

from utils.utils import setup_logger

logger = setup_logger(__name__)


def train_models(models: Dict[str, ClassifierMixin], X_train, y_train) -> Dict[str, ClassifierMixin]:
    """
    Trains a dictionary of models and returns the trained versions.

    Args:
        models (dict): A dictionary with model names as keys and sklearn model instances as values.
        X_train: Training features
        y_train: Training labels

    Returns:
        dict: A dictionary of trained models.
    """
    trained_models = {}

    for name, model in models.items():
        try:
            logger.info(f"Training model: {name}")
            model.fit(X_train, y_train)
            trained_models[name] = model
            logger.info(f"Model trained successfully: {name}")
        except Exception as e:
            logger.error(f"Error training model {name}: {str(e)}")

    return trained_models


if __name__ == "__main__":
    print("This file is intended to be used as a module, not executed directly.")
