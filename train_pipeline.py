from pathlib import Path
from utils.utils import setup_logger, load_config, set_seed
from preprocessing.preprocessing import Preprocessor
from modeling.models import get_all_models
from modeling.model_evaluation import evaluate_all_models
from modeling.model_training import train_models

logger = setup_logger("train_pipeline")


def run_pipeline():
    logger.info("Starting training pipeline...")

    # Load config
    config_path = Path("config/config.yaml")
    config = load_config(config_path)

    # Set seed
    set_seed(config.get("seed", 42))

    # Get paths and parameters from config
    data_path = Path(config["data"]["file_path"])
    target_column = config["data"]["target_column"]

    # Load and preprocess data
    preprocessor = Preprocessor(config_path)
    df_raw = preprocessor.load_data(data_path)
    df_processed = preprocessor.preprocess(df_raw)

    # Ensure target column exists
    if target_column not in df_processed.columns:
        raise ValueError(f"Target column '{target_column}' is missing after preprocessing.")

    X_train, X_test, y_train, y_test = preprocessor.split_data(df= df_processed, target_col=target_column)

    # Load and evaluate models
    models = get_all_models()

    # Inside run_pipeline()
    trained_models = train_models(models, X_train, y_train)
    results_df = evaluate_all_models(trained_models, X_test, y_test)

    # Display final metrics
    logger.info("Final model performance metrics:")
    print(results_df.to_markdown(index=False))

    logger.info("Training pipeline completed successfully.")
