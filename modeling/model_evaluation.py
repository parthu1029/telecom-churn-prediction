import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from modeling.models import get_knn_model
from visualization.visualization import plot_confusion_matrix_heatmap, plot_knn_tuning
from utils.utils import setup_logger

logger = setup_logger(__name__)

def evaluate_all_models(models, X_test, y_test):
    results = []

    for name, model in models.items():
        try:
            logger.info(f"Evaluating model: {name}")
            evaluate_model(model = model, X_test = X_test, y_test = y_test, model_name=name, results=results)
            logger.info(f"Model Evaluation successfully: {name}")
        except Exception as e:
            logger.error(f"Error training model {name}: {str(e)}")
    
    return pd.DataFrame(results)


def evaluate_model(model, X_test, y_test, model_name: str, results: list):
    """
    Evaluate the model and store metrics in the results list.
    """
    try:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        logger.info(f"{model_name} - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
        print(f"\nClassification Report for {model_name}:\n{classification_report(y_test, y_pred)}")

        results.append({
            "model": model_name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

        plot_confusion_matrix_heatmap(y_test, y_pred, model_name)

    except Exception as e:
        logger.error(f"Error evaluating model {model_name}: {e}")


def find_best_knn_k(X_train, y_train, max_k: int):
    """
    Cross-validate KNN from k=1 to max_k and return the best k.
    """
    scores = []
    try:
        for k in range(1, max_k + 1):
            knn = get_knn_model(k)
            acc = cross_val_score(knn, X_train, y_train, cv=5).mean()
            scores.append(acc)

        best_k = np.argmax(scores) + 1
        logger.info(f"✅ Best k found: {best_k}")
        plot_knn_tuning(range(1, max_k + 1), scores)

        return best_k

    except Exception as e:
        logger.error(f"Error during KNN tuning: {e}")
        raise


def format_results(results: list) -> pd.DataFrame:
    """
    Converts results list to DataFrame for reporting/export.
    """
    return pd.DataFrame(results).sort_values(by="f1_score", ascending=False).reset_index(drop=True)
