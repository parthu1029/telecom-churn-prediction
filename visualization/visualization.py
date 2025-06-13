import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
from tabulate import tabulate
from utils.utils import setup_logger

logger = setup_logger(__name__)


def plot_countplot(df, feature, target='Churn', palette='pastel'):
    """
    Plot countplot of a categorical/binary feature with hue by target.
    """
    try:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x=feature, hue=target, palette=palette)
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.title(f'Count of {feature} by {target}')
        plt.show()
        logger.info(f"Plotted countplot for feature: {feature}")
    except Exception as e:
        logger.error(f"Error plotting countplot for {feature}: {e}")


def plot_histogram(df, feature, bins=30):
    """
    Plot histogram for a numeric feature.
    """
    try:
        plt.figure(figsize=(8, 5))
        plt.hist(df[feature].dropna(), bins=bins, alpha=0.7, color='steelblue')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {feature}')
        plt.show()
        logger.info(f"Plotted histogram for feature: {feature}")
    except Exception as e:
        logger.error(f"Error plotting histogram for {feature}: {e}")


def plot_correlation_heatmap(df, numeric_cols=None, cmap='coolwarm'):
    """
    Plot correlation heatmap for numeric columns.
    """
    try:
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include='number').columns
        corr = df[numeric_cols].corr()

        plt.figure(figsize=(12, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, square=True)
        plt.title('Correlation Heatmap')
        plt.show()
        logger.info("Plotted correlation heatmap")
    except Exception as e:
        logger.error(f"Error plotting correlation heatmap: {e}")


def plot_confusion_matrix_heatmap(y_true, y_pred, labels=None, title='Confusion Matrix'):
    """
    Plot confusion matrix heatmap.
    """
    try:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title(title)
        plt.show()
        logger.info("Plotted confusion matrix heatmap")
    except Exception as e:
        logger.error(f"Error plotting confusion matrix heatmap: {e}")


def display_classification_report(y_true, y_pred):
    """
    Prints classification report (precision, recall, f1, accuracy) as a formatted table.
    """
    try:
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)

        metrics = {
            'Accuracy': [accuracy],
            'Precision': [precision],
            'Recall': [recall],
            'F1 Score': [f1]
        }

        df_metrics = pd.DataFrame(metrics)
        print(tabulate(df_metrics, headers='keys', tablefmt='github', floatfmt=".4f"))
        logger.info("Displayed classification metrics")
        return df_metrics
    except Exception as e:
        logger.error(f"Error displaying classification report: {e}")
        return None

def plot_knn_tuning(k_values, accuracies):
    """
    Plots KNN accuracy vs. k value.
    """    
    try:
        plt.figure(figsize=(10, 4))
        plt.plot(k_values, accuracies, marker='o')
        plt.title("KNN Accuracy vs. k")
        plt.xlabel("k")
        plt.ylabel("Cross-validated Accuracy")
        plt.grid()
        plt.tight_layout()
        plt.show()
        logger.info("Plotted KNN accuracy vs. k value")
    except Exception as e:
        logger.error(f"Error plotting KNN accuracy vs. k value: {e}")