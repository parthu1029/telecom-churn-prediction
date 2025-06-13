import yaml
from pathlib import Path

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from utils.utils import setup_logger

logger = setup_logger(__name__)

# Load configuration
CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"
try:
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)
        MODEL_PARAMS = config.get("model_params", {})
except FileNotFoundError as e:
    logger.error(f"Config file not found at {CONFIG_PATH}")
    MODEL_PARAMS = {}

# --- Model Builders ---

def get_logistic_regression():
    params = MODEL_PARAMS.get("logistic_regression", {})
    model = LogisticRegression(max_iter=params.get("max_iter", 100))
    logger.debug(f"Initialized LogisticRegression with: {model.get_params()}")
    return model


def get_ridge_classifier():
    model = RidgeClassifier()
    logger.debug("Initialized RidgeClassifier with default parameters.")
    return model


def get_decision_tree():
    params = MODEL_PARAMS.get("decision_tree", {})
    model = DecisionTreeClassifier(
        criterion=params.get("criterion", "gini"),
        max_depth=params.get("max_depth", 500)
    )
    logger.debug(f"Initialized DecisionTreeClassifier (Gini) with: {model.get_params()}")
    return model


def get_decision_tree_entropy():
    params = MODEL_PARAMS.get("decision_tree_entropy", {})
    model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=params.get("max_depth", 500)
    )
    logger.debug(f"Initialized DecisionTreeClassifier (Entropy) with: {model.get_params()}")
    return model


def get_random_forest():
    params = MODEL_PARAMS.get("random_forest", {})
    model = RandomForestClassifier(
        criterion=params.get("criterion", "gini"),
        max_depth=params.get("max_depth", 500)
    )
    logger.debug(f"Initialized RandomForestClassifier (Gini) with: {model.get_params()}")
    return model


def get_random_forest_entropy():
    params = MODEL_PARAMS.get("random_forest_entropy", {})
    model = RandomForestClassifier(
        criterion="entropy",
        max_depth=params.get("max_depth", 500)
    )
    logger.debug(f"Initialized RandomForestClassifier (Entropy) with: {model.get_params()}")
    return model


def get_svm():
    params = MODEL_PARAMS.get("svm", {})
    model = SVC(
        kernel=params.get("kernel", "rbf"),
        gamma=params.get("gamma", 0.5),
        C=params.get("C", 1.0)
    )
    logger.debug(f"Initialized SVC with: {model.get_params()}")
    return model


def get_knn_model():
    params = MODEL_PARAMS.get("knn", {})
    model = KNeighborsClassifier(
        n_neighbors=params.get("n_neighbors", 5)
    )
    logger.debug(f"Initialized KNeighborsClassifier with: {model.get_params()}")
    return model


def get_naive_bayes():
    model = GaussianNB()
    logger.debug("Initialized GaussianNB with default parameters.")
    return model


# --- Model Aggregator ---

def get_all_models():
    """
    Returns a dictionary of all models initialized with parameters from config.
    """
    return {
        "Logistic Regression": get_logistic_regression(),
        "Ridge Classifier": get_ridge_classifier(),
        "Decision Tree (Gini)": get_decision_tree(),
        "Decision Tree (Entropy)": get_decision_tree_entropy(),
        "Random Forest (Gini)": get_random_forest(),
        "Random Forest (Entropy)": get_random_forest_entropy(),
        "Support Vector Machine": get_svm(),
        "Naive Bayes": get_naive_bayes(),
        "K-Nearest Neighbors": get_knn_model(),
    }


if __name__ == "__main__":
    for name, model in get_all_models().items():
        print(f"{name}: {model.__class__.__name__}")
