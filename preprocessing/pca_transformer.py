import joblib
from sklearn.decomposition import PCA

class PCATransformer:
    def __init__(self, n_components=10):
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)
        self.is_fitted = False

    def fit(self, X):
        """
        Fit PCA on data X.
        """
        self.pca.fit(X)
        self.is_fitted = True
        return self

    def transform(self, X):
        """
        Transform data using the fitted PCA model.
        """
        if not self.is_fitted:
            raise RuntimeError("PCA transformer not fitted yet. Call fit() first.")
        return self.pca.transform(X)

    def fit_transform(self, X):
        """
        Fit PCA and transform data in one step.
        """
        self.pca.fit(X)
        self.is_fitted = True
        return self.pca.transform(X)

    def save(self, filepath):
        """
        Save the fitted PCA model to disk.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted PCA model.")
        joblib.dump(self.pca, filepath)

    def load(self, filepath):
        """
        Load a saved PCA model from disk.
        """
        self.pca = joblib.load(filepath)
        self.is_fitted = True
        return self

