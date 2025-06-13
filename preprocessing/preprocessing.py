import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import yaml

from preprocessing.pca_transformer import PCATransformer
from utils.utils import setup_logger

logger = setup_logger(__name__)

class Preprocessor:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self._load_config()
        self.scaler = None
        self.label_encoders = {}
        self.pca_transformer = None

    def _load_config(self):
        logger.info(f"Loading config from {self.config_path}")
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        return config.get("preprocessing", {})

    def load_data(self, filepath: Path) -> pd.DataFrame:
        logger.info(f"Loading data from {filepath}")
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Data loaded successfully: shape={df.shape}")
            return df
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def drop_columns(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        logger.info(f"Dropping columns: {columns}")
        return df.drop(columns=columns, errors='ignore')

    def encode_binary_columns(self, df: pd.DataFrame, binary_cols: list) -> pd.DataFrame:
        logger.info("Encoding binary columns (Yes/No -> 1/0)")
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0)
        return df

    def encode_gender(self, df: pd.DataFrame, col='gender') -> pd.DataFrame:
        if col in df.columns:
            logger.info(f"Encoding gender column: {col}")
            df[col] = df[col].apply(lambda x: 1 if str(x).strip().lower() == 'male' else 0)
        return df

    def encode_label_columns(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        logger.info(f"Label encoding categorical columns: {cols}")
        for col in cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        return df

    def convert_to_numeric(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        logger.info(f"Converting column to numeric: {col}")
        df[col] = pd.to_numeric(df[col], errors='coerce')
        before_drop = len(df)
        df = df.dropna(subset=[col])
        after_drop = len(df)
        logger.info(f"Dropped {before_drop - after_drop} rows with NaN in {col}")
        return df

    '''def replace_no_phone_service(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        logger.info(f"Replacing 'No phone service' and 'No' with 0 in columns: {cols}")
        for col in cols:
            if col in df.columns:
                df[col] = df[col].replace({'No phone service': 0, 'No': 0, 'Yes': 1})
        return df'''

    def scale_features(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        logger.info(f"Scaling features: {feature_cols}")
        self.scaler = MinMaxScaler()
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        return df

    def apply_pca(self, df: pd.DataFrame, features: list, n_components: int) -> pd.DataFrame:
        logger.info(f"Applying PCA with n_components={n_components} and features {features}")
        self.pca_transformer = PCATransformer(n_components=n_components)
        pca_data = self.pca_transformer.fit_transform(df[features])
        pca_cols = [f"PCA_{i+1}" for i in range(n_components)]
        df_pca = pd.DataFrame(pca_data, columns=pca_cols, index=df.index)
        df = pd.concat([df, df_pca], axis=1)
        logger.info(f"PCA applied and components added to dataframe")
        print(df.head().to_markdown(index=False))
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # Drop columns from config
        drop_cols = self.config.get("drop_columns", [])
        df = self.drop_columns(df, drop_cols)

        # Replace 'No phone service' and 'No' with 0
        '''replace_cols = self.config.get("replace_no_phone_service_cols", [])
        df = self.replace_no_phone_service(df, replace_cols)'''

        # Encode binary features (Yes/No)
        binary_cols = self.config.get("binary_features", [])
        df = self.encode_binary_columns(df, binary_cols)

        # Encode gender column
        gender_col = self.config.get("gender_column", "gender")
        df = self.encode_gender(df, gender_col)

        # Encode categorical label columns
        label_cols = self.config.get("label_encode_features", [])
        df = self.encode_label_columns(df, label_cols)

        # Convert specified columns to numeric and drop NaNs
        numeric_cols = self.config.get("numeric_columns", [])
        for col in numeric_cols:
            df = self.convert_to_numeric(df, col)

        # Scale specified features
        scale_cols = self.config.get("cols_to_scale", [])
        df = self.scale_features(df, scale_cols)

        # Apply PCA
        if self.config.get("apply_pca", False):
            pca_features = df.columns.to_list()
            pca_n_components = self.config.get("pca_n_components", 10)
            df = self.apply_pca(df, pca_features, pca_n_components)

        return df

    def split_data(self, df, target_col: str):
        """
        Splits dataset into X/y and then train/test.
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]
        test_size= self.config.get("test_size", 0.2)
        random_state= self.config.get("seed", 42)
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
