"""Feature preprocessing pipeline."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


class Preprocessor:
    """Splits data and builds the scikit-learn preprocessing transformer."""

    def __init__(self, test_size: float = config.TEST_SIZE, random_state: int = config.RANDOM_STATE):
        self.test_size = test_size
        self.random_state = random_state

    def split(self, df: pd.DataFrame):
        X = df.drop(columns=[config.TARGET, config.ID_COLUMN])
        y = df[config.TARGET]
        return train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

    @staticmethod
    def build_transformer(X: pd.DataFrame) -> ColumnTransformer:
        numeric = X.select_dtypes(include="number").columns.tolist()
        categorical = X.select_dtypes(exclude="number").columns.tolist()
        return ColumnTransformer(
            [
                ("num", Pipeline([("scale", StandardScaler())]), numeric),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ]
        )
