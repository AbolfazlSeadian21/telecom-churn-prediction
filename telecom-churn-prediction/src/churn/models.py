"""Model wrappers (OOP): one base class, several concrete models."""
from abc import ABC, abstractmethod

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import config
from .exceptions import ModelNotTrainedError
from .utils import timer


class BaseChurnModel(ABC):
    """Common interface for all churn models."""

    name = "base"

    def __init__(self, transformer):
        self.transformer = transformer
        self.pipeline = None

    @abstractmethod
    def build_estimator(self):
        """Return an unfitted scikit-learn estimator."""

    @timer
    def fit(self, X, y):
        self.pipeline = Pipeline(
            [("prep", self.transformer), ("model", self.build_estimator())]
        )
        self.pipeline.fit(X, y)
        return self

    def _check(self):
        if self.pipeline is None:
            raise ModelNotTrainedError(f"{self.name} must be fitted before prediction.")

    def predict(self, X) -> np.ndarray:
        self._check()
        return self.pipeline.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        self._check()
        return self.pipeline.predict_proba(X)[:, 1]


class LogisticRegressionModel(BaseChurnModel):
    name = "Logistic Regression"

    def build_estimator(self):
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config.RANDOM_STATE)


class RandomForestModel(BaseChurnModel):
    name = "Random Forest"

    def build_estimator(self):
        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced",
            random_state=config.RANDOM_STATE, n_jobs=-1,
        )


class GradientBoostingModel(BaseChurnModel):
    name = "Gradient Boosting"

    def build_estimator(self):
        return GradientBoostingClassifier(random_state=config.RANDOM_STATE)
