"""Custom exceptions used across the project."""


class ChurnError(Exception):
    """Base class for project errors."""


class DataLoadError(ChurnError):
    """Raised when the dataset cannot be loaded or is invalid."""


class ModelNotTrainedError(ChurnError):
    """Raised when a model is used before being trained."""
