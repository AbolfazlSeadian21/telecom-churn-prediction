"""Loading and validating the raw dataset."""
from pathlib import Path

import pandas as pd

from . import config
from .exceptions import DataLoadError


class DataLoader:
    """Loads the Telco churn CSV and performs basic cleaning."""

    REQUIRED_COLUMNS = {config.TARGET, config.ID_COLUMN, "tenure", "MonthlyCharges", "TotalCharges"}

    def __init__(self, path: Path = config.DATA_PATH):
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.path)
        except FileNotFoundError as exc:
            raise DataLoadError(f"Dataset not found at {self.path}") from exc
        except pd.errors.ParserError as exc:
            raise DataLoadError(f"Could not parse {self.path}") from exc

        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise DataLoadError(f"Missing required columns: {sorted(missing)}")
        return self._clean(df)

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # TotalCharges contains blank strings for brand-new customers (tenure = 0)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
        df[config.TARGET] = df[config.TARGET].map({"Yes": 1, "No": 0})
        return df.drop_duplicates(subset=config.ID_COLUMN)
