import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churn.data_loader import DataLoader  # noqa: E402
from churn.exceptions import DataLoadError, ModelNotTrainedError  # noqa: E402
from churn.models import LogisticRegressionModel  # noqa: E402
from churn.preprocessing import Preprocessor  # noqa: E402


def test_missing_file_raises():
    with pytest.raises(DataLoadError):
        DataLoader("does_not_exist.csv").load()


def test_data_is_clean():
    df = DataLoader().load()
    assert df["TotalCharges"].isna().sum() == 0
    assert set(df["Churn"].unique()) == {0, 1}


def test_predict_before_fit_raises():
    df = DataLoader().load()
    X_train, X_test, _, _ = Preprocessor().split(df)
    model = LogisticRegressionModel(Preprocessor.build_transformer(X_train))
    with pytest.raises(ModelNotTrainedError):
        model.predict(X_test)
