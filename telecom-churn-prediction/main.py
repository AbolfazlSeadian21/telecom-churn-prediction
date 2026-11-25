"""Run the full pipeline: load data -> train models -> evaluate -> save reports."""
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from churn import config, evaluation  # noqa: E402
from churn.data_loader import DataLoader  # noqa: E402
from churn.exceptions import DataLoadError  # noqa: E402
from churn.models import GradientBoostingModel, LogisticRegressionModel, RandomForestModel  # noqa: E402
from churn.preprocessing import Preprocessor  # noqa: E402


def main():
    try:
        df = DataLoader().load()
    except DataLoadError as err:
        print(f"Error: {err}")
        sys.exit(1)

    print(f"Loaded {len(df)} customers | churn rate: {df[config.TARGET].mean():.1%}")
    evaluation.plot_churn_overview(df)

    prep = Preprocessor()
    X_train, X_test, y_train, y_test = prep.split(df)

    models = [
        LogisticRegressionModel(prep.build_transformer(X_train)),
        RandomForestModel(prep.build_transformer(X_train)),
        GradientBoostingModel(prep.build_transformer(X_train)),
    ]

    rows, probas, preds = [], {}, {}
    for model in models:
        model.fit(X_train, y_train)
        pred, proba = model.predict(X_test), model.predict_proba(X_test)
        probas[model.name], preds[model.name] = proba, pred
        rows.append({"Model": model.name, **evaluation.compute_metrics(y_test, pred, proba)})

    results = pd.DataFrame(rows).set_index("Model").round(3)
    print("\n" + results.to_string())

    best_name = results["ROC-AUC"].idxmax()
    best = next(m for m in models if m.name == best_name)
    print(f"\nBest model by ROC-AUC: {best_name}")

    evaluation.plot_roc(probas, y_test)
    evaluation.plot_confusion(y_test, preds[best_name], best_name)
    evaluation.plot_feature_importance(best)

    config.REPORTS_DIR.mkdir(exist_ok=True)
    results.to_csv(config.REPORTS_DIR / "model_comparison.csv")
    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(best.pipeline, config.MODELS_DIR / "best_model.joblib")
    print("Saved: reports/model_comparison.csv, reports/figures/*, models/best_model.joblib")


if __name__ == "__main__":
    main()
