"""Metrics and plots."""
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)

from . import config


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
    }


def plot_churn_overview(df: pd.DataFrame):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    rate = df.groupby("Contract")[config.TARGET].mean().mul(100)
    rate.plot(kind="bar", ax=axes[0], color="#8a8a8a")
    axes[0].set(title="Churn rate by contract type (%)", xlabel="")
    axes[0].tick_params(axis="x", rotation=0)
    for label, grp in df.groupby(config.TARGET):
        axes[1].hist(grp["tenure"], bins=30, alpha=0.6, label="Churned" if label else "Stayed")
    axes[1].set(title="Tenure distribution (months)", xlabel="Tenure")
    axes[1].legend()
    for label, grp in df.groupby(config.TARGET):
        axes[2].hist(grp["MonthlyCharges"], bins=30, alpha=0.6, label="Churned" if label else "Stayed")
    axes[2].set(title="Monthly charges distribution", xlabel="Monthly charges")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "01_churn_overview.png", dpi=150)
    plt.close(fig)


def plot_roc(results: dict, y_test):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, proba in results.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, proba):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(title="ROC curves", xlabel="False positive rate", ylabel="True positive rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "02_roc_curves.png", dpi=150)
    plt.close(fig)


def plot_confusion(y_true, y_pred, name: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Stayed", "Churned"], cmap="Greys", ax=ax
    )
    ax.set_title(f"Confusion matrix - {name}")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "03_confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(model, top_n: int = 15):
    pipe = model.pipeline
    names = pipe.named_steps["prep"].get_feature_names_out()
    est = pipe.named_steps["model"]
    values = est.feature_importances_ if hasattr(est, "feature_importances_") else np.abs(est.coef_[0])
    s = pd.Series(values, index=[n.split("__", 1)[1] for n in names]).nlargest(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    s.plot(kind="barh", ax=ax, color="#8a8a8a")
    ax.set_title(f"Top {top_n} features - {model.name}")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "04_feature_importance.png", dpi=150)
    plt.close(fig)
