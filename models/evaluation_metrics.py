import os
from pathlib import Path
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

def load_test_data():
    base_dir = Path.cwd()
    test_data_path = base_dir / "data" / "test_data.csv"

    if not test_data_path.exists():
        raise FileNotFoundError(f"Test data not found at: {test_data_path}")

    test_df = pd.read_csv(test_data_path)

    X_test = test_df.drop(columns=["diagnosis"])
    y_test = test_df["diagnosis"]

    return X_test, y_test


def load_models():
    base_dir = Path.cwd()
    models_dir = base_dir / "savedModels"

    if not models_dir.exists():
        raise FileNotFoundError(f"Model directory not found at: {models_dir}")

    models_dict = {}

    for model_f in models_dir.glob("*.pkl"):
        model_nm = model_f.stem
        models_dict[model_nm] = joblib.load(model_f)

    if not models_dict:
        raise ValueError("No models found in savedModels directory.")

    return models_dict


def evaluate_models():
    X_test, y_test = load_test_data()
    models_dict = load_models()

    results_list = []

    for model_nm, model_obj in models_dict.items():
        y_pred = model_obj.predict(X_test)

        if hasattr(model_obj, "predict_proba"):
            y_prob = model_obj.predict_proba(X_test)[:, 1]
            auc_val = roc_auc_score(y_test, y_prob)
        else:
            auc_val = None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1_val = f1_score(y_test, y_pred)
        mcc_val = matthews_corrcoef(y_test, y_pred)

        results_list.append([
            model_nm,
            acc,
            auc_val,
            prec,
            rec,
            f1_val,
            mcc_val
        ])

    results_df = pd.DataFrame(
        results_list,
        columns=["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    )

    return results_df


def main():
    print("🔎 Evaluating models...\n")

    results_df = evaluate_models()

    print("✅ Evaluation Completed.\n")
    print(results_df)

    base_dir = Path.cwd()
    results_df.to_csv(base_dir / "model_metrics.csv", index=False)
    print("\n📁 Metrics saved as model_metrics.csv")


if __name__ == "__main__":
    main()
