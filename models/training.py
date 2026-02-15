import os
import pandas as pd
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

def load_wdbc_data():
    if os.path.exists("data/wdbc.data"):
        file_input_path = "data/wdbc.data"
    else:
        file_input_path = os.path.join("..", "data", "wdbc.data")
    
    file_input_path = os.path.abspath(file_input_path)
    print(os.getcwd())
    print("Loading dataset from:", file_input_path)

    breast_cancer_data = pd.read_csv(file_input_path, header=None)
    print("✅ Dataset loaded successfully. Shape:", breast_cancer_data.shape)

    feature_list = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
        "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
        "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
        "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
    ]

    breast_cancer_data.columns = ["id", "diagnosis"] + feature_list
    breast_cancer_data['diagnosis'] = breast_cancer_data['diagnosis'].map({'M': 1, 'B': 0})

    return breast_cancer_data


def main():
    breast_cancer_data = load_wdbc_data()
    feature_cols = breast_cancer_data.drop(columns=["id", "diagnosis"])
    target_col = breast_cancer_data["diagnosis"]

    X_tr, X_te, y_tr, y_te = train_test_split(feature_cols, target_col, test_size=0.2, random_state=42, stratify=target_col)

    work_dir = Path.cwd()
    model_save_path = work_dir / "savedModels"
    model_save_path.mkdir(parents=True, exist_ok=True)

    data_dir = work_dir / "data"
    print("Data directory:", data_dir)
    data_dir.mkdir(exist_ok=True)

    X_tr = X_tr.reset_index(drop=True)
    X_te = X_te.reset_index(drop=True)
    y_tr = y_tr.reset_index(drop=True)
    y_te = y_te.reset_index(drop=True)

    train_data = pd.concat([X_tr, y_tr], axis=1)
    test_data = pd.concat([X_te, y_te], axis=1)

    train_data.to_csv(data_dir / "train_data.csv", index=False)
    test_data.to_csv(data_dir / "test_data.csv", index=False)
    print("✅ Datasets saved successfully.")

    model_dict = {
        "logistic_regression": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, C=1.2, solver="liblinear"))]),
        "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "knn": Pipeline([("scaler", StandardScaler()),("clf", KNeighborsClassifier(n_neighbors=7, weights="distance"))]),
        "naive_bayes": GaussianNB(),
        "random_forest": RandomForestClassifier(n_estimators=250, max_depth=8, random_state=42),
        "xg_boost": XGBClassifier(n_estimators=250, learning_rate=0.07, max_depth=4, subsample=0.9, colsample_bytree=0.9, random_state=42, eval_metric="logloss")
    }

    for name, inst in model_dict.items():
        inst.fit(X_tr, y_tr)
        joblib.dump(inst, model_save_path / f"{name}.pkl")
        print(f"✅ Saved model: {name}.pkl")


if __name__ == "__main__":
    main()

