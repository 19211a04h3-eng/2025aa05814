import streamlit as st
import pandas as pd
import joblib
import os
from pathlib import Path
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Model Evaluator - Classification Analytics", layout="wide")

st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🤖 AI Model Evaluator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: #555;'>Advanced Classification Model Testing & Performance Analytics</p>", unsafe_allow_html=True)
st.divider()

app_base_path = Path(__file__).parent
model_dir_path = app_base_path / "models" / "savedModels"

if not model_dir_path.exists():
    st.error("❌ savedModels folder not found.")
    st.stop()

model_file_list = list(model_dir_path.glob("*.pkl"))

if not model_file_list:
    st.error("❌ No trained models found.")
    st.stop()

model_options = [file.stem for file in model_file_list]

sel_col_main, sel_col_count = st.columns([2, 1])
with sel_col_main:
    chosen_model = st.selectbox("🔧 Select Classification Model", model_options, help="Choose a trained model for evaluation")
with sel_col_count:
    st.metric(label="Models Available", value=len(model_options))

metric_csv_path = app_base_path / "models" / "model_metrics.csv"

if metric_csv_path.exists():
    metric_data = pd.read_csv(metric_csv_path)

    with st.expander("📊 View All Models Performance", expanded=False):
        st.dataframe(metric_data, use_container_width=True)

else:
    st.warning("⚠ model_metrics.csv not found. Run evaluate_models.py first.")

st.subheader("📁 Test Data Upload")
csv_input = st.file_uploader("📤 Upload Test CSV File", type=["csv"], help="Select a CSV file with 'diagnosis' column")

if csv_input is not None:

    input_data = pd.read_csv(csv_input)

    with st.expander("👁️ Data Preview", expanded=True):
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.write(f"**Total Records:** {len(input_data)}")
            st.write(f"**Total Features:** {len(input_data.columns)}")
        with info_col2:
            st.write(f"**Diagnosis Classes:** {input_data['diagnosis'].nunique()}")
            st.write(f"**Class Distribution:** M={len(input_data[input_data['diagnosis']==1])}, B={len(input_data[input_data['diagnosis']==0])}")
        st.dataframe(input_data.head(10), use_container_width=True)

    if "diagnosis" not in input_data.columns:
        st.error("❌ Uploaded file must contain 'diagnosis' column.")
        st.stop()

    X = input_data.drop(columns=["diagnosis"])
    y_true = input_data["diagnosis"]

    model_pkl_path = model_dir_path / f"{chosen_model}.pkl"

    if not model_pkl_path.exists():
        st.error("❌ Selected model file not found.")
        st.stop()

    classifier = joblib.load(model_pkl_path)

    y_pred = classifier.predict(X)

    st.divider()
    st.subheader(f"📊 Model Performance: {chosen_model.replace('_', ' ').title()}")

    acc_score = accuracy_score(y_true, y_pred)
    prec_score = precision_score(y_true, y_pred)
    rec_score = recall_score(y_true, y_pred)
    f1_score_val = f1_score(y_true, y_pred)
    mcc_score = matthews_corrcoef(y_true, y_pred)

    if hasattr(classifier, "predict_proba"):
        prob_scores = classifier.predict_proba(X)[:, 1]
        auc_score = roc_auc_score(y_true, prob_scores)
    else:
        auc_score = None

    scores_dict = {
        "Accuracy": acc_score,
        "AUC": auc_score,
        "Precision": prec_score,
        "Recall": rec_score,
        "F1 Score": f1_score_val,
        "MCC": mcc_score
    }

    metrics_results_df = pd.DataFrame(scores_dict.items(), columns=["Metric", "Value"])
    
    col_group1 = st.columns(3)
    score_vals = list(scores_dict.values())
    score_names = list(scores_dict.keys())
    
    for idx, col in enumerate(col_group1):
        if idx < len(score_names):
            with col:
                if score_vals[idx] is not None:
                    st.metric(label=score_names[idx], value=f"{score_vals[idx]:.4f}")
                else:
                    st.metric(label=score_names[idx], value="N/A")
    
    col_group2 = st.columns(3)
    for idx, col in enumerate(col_group2):
        if idx + 3 < len(score_names):
            with col:
                if score_vals[idx + 3] is not None:
                    st.metric(label=score_names[idx + 3], value=f"{score_vals[idx + 3]:.4f}")
                else:
                    st.metric(label=score_names[idx + 3], value="N/A")

    st.subheader("🔍 Confusion Matrix Analysis")

    cm = confusion_matrix(y_true, y_pred)

    cm_visual_col, cm_stats_col = st.columns([2, 1])
    
    with cm_visual_col:
        cm_fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="RdYlGn", ax=ax, cbar_kws={'label': 'Count'})
        ax.set_xlabel("Predicted Label", fontsize=12, fontweight='bold')
        ax.set_ylabel("Actual Label", fontsize=12, fontweight='bold')
        ax.set_xticklabels(['Benign', 'Malignant'])
        ax.set_yticklabels(['Benign', 'Malignant'])
        st.pyplot(cm_fig)
    
    with cm_stats_col:
        true_neg, false_pos, false_neg, true_pos = cm.ravel()
        st.metric("True Negatives", true_neg)
        st.metric("False Positives", false_pos)
        st.metric("False Negatives", false_neg)
        st.metric("True Positives", true_pos)

    st.subheader("📋 Classification Report")

    class_report = classification_report(y_true, y_pred, output_dict=True)
    class_report_data = pd.DataFrame(class_report).transpose()

    st.dataframe(class_report_data, use_container_width=True)

else:
    st.info("📝 **Getting Started:** Upload a CSV test file to evaluate model performance and view detailed analytics.")
