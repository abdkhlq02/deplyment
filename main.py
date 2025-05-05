# main.py

import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.set_page_config(page_title="Federated IDS Evaluation", layout="wide")
st.title("Federated Learning IDS Model Evaluation")
st.write("Project: Federated Learning for Decentralized IDS in IoT Network")

# File upload
st.header("Step 1: Upload Model(s)")
model_file = st.file_uploader("Upload your trained model (.pkl)", type=["pkl"], key="model")

st.header("Step 2 (Optional): Upload Scaler (for SVM only)")
scaler_file = st.file_uploader("Upload your scaler (.pkl) if using SVM model", type=["pkl"], key="scaler")

st.header("Step 3: Upload Test Dataset")
test_data_file = st.file_uploader("Upload your test dataset (.csv)", type=["csv"], key="data")

# Evaluation option
st.header("Step 4: Choose Evaluation Type")
evaluation_option = st.selectbox(
    "Select Evaluation Type",
    ("Accuracy Only", "Full Evaluation (Precision, Recall, F1)", "Confusion Matrix", "ROC Curve (binary only)")
)

if model_file and test_data_file:
    # Load model and dataset
    model = pickle.load(model_file)
    test_data = pd.read_csv(test_data_file)

    # Check for label
    if 'label' not in test_data.columns:
        st.error("The dataset must contain a 'label' column.")
    else:
        X_test = test_data.drop('label', axis=1)
        y_test = test_data['label']

        # If SVM, apply scaler
        if hasattr(model, "predict_proba") is False and scaler_file is not None:
            try:
                scaler = pickle.load(scaler_file)
                X_test = scaler.transform(X_test)
            except Exception as e:
                st.error(f"Failed to apply scaler: {e}")
        elif hasattr(model, "predict_proba") is False and scaler_file is None:
            st.warning("⚠️ SVM model detected. You should upload the associated scaler.")

        # Predict
        try:
            y_pred = model.predict(X_test)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        # Evaluation Options
        if evaluation_option == "Accuracy Only":
            st.subheader("Accuracy")
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"**Accuracy:** {accuracy:.4f}")

        elif evaluation_option == "Full Evaluation (Precision, Recall, F1)":
            st.subheader("Full Evaluation Metrics")
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            st.write(f"**Accuracy:** {accuracy:.4f}")
            st.write(f"**Precision:** {precision:.4f}")
            st.write(f"**Recall:** {recall:.4f}")
            st.write(f"**F1-Score:** {f1:.4f}")

            # Download CSV
            results_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                'Score': [accuracy, precision, recall, f1]
            })
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Evaluation Metrics", csv, "evaluation_metrics.csv", "text/csv")

        elif evaluation_option == "Confusion Matrix":
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        elif evaluation_option == "ROC Curve (binary only)":
            if len(np.unique(y_test)) != 2:
                st.error("ROC Curve is only applicable for binary classification.")
            else:
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)[:, 1]
                else:
                    st.error("Model does not support probability prediction required for ROC Curve.")
                    st.stop()

                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = roc_auc_score(y_test, y_prob)

                fig2, ax2 = plt.subplots()
                ax2.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
                ax2.plot([0, 1], [0, 1], linestyle='--')
                ax2.set_xlabel('False Positive Rate')
                ax2.set_ylabel('True Positive Rate')
                ax2.set_title('Receiver Operating Characteristic (ROC)')
                ax2.legend()
                st.pyplot(fig2)
