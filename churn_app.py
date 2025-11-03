
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📉 Customer Churn Prediction")
st.caption("This page should always render. If something fails, an error box will show below.")

def show_err(e, where=""):
    st.error(f"⚠️ {where}: {type(e).__name__}: {e}")

st.sidebar.header("Status")
st.sidebar.success("App loaded. Choose data below.")

# Tabs for data input
data_tab, upload_tab = st.tabs(["Use Local CSV (Telco-Customer-Churn.csv)", "Upload CSV"])

df_clean = None

def clean_encode(df: pd.DataFrame):
    # Coerce numerics & fill
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Encode categoricals
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
    return df

with data_tab:
    if st.button("Load local Telco-Customer-Churn.csv"):
        try:
            raw = pd.read_csv("Telco-Customer-Churn.csv")
            st.success(f"Loaded local CSV with shape {raw.shape}")
            st.dataframe(raw.head(), use_container_width=True)
            df_clean = clean_encode(raw.copy())
        except Exception as e:
            show_err(e, "Loading local CSV")

with upload_tab:
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try:
            raw = pd.read_csv(up)
            st.success(f"Uploaded CSV with shape {raw.shape}")
            st.dataframe(raw.head(), use_container_width=True)
            df_clean = clean_encode(raw.copy())
        except Exception as e:
            show_err(e, "Reading uploaded CSV")

if df_clean is None:
    st.info("Load or upload a dataset to enable training.")
else:
    try:
        if "Churn" not in df_clean.columns:
            raise ValueError("Dataset must contain a 'Churn' column.")
        X = df_clean.drop("Churn", axis=1)
        y = df_clean["Churn"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None
        )

        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        model = XGBClassifier(random_state=42)
        model.fit(X_train, y_train)

        st.success("Model trained.")

        # Metrics (safe)
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
        y_pred = model.predict(X_test)
        st.subheader("Results")
        col1, col2 = st.columns(2)
        with col1:
            st.text("Classification Report")
            st.text(classification_report(y_test, y_pred, zero_division=0))
        with col2:
            st.text("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            im = ax.imshow(cm, interpolation="nearest")
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j], ha="center", va="center")
            st.pyplot(fig)

        # Optional ROC-AUC if probabilities available
        try:
            y_proba = model.predict_proba(X_test)[:,1]
            auc = roc_auc_score(y_test, y_proba)
            st.metric("ROC-AUC", f"{auc:.3f}")
        except Exception as e:
            st.caption("ROC-AUC not available for this model build.")
    except Exception as e:
        show_err(e, "Training/Evaluation")
