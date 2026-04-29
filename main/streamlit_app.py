import logging
from pathlib import Path
from typing import Optional
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from config import (MODEL_PATH, CLEANED_DATA_PATH, PREDICTIONS_OUTPUT_PATH,
                    CONFUSION_MATRIX_PATH, FEATURE_IMPORTANCE_PATH)

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Census Income Prediction Dashboard", layout="wide")

# === Sidebar ===
st.sidebar.image("https://avatars.githubusercontent.com/u/39598206?v=4", width=100)
st.sidebar.title("Census Income Dashboard")
st.sidebar.markdown(
    """
    **Project by [Kunal Jha](https://www.linkedin.com/in/kunaljhadgtl/)**
    Data Science Portfolio Project
    [GitHub Repo](https://github.com/yokunal/census-income-analysis)
    """
)

# --- Upload Section
st.sidebar.header("Predict on Your Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV for Prediction", type=['csv'])

# --- Model Selection (extend as more models added)
model_option = st.sidebar.selectbox("Select Model for Prediction", ["Support Vector Machine (SVM)"])
model = None
if model_option == "Support Vector Machine (SVM)":
    try:
        model = joblib.load(MODEL_PATH)
        st.sidebar.success("SVM model loaded")
    except FileNotFoundError as e:
        st.sidebar.warning(f"Model file not found: {e}")
        logger.warning(f"Model not found: {e}")
    except Exception as e:
        st.sidebar.warning(f"Model file could not be loaded: {e}")
        logger.error(f"Unexpected error loading model: {e}", exc_info=True)

# --- Section Selection
section = st.sidebar.radio("Go to section:", [
    "Project Overview",
    "Data Preview",
    "Exploratory Data Analysis",
    "Statistical Analysis",
    "Model Results",
    "Business Insights"
])


@st.cache_data
def load_data() -> Optional[pd.DataFrame]:
    """
    Load the cleaned dataset for display in the dashboard.

    Returns:
        pd.DataFrame or None if file not found.
    """
    try:
        df = pd.read_csv(CLEANED_DATA_PATH)
        return df
    except FileNotFoundError as e:
        st.error(f"cleaned_data.csv not found at {CLEANED_DATA_PATH}. Run eda_pipeline.py first.")
        logger.error(f"File not found: {e}")
        return None
    except Exception as e:
        st.error(f"Error reading cleaned_data.csv: {e}")
        logger.error(f"Unexpected error loading data: {e}", exc_info=True)
        return None


# --- PREDICTION on Uploaded Data
if uploaded_file is not None:
    try:
        user_df = pd.read_csv(uploaded_file)
        st.write("**Uploaded Data Preview:**", user_df.head())
        if model is not None:
            try:
                preds = model.predict(user_df)
                st.write("**Predicted Income:**")
                st.write(preds)
            except Exception as e:
                st.warning("Uploaded data shape/columns may not match model requirements. Please preprocess or check columns.")
                logger.warning(f"Prediction failed: {e}")
        else:
            st.warning("Model not loaded. Skipping prediction.")
    except Exception as ex:
        st.error(f"Error reading uploaded file: {ex}")
        logger.error(f"Error reading uploaded file: {ex}", exc_info=True)

# --- Main Data Load
df = load_data()
if df is None:
    st.stop()

# === Project Overview ===
if section == "Project Overview":
    st.title("Income Prediction Analysis - Machine Learning Pipeline")
    st.markdown("""
    - **Business Problem**: Predict if income > $50,000 based on census data
    - **Tech Stack**: Python, pandas, scikit-learn, Streamlit
    - **Key Steps**: EDA, statistical tests, predictive modeling, business insights
    """)
    st.info("Best Model: Support Vector Machine (SVM)\nAccuracy: 85%, Weighted F1: 0.84")

# === Data Preview ===
elif section == "Data Preview":
    st.title("Dataset Preview")
    st.write(df.head(20))
    st.markdown(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
    st.subheader("Missing Value Summary")
    st.write(df.isnull().sum())
    st.subheader("Value Counts for Each Feature")
    for col in st.multiselect("Select features to view value counts:", df.columns, default=[]):
        st.write(f"**{col}**:")
        st.write(df[col].value_counts())

# === EDA ===
elif section == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis (EDA)")
    plot_type = st.selectbox("Type of plot", [
        "Income Distribution", "Boxplot: Feature vs Income", "Histogram", "Pairplot", "Correlation Heatmap"
    ])

    if plot_type == "Income Distribution":
        fig, ax = plt.subplots()
        sns.countplot(x="income", data=df, ax=ax)
        st.pyplot(fig)

    elif plot_type == "Boxplot: Feature vs Income":
        feature = st.selectbox("Choose feature:", [col for col in df.columns if col not in ["income"]])
        fig, ax = plt.subplots()
        sns.boxplot(x="income", y=feature, data=df, ax=ax)
        st.pyplot(fig)

    elif plot_type == "Histogram":
        feature = st.selectbox("Select feature for histogram:", [col for col in df.columns if df[col].dtype in [np.int64, np.float64]])
        group = st.selectbox("Group by income?", ["No", "Yes"])
        fig, ax = plt.subplots()
        if group == "Yes":
            for label in df["income"].unique():
                ax.hist(df[df["income"]==label][feature], alpha=0.5, label=label)
            ax.legend()
        else:
            ax.hist(df[feature], bins=30)
        st.pyplot(fig)

    elif plot_type == "Pairplot":
        subset = st.multiselect("Select features (min 2):", [col for col in df.columns if df[col].dtype in [np.int64, np.float64]], default=[])
        if len(subset) >= 2:
            fig = sns.pairplot(df, vars=subset, hue="income")
            st.pyplot(fig)
        else:
            st.info("Select at least 2 features.")

    elif plot_type == "Correlation Heatmap":
        fig, ax = plt.subplots()
        sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# === Statistical Analysis ===
elif section == "Statistical Analysis":
    st.title("Statistical Hypothesis Testing Results")
    st.info("Education level, gender, and age group are all significantly correlated with income (p < 0.001)")
    st.write("""
        - Education → Income (**Confirmed**)<br>
        - Gender → Work hours, income (**Confirmed**)<br>
        - Age Group → Income (**Confirmed**)
    """, unsafe_allow_html=True)

# === Model Results ===
elif section == "Model Results":
    st.title("Machine Learning Model Results")
    if os.path.exists(PREDICTIONS_OUTPUT_PATH):
        results_df = pd.read_csv(PREDICTIONS_OUTPUT_PATH)
        st.subheader("Predictions Breakdown")
        st.write(results_df["income"].value_counts())
        if results_df.empty:
            st.warning("No predictions available.")
            st.stop()
        row_choice = st.slider("Pick a row to view details",
                               min_value=0, max_value=len(results_df)-1, value=0)
        row_choice = min(row_choice, len(results_df) - 1)
        st.write("Data Row:", results_df.iloc[row_choice, :])
    else:
        st.warning("No predictions_output.csv found for result breakdown.")

    st.subheader("Confusion Matrix & Metrics")
    if os.path.exists(CONFUSION_MATRIX_PATH):
        st.image(CONFUSION_MATRIX_PATH)
    else:
        st.warning("Confusion matrix image not found. (Generate & save to main/confusion_matrix.png)")
    if os.path.exists(FEATURE_IMPORTANCE_PATH):
        st.subheader("Feature Importance")
        st.image(FEATURE_IMPORTANCE_PATH)
    else:
        st.warning("Feature importance image not found.")

    st.markdown("""
    - **Best Model**: SVM
    - **Accuracy**: 85%
    - **Precision <=50K**: 88%
    - **Precision >50K**: 71%
    - **Weighted F1-Score**: 0.84
    """)
    st.info("The model is conservative in high-income predictions due to class imbalance.")

# === Business Insights ===
elif section == "Business Insights":
    st.title("Business Insights & Recommendations")
    st.markdown("""
    - **76–78%** of predictions fall into <= $50K: reflects strong class imbalance<br>
    - **Education**: Higher levels show 3x higher >$50K rates<br>
    - **Work Hours**: Strong income correlation, actionable for HR/policy<br>
    - **Future Ideas**: Try SMOTE for class balance, deploy as REST API, enhance with deep learning models
    """, unsafe_allow_html=True)

    st.success("This dashboard is ideal for HR, policy makers, and analysts seeking actionable insights from income prediction models!")

# === Footer ===
st.markdown("---")
st.caption("© 2025 by Kunal Jha | Inspired by real-world income analysis challenges | Dataset: UCI ML Repo")
