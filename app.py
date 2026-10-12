
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "student_model.pkl"
DATA_PATH = BASE_DIR / "student_performance.csv"

# --------------------------------------------------
# 2. PROFESSIONAL CSS DESIGN
# --------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }

    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    .hero {
        padding: 28px;
        border-radius: 18px;
        background: linear-gradient(120deg, #172554, #312e81, #164e63);
        border: 1px solid #334155;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: #ffffff;
        font-size: 2.15rem;
        margin-bottom: 8px;
    }

    .hero p {
        color: #dbeafe;
        font-size: 1rem;
        margin-bottom: 0;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 12px;
        margin-bottom: 14px;
    }

    div[data-testid="stMetric"] {
        background: #171c27;
        border: 1px solid #293548;
        padding: 18px;
        border-radius: 14px;
    }

    div[data-testid="stMetricLabel"] {
        color: #cbd5e1;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        min-height: 45px;
        font-weight: 600;
    }

    .note {
        padding: 13px 16px;
        border-radius: 10px;
        background: #172033;
        border: 1px solid #334155;
        color: #cbd5e1;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. LOAD DATA AND MODEL
# --------------------------------------------------
@st.cache_data
def load_dataset(path):
    return pd.read_csv(path)


@st.cache_resource
def load_model(path, modified_time):
    # modified_time allows the model to reload if the file changes
    return joblib.load(path)


if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Run train_model.ipynb "
        "and make sure student_model.pkl is in this folder."
    )
    st.stop()

if not DATA_PATH.exists():
    st.error(
        "Dataset not found. Make sure student_performance.csv "
        "is in the same folder as app.py."
    )
    st.stop()

try:
    df = load_dataset(DATA_PATH)
    artifact = load_model(
        MODEL_PATH,
        MODEL_PATH.stat().st_mtime
    )
    model = artifact["model"]
except Exception as error:
    st.error(f"Could not load the project files: {error}")
    st.stop()


FEATURES = [
    "study_hours",
    "attendance",
    "previous_marks"
]
TARGET = "final_marks"

required_columns = FEATURES + [TARGET]

if not all(column in df.columns for column in required_columns):
    st.error(
        "The CSV must contain: study_hours, attendance, "
        "previous_marks, and final_marks."
    )
    st.stop()

df = df[required_columns].apply(pd.to_numeric, errors="coerce")
df = df.replace([np.inf, -np.inf], np.nan).dropna()

if df.empty:
    st.error("The dataset has no valid records to display.")
    st.stop()


# --------------------------------------------------
# 4. SIDEBAR NAVIGATION
# --------------------------------------------------
st.sidebar.title("📊 Student Analytics")
st.sidebar.caption("Performance monitoring system")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard Overview",
        "Predict Performance",
        "Data Analysis",
        "Model Information"
    ]
)

st.sidebar.divider()
st.sidebar.write("**Project:** Student Performance Prediction")
st.sidebar.caption("Built with Python, Pandas, scikit-learn and Streamlit")


# --------------------------------------------------
# 5. HEADER
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>Student Performance Analytics</h1>
    <p>Explore academic patterns, analyze student data,
    and estimate final examination marks.</p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 6. DASHBOARD OVERVIEW
# --------------------------------------------------
if page == "Dashboard Overview":

    st.markdown('<div class="section-title">Key Statistics</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Valid Records", f"{len(df):,}")
    c2.metric("Average Final Marks", f"{df[TARGET].mean():.2f}")
    c3.metric("Average Attendance", f"{df['attendance'].mean():.1f}%")
    c4.metric("Average Study Hours", f"{df['study_hours'].mean():.1f} h")

    st.divider()

    left, right = st.columns(2)

    # Final marks distribution
    with left:
        st.subheader("Final Marks Distribution")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(
            df[TARGET],
            bins=20,
            edgecolor="white",
            alpha=0.85
        )
        ax.set_xlabel("Final Marks")
        ax.set_ylabel("Number of Records")
        ax.set_title("Distribution of Final Marks")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Average marks by study-hour group
    with right:
        st.subheader("Study Hours vs Final Marks")

        chart_data = df[["study_hours", TARGET]].copy()
        chart_data["Study Group"] = pd.cut(
            chart_data["study_hours"],
            bins=[-0.01, 2, 4, 6, 8, 24],
            labels=[
                "0–2 hours",
                "2–4 hours",
                "4–6 hours",
                "6–8 hours",
                "8+ hours"
            ],
            duplicates="drop"
        )

        grouped = chart_data.groupby(
            "Study Group", observed=False
        )[TARGET].mean().dropna()

        if not grouped.empty:
            st.bar_chart(grouped)
        else:
            st.info("Not enough study-hour variation for this chart.")

    st.divider()

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown(
        '<div class="note">These records are synthetic sample data. '
        'The statistics describe this dataset, not necessarily real students.</div>',
        unsafe_allow_html=True
    )


# --------------------------------------------------
# 7. PREDICT PERFORMANCE
# --------------------------------------------------
elif page == "Predict Performance":

    st.subheader("Estimate Final Examination Marks")
    st.write("Enter the student's information below.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            study_hours = st.number_input(
                "Study hours per day",
                min_value=0.0,
                max_value=24.0,
                value=5.0,
                step=0.5
            )

        with col2:
            attendance = st.number_input(
                "Attendance (%)",
                min_value=0.0,
                max_value=100.0,
                value=85.0,
                step=1.0
            )

        with col3:
            previous_marks = st.number_input(
                "Previous exam marks",
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=1.0
            )

        submitted = st.form_submit_button(
            "Predict Final Marks",
            type="primary"
        )

    if submitted:
        student = pd.DataFrame([{
            "study_hours": study_hours,
            "attendance": attendance,
            "previous_marks": previous_marks
        }], columns=FEATURES)

        try:
            prediction = float(model.predict(student)[0])
            prediction = max(0.0, min(100.0, prediction))

            st.divider()
            st.subheader("Prediction Result")

            result_col, detail_col = st.columns([1, 2])

            with result_col:
                st.metric(
                    "Estimated Final Marks",
                    f"{prediction:.2f} / 100"
                )

                st.progress(prediction / 100.0)

            with detail_col:
                st.write("**Input Summary**")
                st.write(f"- Study hours: {study_hours:.1f} per day")
                st.write(f"- Attendance: {attendance:.1f}%")
                st.write(f"- Previous marks: {previous_marks:.1f}")

                if prediction >= 75:
                    st.success("Estimated range: High marks")
                elif prediction >= 50:
                    st.info("Estimated range: Moderate marks")
                else:
                    st.warning("Estimated range: Lower marks")

            report = pd.DataFrame([{
                "study_hours": study_hours,
                "attendance": attendance,
                "previous_marks": previous_marks,
                "predicted_final_marks": round(prediction, 2),
                "model": artifact.get("model_name", "Loaded model")
            }])

            st.download_button(
                "Download Prediction Report (CSV)",
                data=report.to_csv(index=False).encode("utf-8"),
                file_name="student_prediction_report.csv",
                mime="text/csv"
            )

            st.caption(
                "This prediction is an estimate based on a model trained "
                "on synthetic data. It is not a guaranteed examination result."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# --------------------------------------------------
# 8. DATA ANALYSIS
# --------------------------------------------------
elif page == "Data Analysis":

    st.subheader("Explore Student Data")

    col1, col2 = st.columns(2)

    with col1:
        selected_column = st.selectbox(
            "Choose a variable to analyze",
            required_columns,
            index=3
        )

    with col2:
        bins = st.slider(
            "Histogram bins",
            min_value=5,
            max_value=40,
            value=20
        )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(
        df[selected_column],
        bins=bins,
        edgecolor="white",
        alpha=0.85
    )
    ax.set_title(f"Distribution of {selected_column.replace('_', ' ').title()}")
    ax.set_xlabel(selected_column.replace("_", " ").title())
    ax.set_ylabel("Number of Records")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Feature Correlation Heatmap")

    correlation = df[required_columns].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    image = ax.imshow(
        correlation,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="auto"
    )

    ax.set_xticks(range(len(correlation.columns)))
    ax.set_xticklabels(
        correlation.columns,
        rotation=30,
        ha="right"
    )
    ax.set_yticks(range(len(correlation.index)))
    ax.set_yticklabels(correlation.index)

    for i in range(len(correlation.index)):
        for j in range(len(correlation.columns)):
            ax.text(
                j, i,
                f"{correlation.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="black"
            )

    fig.colorbar(image, ax=ax, label="Correlation")
    ax.set_title("Correlation Between Student Variables")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe().round(2), use_container_width=True)

    st.download_button(
        "Download Clean Dataset (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="student_performance_clean.csv",
        mime="text/csv"
    )


# --------------------------------------------------
# 9. MODEL INFORMATION
# --------------------------------------------------
elif page == "Model Information":

    st.subheader("Machine Learning Model")

    st.write(
        "**Selected model:**",
        artifact.get("model_name", type(model).__name__)
    )

    st.write("**Input features:**")
    for feature in FEATURES:
        st.write(f"- `{feature}`")

    st.write("**Prediction target:** `final_marks`")

    st.write(
        "**Problem type:** Supervised learning — regression"
    )

    st.write(
        "**Saved model file:** `student_model.pkl`"
    )

    st.divider()
    st.subheader("Evaluation Metrics")

    metrics = artifact.get("metrics")

    if isinstance(metrics, dict) and metrics:
        metric_cols = st.columns(len(metrics))

        for col, (name, value) in zip(metric_cols, metrics.items()):
            try:
                col.metric(name, f"{float(value):.4f}")
            except (TypeError, ValueError):
                col.metric(name, str(value))
    else:
        st.info(
            "Evaluation metrics were not stored in the model artifact. "
            "Use the evaluation results from train_model.ipynb to report "
            "MAE, RMSE and R². Do not treat training-set scores as test scores."
        )

    st.subheader("How the System Works")

    st.markdown("""
    1. Load the student dataset.
    2. Clean and analyze the data.
    3. Train and evaluate regression models.
    4. Save the selected model.
    5. Collect student details through this dashboard.
    6. Generate and download a predicted-marks report.
    """)

    st.markdown(
        '<div class="note">Educational prototype: the model uses '
        'synthetic data and has not been validated for real-world '
        'student assessment.</div>',
        unsafe_allow_html=True
    )


# --------------------------------------------------
# 10. FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "Student Performance Analytics | Python • Pandas • "
    "scikit-learn • Matplotlib • Streamlit"
)
