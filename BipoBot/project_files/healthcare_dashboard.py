import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(page_title="Healthcare Mood Dashboard", layout="wide")

@st.cache_resource
def load_resources():
    df = pd.read_csv("project_files/final_project_data.csv")
    model = joblib.load("model/mood_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    le = joblib.load("model/label_encoder.pkl")
    return df, model, scaler, le

df, model, scaler, le = load_resources()

st.title("Healthcare-Focused Bipolar Mood Dashboard")
st.markdown("**Author: Rishikesh M (22MIA1163), VIT Chennai**")

patient_ids = sorted(df["patient_id"].unique())
selected_patient = st.sidebar.selectbox("Select Patient ID", patient_ids)

patient_data = df[df["patient_id"] == selected_patient].sort_values("session")

st.sidebar.markdown("---")
st.sidebar.metric("Total Sessions", patient_data["session"].nunique())
st.sidebar.metric(
    "Overall Model Accuracy (demo)", "0.65"
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Actual vs Predicted Mood")

    mood_map = {"depressive": 0, "stable": 1, "manic": 2}
    plot_df = patient_data.copy()
    plot_df["mood_code"] = plot_df["mood"].map(mood_map)
    plot_df["pred_mood_code"] = plot_df["predicted_mood"].map(mood_map)

    fig_mood = px.line(
        plot_df,
        x="session",
        y=["mood_code", "pred_mood_code"],
        labels={"value": "Mood Code", "session": "Session"},
        title="Actual vs Predicted Mood Trajectory",
    )
    st.plotly_chart(fig_mood, use_container_width=True)

with col2:
    st.subheader("Symptom Count per Session")
    fig_sym = px.bar(
        patient_data,
        x="session",
        y="symptom_count",
        labels={"symptom_count": "Symptom Count", "session": "Session"},
        title="Reported Symptoms over Sessions",
    )
    st.plotly_chart(fig_sym, use_container_width=True)

st.subheader("Session-Level Details")
st.dataframe(
    patient_data[
        [
            "patient_id",
            "session",
            "mood",
            "predicted_mood",
            "sentiment_compound",
            "symptom_count",
        ]
    ],
    use_container_width=True,
)
