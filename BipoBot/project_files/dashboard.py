import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Bipolar Emotion Trajectory", layout="wide")

@st.cache_data
def load_data():
    # CSV inside project_files
    df = pd.read_csv("project_files/final_project_data.csv")
    return df

df = load_data()

st.title("Bipolar Disorder Emotion Trajectory Dashboard")
st.markdown("**Author: Rishikesh M (22MIA1163), VIT Chennai**")

patient_ids = sorted(df["patient_id"].unique())
selected_patient = st.sidebar.selectbox("Select Patient ID", patient_ids)

patient_data = df[df["patient_id"] == selected_patient].sort_values("session")

st.sidebar.markdown("---")
st.sidebar.write(f"Total sessions: {patient_data['session'].nunique()}")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Mood vs Predicted Mood")

    # Encode mood text into numbers just for plotting
    mood_map = {"depressive": 0, "stable": 1, "manic": 2}
    plot_df = patient_data.copy()
    plot_df["mood_code"] = plot_df["mood"].map(mood_map)
    plot_df["pred_mood_code"] = plot_df["predicted_mood"].map(mood_map)

    st.line_chart(
        plot_df.set_index("session")[["mood_code", "pred_mood_code"]],
        height=300,
    )

with col2:
    st.subheader("Sentiment & Symptom Count")
    fig = px.line(
        patient_data,
        x="session",
        y=["sentiment_compound", "symptom_count"],
        labels={"value": "Score / Count", "session": "Session"},
        title="Sentiment and Symptoms over Sessions",
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Raw Records")
st.dataframe(patient_data, use_container_width=True)
