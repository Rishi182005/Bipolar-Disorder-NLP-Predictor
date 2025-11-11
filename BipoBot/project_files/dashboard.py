import streamlit as st
import pandas as pd
import plotly.express as px

# Install dependencies if not already installed: pip install streamlit plotly

# Load data
df = pd.read_csv('final_project_data.csv')

# Sidebar for patient selection
st.sidebar.header("Patient Selection")
patient_id = st.sidebar.selectbox("Select Patient ID", df['patient_id'].unique())

# Filter data
patient_data = df[df['patient_id'] == patient_id].sort_values('session')

# Title
st.title(f"Patient {patient_id} Mood and Emotion Dashboard")

# Mood Prediction and Clinical Data
st.subheader("Mood Prediction and Clinical Overview")
mood_data = patient_data[['session', 'mood', 'predicted_mood', 'progress_score', 'decision']]
st.table(mood_data.style.highlight_max(axis=0))  # Highlight max values for better visualization

# Emotion Trajectories (Interactive Plotly)
st.subheader("Emotion Trajectories")
emotion_cols = [f'emotion_{key}' for key in ['joy', 'sadness', 'fear']]
fig_emotion = px.line(patient_data, x='session', y=emotion_cols, title='Emotion Scores Over Sessions',
                      labels={'session': 'Session', 'value': 'Score'}, hover_data=['mood'],
                      color_discrete_map={'emotion_joy': '#FFD700', 'emotion_sadness': '#1E90FF', 'emotion_fear': '#FF4500'})
fig_emotion.update_layout(legend_title_text='Emotions')
st.plotly_chart(fig_emotion)

# Symptom Count
st.subheader("Symptom Count Per Session")
fig_symptom = px.bar(patient_data, x='session', y='symptom_count', title='Symptom Count',
                     hover_data=['symptom_list'], color='symptom_count',
                     color_continuous_scale=px.colors.sequential.Viridis)
st.plotly_chart(fig_symptom)

# Delta Trends
st.subheader("Emotion Delta Trends")
delta_cols = [f'{key}_delta' for key in ['joy', 'sadness', 'fear']]
fig_delta = px.line(patient_data, x='session', y=delta_cols, title='Emotion Changes Over Sessions',
                    labels={'session': 'Session', 'value': 'Delta'}, hover_data=['predicted_mood'],
                    color_discrete_map={'joy_delta': '#FFD700', 'sadness_delta': '#1E90FF', 'fear_delta': '#FF4500'})
fig_delta.update_layout(legend_title_text='Deltas')
st.plotly_chart(fig_delta)

# Notes Section
st.subheader("Doctor Notes (Preview)")
st.text_area("Notes", value="\n".join(patient_data['doctor_note'].dropna()), height=200, disabled=True)
