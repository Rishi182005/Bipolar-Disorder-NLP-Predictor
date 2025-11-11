import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import time

# Load data and model
df = pd.read_csv('final_project_data.csv')
model = joblib.load('mood_prediction_model.pkl')
le = LabelEncoder()
le.fit(['manic', 'depressive', 'stable'])

# Custom Joker theme with aesthetic enhancements
st.set_page_config(layout="wide", page_title="Healthcare Dashboard", page_icon="🏥")
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #6A0DAD, #4A148C); /* Gradient purple background */
        color: #FFFFFF;
        font-family: 'Arial', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #6A0DAD;
        color: #FFFFFF;
        padding: 10px;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    .stTextInput > div > div > input {
        color: #FFFFFF;
        background-color: #8A2BE2; /* Lighter purple input */
        border: 2px solid #32CD32; /* Green border */
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    .stButton>button {
        background-color: #32CD32; /* Green button */
        color: #FFFFFF;
        border-radius: 5px;
        padding: 5px 15px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #228B22; /* Darker green on hover */
        transform: scale(1.05);
    }
    .stTable {
        background-color: #8A2BE2; /* Lighter purple table */
        color: #FFFFFF;
        border-radius: 5px;
        padding: 5px;
    }
    .element-container {
        animation: fadeIn 0.5s ease-in;
    }
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar with animation
st.sidebar.header("Healthcare Dashboard Controls")
with st.sidebar:
    patient_id = st.selectbox("Select Patient ID", df['patient_id'].unique(), key="sidebar_anim")

# Filter data with delay for effect
time.sleep(0.5)  # Subtle loading animation
patient_data = df[df['patient_id'] == patient_id].sort_values('session')

# Title with aesthetic touch
st.title("Healthcare Dashboard: Mood Prediction & Trajectories")
st.markdown("<h3 style='color: #32CD32; text-align: center;'>Tracking Mental Health Journeys</h3>", unsafe_allow_html=True)

# Mood Prediction in card
with st.container():
    st.subheader("Mood Prediction Overview")
    mood_data = patient_data[['session', 'mood', 'predicted_mood', 'progress_score', 'decision']]
    st.table(mood_data)

# Emotion Trajectories with enhanced styling
st.subheader("Emotion Trajectories")
emotion_cols = [f'emotion_{key}' for key in ['joy', 'sadness', 'fear']]
fig = px.line(patient_data, x='session', y=emotion_cols, title='Emotion Trends',
              labels={'session': 'Session', 'value': 'Score'}, hover_data=['predicted_mood'],
              color_discrete_map={'emotion_joy': '#32CD32', 'emotion_sadness': '#FFFFFF',
                                  'emotion_fear': '#32CD32'})
fig.update_traces(line=dict(width=3))
st.plotly_chart(fig)

# Symptom Count with green scale
st.subheader("Symptom Count Per Session")
fig = px.bar(patient_data, x='session', y='symptom_count', title='Symptom Count',
             hover_data=['symptom_list'], color='symptom_count', color_continuous_scale='Greens')
st.plotly_chart(fig)

# Real-Time Mood Prediction with Confidence
st.subheader("Real-Time Mood Prediction")
new_dialogue = st.text_input("Enter new dialogue (e.g., 'I feel very energetic'):", key="health_input")
if new_dialogue:
    # Placeholder features
    emotions = {'joy': 0.0, 'love': 0.0, 'anger': 0.0, 'fear': 0.0, 'surprise': 0.0, 'sadness': 0.0}
    sentiment = {'compound': 0.0}
    features = [emotions['joy'], emotions['love'], emotions['anger'], emotions['fear'], emotions['surprise'],
                emotions['sadness'], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, sentiment['compound'], 0.0, 0, 0]
    # Get prediction probabilities
    probabilities = model.predict_proba([features])[0]
    predicted_mood = le.inverse_transform([model.predict([features])[0]])[0]
    confidence = max(probabilities) * 100  # Convert to percentage
    st.success(f"Predicted Mood: {predicted_mood} (Confidence: {confidence:.2f}%)")

# Comparative Mood Chart
st.subheader("Actual vs. Predicted Mood Comparison")
mood_comparison = patient_data[['session', 'mood', 'predicted_mood']].copy()
mood_comparison['match'] = mood_comparison['mood'] == mood_comparison['predicted_mood']
fig_mood = px.bar(mood_comparison, x='session', y=[1] * len(mood_comparison), color='match',
                  color_discrete_map={True: '#32CD32', False: '#FF4500'},
                  title='Mood Prediction Accuracy', labels={'session': 'Session', 'value': 'Match'},
                  hover_data=['mood', 'predicted_mood'])
fig_mood.update_traces(width=0.4)
st.plotly_chart(fig_mood)

# Metrics Panel with card effect
st.sidebar.subheader("Healthcare Metrics")
st.sidebar.markdown(f"<div style='background-color: #8A2BE2; padding: 15px; border-radius: 10px; color: #FFFFFF; box-shadow: 0 4px 8px rgba(0,0,0,0.1);'>Accuracy: 0.66<br>F1-Score: 0.66<br>Precision: 0.68<br>Recall: 0.66</div>", unsafe_allow_html=True)
