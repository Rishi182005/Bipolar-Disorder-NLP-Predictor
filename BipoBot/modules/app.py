import streamlit as st
import joblib
import time
from modules.preprocess_live import clean_text
from modules.emotions_live import detect_emotions
from modules.sentiment_live import analyze_sentiment
from modules.simple_features import build_features, classify_act, extract_symptoms
import numpy as np

# ------------------------------------------
# PAGE CONFIG
# ------------------------------------------
st.set_page_config(page_title="🧠 BipoBot", page_icon="🧠", layout="wide")

st.markdown("""
    <h2 style='text-align: center; color: #5B84B1;'>🧠 BipoBot – AI Bipolar Mood Companion</h2>
    <p style='text-align:center; font-size:16px; color:gray;'>Talk to me like your therapist. I’ll listen, predict, and protect you from mood crashes.</p>
    <hr>
""", unsafe_allow_html=True)

# ------------------------------------------
# LOAD MODEL ARTIFACTS
# ------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model/mood_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    enc = joblib.load("model/label_encoder.pkl")
    return model, scaler, enc

model, scaler, enc = load_artifacts()

# ------------------------------------------
# SESSION STATE
# ------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "prev_emotions" not in st.session_state:
    st.session_state.prev_emotions = None
if "prev_sent" not in st.session_state:
    st.session_state.prev_sent = None

# ------------------------------------------
# USER CHAT INPUT
# ------------------------------------------
user_text = st.chat_input("🗣️ How are you feeling today?")
if user_text:
    st.session_state.history.append(user_text)

# ------------------------------------------
# CHAT LOOP
# ------------------------------------------
for msg in st.session_state.history[-6:]:
    st.chat_message("user").markdown(f"💬 *{msg}*")

    clean = clean_text(msg)
    emotions = detect_emotions(clean)
    sentiment = analyze_sentiment(clean)
    act = classify_act(clean)
    symptoms = extract_symptoms(clean)

    X = build_features(
        emotions,
        sentiment,
        symptoms,
        act,
        st.session_state.prev_emotions,
        st.session_state.prev_sent
    )

    Xs = scaler.transform([X])
    mood = enc.inverse_transform(model.predict(Xs))[0]
    try:
        conf = model.predict_proba(Xs).max()
    except:
        conf = 0.65

    # ------------------------------------------
    # MOOD COLOR AND EMOJI MAPS
    # ------------------------------------------
    color_map = {
        "manic": "#FFD700",
        "depressive": "#FF6B6B",
        "stable": "#5CB85C"
    }
    emoji_map = {
        "manic": "🔥",
        "depressive": "🌧️",
        "stable": "🌤️"
    }

    # ------------------------------------------
    # DISPLAY RESULT
    # ------------------------------------------
    with st.chat_message("assistant"):
        st.markdown(
            f"<h4 style='color:{color_map.get(mood)};'>{emoji_map.get(mood)} Predicted Mood: {mood.upper()}</h4>",
            unsafe_allow_html=True,
        )
        st.progress(conf)
        st.write("**Confidence:**", round(conf * 100, 2), "%")
        st.write("**Top Emotions:**", dict(sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:5]))
        st.write("**Sentiment Score:**", sentiment.get("compound", 0.0))
        st.write("**Symptoms Detected:**", ", ".join(symptoms))
        st.write("**Dialogue Type:**", act)

        # Coping Tips
        tips = {
            "manic": "🧘 Try 4-7-8 breathing and avoid overstimulation.",
            "depressive": "💬 Reach out to someone or do a small positive activity.",
            "stable": "😊 Keep your healthy habits consistent!"
        }
        st.info(tips.get(mood, "Take a moment to reflect on your emotions."))

    # Update previous session data for delta calculations
    st.session_state.prev_emotions = emotions
    st.session_state.prev_sent = sentiment.get("compound", 0.0)

# ------------------------------------------
# SIDEBAR OPTIONS
# ------------------------------------------
st.sidebar.header("⚙️ Controls")

if st.sidebar.button("🔄 Clear Chat"):
    st.session_state.history = []
    st.session_state.prev_emotions = None
    st.session_state.prev_sent = None
    st.success("Chat cleared! Start fresh.")

if st.sidebar.button("🔴 Stop BipoBot"):
    with st.spinner("Shutting down BipoBot safely..."):
        time.sleep(1)
    st.warning("🛑 BipoBot stopped. You can now close this tab.")
    st.stop()
