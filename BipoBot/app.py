import streamlit as st
import joblib
from modules.preprocess_live import clean_text
from modules.emotions_live import detect_emotions, LABELS
from modules.sentiment_live import analyze_sentiment
from modules.simple_features import build_features, classify_act, extract_symptoms

st.set_page_config(page_title="BipoBot", page_icon="🧠", layout="wide")
st.title("🧠 BipoBot – AI Bipolar Mood Companion")
st.caption("Talk to me like your therapist. I’ll listen, predict, and protect you from mood crashes.")

@st.cache_resource
def load_artifacts():
    model = joblib.load("model/mood_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    enc = joblib.load("model/label_encoder.pkl")
    return model, scaler, enc

model, scaler, enc = load_artifacts()

if "history" not in st.session_state: st.session_state.history = []
if "prev_emotions" not in st.session_state: st.session_state.prev_emotions = None
if "prev_sent" not in st.session_state: st.session_state.prev_sent = None

user_text = st.chat_input("How are you feeling right now?")
if user_text:
    st.session_state.history.append(user_text)

for msg in st.session_state.history[-6:]:
    st.chat_message("user").write(msg)
    clean = clean_text(msg)
    emotions = detect_emotions(clean)
    sentiment = analyze_sentiment(clean)
    act = classify_act(clean)
    symptoms = extract_symptoms(clean)

    X = build_features(emotions, sentiment, symptoms, act,
                       st.session_state.prev_emotions, st.session_state.prev_sent)
    Xs = scaler.transform([X])
    mood = enc.inverse_transform(model.predict(Xs))[0]
    try:
        conf = model.predict_proba(Xs).max()
    except:
        conf = 0.65

    with st.chat_message("assistant"):
        st.markdown(f"**Predicted mood:** `{mood.upper()}`  (confidence {conf:.2f})")
        st.write("**Top emotions:**", dict(sorted(emotions.items(), key=lambda x:x[1], reverse=True)[:5]))
        st.write("**Sentiment:**", sentiment.get('compound', 0.0))
        st.write("**Symptoms:**", symptoms)
        st.write("**Dialogue type:**", act)
        tip = {
            "manic":"Try 4-7-8 breathing; dim lights & avoid screens for 30 mins.",
            "depressive":"Do a 5-minute activity: drink water, small walk, or text a friend.",
            "stable":"Great! Keep routine: regular sleep & hydration."
        }.get(mood, "Practice slow deep breathing for 2 minutes.")
        st.success("💡 Coping Tip: " + tip)

    st.session_state.prev_emotions = emotions
    st.session_state.prev_sent = sentiment.get('compound', 0.0)
