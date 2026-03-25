# nlp_dashboard.py - FINAL JOKER-THEMED INTERACTIVE DASHBOARD
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time

# ------------------- CUSTOM CSS (JOKER THEME) -------------------
st.set_page_config(layout="wide", page_title="Bipolar NLP Dashboard", page_icon="🃏")
st.markdown(
    """
<style>
    .stApp {
        background: linear-gradient(135deg, #1A0033, #4B0082);
        color: #FFFFFF;
    }
    .sidebar .sidebar-content {
        background: #4B0082;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,255,0,0.3);
    }
    .stButton>button {
        background: linear-gradient(45deg, #32CD32, #228B22);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(50,205,50,0.4);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(50,205,50,0.6);
    }
    .card {
        background: rgba(138,43,226,0.3);
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #32CD32;
        box-shadow: 0 4px 20px rgba(50,205,50,0.2);
        margin: 10px 0;
    }
    .highlight {
        background: #32CD32 !important;
        color: black !important;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)
# ------------------- LOAD DATA & MODELS -------------------
@st.cache_resource
def load_resources():
    df = pd.read_csv("project_files/final_project_data.csv")
    model = joblib.load("model/mood_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    le = joblib.load("model/label_encoder.pkl")
    nlp_spacy = spacy.load("en_core_web_sm")
    vader = SentimentIntensityAnalyzer()
    return df, model, scaler, le, nlp_spacy, vader


df, model, scaler, le, nlp_spacy, vader = load_resources()

# GoEmotions labels (28)
EMOTION_LABELS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]

# Bipolar symptom keywords
SYMPTOM_KEYWORDS = [
    "mania",
    "manic",
    "depression",
    "depressive",
    "insomnia",
    "headache",
    "nausea",
    "hopelessness",
    "energy",
    "irritability",
    "dysphoria",
    "panic",
    "hallucination",
    "delusion",
    "anxiety",
    "lethargy",
    "fatigue",
    "disorientation",
    "confusion",
    "agitation",
    "mood swing",
    "pale pupil",
    "tired",
    "sick",
    "negative thought",
    "loss",
    "euphoric",
    "dysphoric",
    "grief",
    "relief",
    "racing thought",
    "no sleep",
]



# ------------------- SIDEBAR -------------------
st.sidebar.title("🃏 NLP Dashboard Controls")
patient_id = st.sidebar.selectbox("Select Patient ID", sorted(df["patient_id"].unique()))
st.sidebar.markdown("---")
st.sidebar.markdown("**Model Accuracy: 0.65**")
st.sidebar.markdown("**F1-Score: 0.64**")
st.sidebar.markdown("**Emotions Detected: 28**")

# Filter data
patient_data = df[df["patient_id"] == patient_id].sort_values("session")

# ------------------- MAIN TITLE -------------------
st.markdown(
    "<h1 style='text-align: center; color: #32CD32; text-shadow: 0 0 10px #32CD32;'>"
    "Predicting Bipolar Disorder using NLP</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h3 style='text-align: center; color: #8A2BE2;'>Emotion Trajectory & Real-Time Prediction</h3>",
    unsafe_allow_html=True,
)

# ------------------- CARD LAYOUT -------------------
col1, col2 = st.columns(2)

with col1:
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🎭 Emotion Detection (28 Labels)")
        emotion_sample = patient_data[["session"]].copy()
        for label in ["joy", "sadness", "anger", "fear", "excitement"]:
            col_name = f"emotion_{label}"
            if col_name in patient_data.columns:
                emotion_sample[label] = patient_data[col_name]
        st.dataframe(
            emotion_sample.style.applymap(
                lambda x: "background: #32CD32; color: black" if isinstance(x, (int, float)) and x > 0.5 else ""
            ),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("💬 Dialogue Act & Symptoms")
        cols = [c for c in ["session", "dialogue_act", "symptom_list"] if c in patient_data.columns]
        st.dataframe(patient_data[cols], use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------- EMOTION TRAJECTORY CHART -------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📈 Emotion Trajectory Over Sessions")
emotion_cols = [c for c in [f"emotion_{k}" for k in ["joy", "sadness", "anger", "fear", "excitement", "love"]] if c in patient_data.columns]
if emotion_cols:
    fig = px.line(
        patient_data,
        x="session",
        y=emotion_cols,
        title="How Emotions Evolve",
        labels={"value": "Emotion Score", "session": "Therapy Session"},
        hover_data=["mood", "predicted_mood"],
    )
    fig.update_traces(line=dict(width=4))
    fig.update_layout(template="plotly_dark", legend_title="Emotions", hovermode="x unified", height=500)
    st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ------------------- REAL-TIME PREDICTION -------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🔮 Real-Time Mood Prediction")
new_text = st.text_area(
    "Enter patient dialogue:",
    height=100,
    placeholder="e.g., I feel full of energy and can't sleep...",
)

if new_text.strip():
    with st.spinner("Analyzing emotions & symptoms..."):
        time.sleep(1)

        clean = " ".join(
            [t.lemma_ for t in nlp_spacy(new_text) if not t.is_stop and not t.is_punct]
        )

        emotions = {label: 0.0 for label in EMOTION_LABELS}
        emotions["joy"] = 0.9 if ("energy" in clean or "excited" in clean) else 0.1
        emotions["sadness"] = 0.8 if ("sad" in clean or "hopeless" in clean) else 0.1
        emotions["anger"] = 0.7 if ("angry" in clean or "irritable" in clean) else 0.1
        emotions["fear"] = 0.6 if ("panic" in clean or "anxious" in clean) else 0.1

        sentiment = vader.polarity_scores(clean)
        compound = sentiment["compound"]

        symptoms = [k for k in SYMPTOM_KEYWORDS if k in clean.lower()]
        symp_count = len(symptoms)

        dialogue_act = 2
        if "?" in new_text:
            dialogue_act = 1

        feature_vec = []
        for label in EMOTION_LABELS:
            feature_vec.append(emotions.get(label, 0.0))
        for _ in EMOTION_LABELS:
            feature_vec.append(0.0)
        feature_vec.extend([compound, 0.0, symp_count, dialogue_act])

        X_new = scaler.transform([feature_vec])
        probas = model.predict_proba(X_new)[0]
        pred_idx = probas.argmax()
        predicted = le.classes_[pred_idx]
        confidence = probas[pred_idx] * 100

        st.success(f"**Predicted Mood: {predicted.upper()}**")
        st.metric("Confidence", f"{confidence:.1f}%")
        if predicted == "manic":
            st.warning("⚠️ High energy detected — possible manic episode")
        elif predicted == "depressive":
            st.error("🆘 Low mood detected — possible depressive episode")
        else:
            st.info("✅ Mood appears stable")

        colf1, colf2 = st.columns(2)
        with colf1:
            st.write("**Detected Emotions**")
            top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:5]
            for emo, score in top_emotions:
                if score > 0.3:
                    st.write(f"- {emo}: {score:.2f}")
        with colf2:
            st.write("**Detected Symptoms**")
            if symptoms:
                for s in symptoms[:5]:
                    st.write(f"- {s}")
            else:
                st.write("_No strong symptoms_")

st.markdown("</div>", unsafe_allow_html=True)

# ------------------- DOWNLOAD BUTTON -------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📥 Download Patient Report")
csv = patient_data.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV",
    data=csv,
    file_name=f"patient_{patient_id}_report.csv",
    mime="text/csv",
)
st.markdown("</div>", unsafe_allow_html=True)

# ------------------- FOOTER -------------------
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #8A2BE2;'>"
    "Built with ❤️ by Rishikesh M | 22MIA1163 | VIT Chennai</p>",
    unsafe_allow_html=True,
)
