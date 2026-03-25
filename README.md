# 🧠 BipoBot — Bipolar Disorder Emotion Trajectory Predictor

> An end-to-end NLP pipeline that models bipolar disorder mood trajectories from psychotherapy session transcripts — featuring emotion extraction, mood classification, and interactive Streamlit dashboards.

---

## 📌 Overview

BipoBot processes psychotherapy dialogue transcripts to track and predict patient mood states across sessions. It extracts rich NLP features (emotions, sentiment, symptoms, dialogue acts), trains a mood classifier, and presents patient journeys through three purpose-built dashboards.

**Predicted mood classes:** `manic` · `depressive` · `stable`

---

## ✨ Features

- **Synthetic data generation** — Bipolar therapy dialogues generated via GPT-2
- **Text preprocessing** — Lemmatization, custom stopwords, speaker detection (spaCy)
- **Emotion extraction** — 28-label GoEmotions classification per session
- **Sentiment analysis** — VADER compound scores with session-to-session deltas
- **Symptom extraction** — Keyword-based clinical symptom detection
- **Dialogue act tagging** — Statement / Question / Reveal classification
- **Mood classifier** — Logistic Regression on standardized NLP feature vectors
- **Three dashboards** — Core, Healthcare, and NLP-focused views (Streamlit + Plotly)

---

## 🗂️ Project Structure

```
BipoBot/
├── emotion_trajectory_prediction.py   # Model training entry point
├── requirements.txt
├── model/
│   ├── mood_model.pkl                 # Trained Logistic Regression classifier
│   ├── scaler.pkl                     # StandardScaler
│   └── label_encoder.pkl              # LabelEncoder for mood classes
└── project_files/
    ├── data_gen.py                    # Synthetic dialogue generation
    ├── preprocess.py                  # Text cleaning & preprocessing
    ├── emotion_detection.py           # GoEmotions feature extraction
    ├── sentiment_analysis.py          # VADER sentiment scoring
    ├── symptom_extraction.py          # Keyword-based symptom features
    ├── consolidate_features.py        # Feature table consolidation
    ├── dialogue_act_classification.py # Dialogue act tagging
    ├── clinicalbert_symptom.py        # Experimental ClinicalBERT symptoms
    ├── dashboard.py                   # Core mood & emotion dashboard
    ├── healthcare_dashboard.py        # Clinician-oriented dashboard
    ├── nlp_dashboard.py               # Advanced NLP / real-time dashboard
    ├── synthetic_bd_data.csv
    ├── preprocessed_bd_data.csv
    ├── data_with_emotions.csv
    ├── data_with_sentiment.csv
    ├── data_with_symptoms.csv
    ├── final_nlp_features_complete.csv
    └── final_project_data.csv         # Final dataset used by all dashboards
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.x |
| NLP | spaCy (`en_core_web_sm`), VADER Sentiment, GoEmotions |
| Modeling | scikit-learn (LogisticRegression, StandardScaler, LabelEncoder), joblib |
| Dashboards | Streamlit, Plotly Express |
| Data Generation | Hugging Face Transformers (GPT-2) |

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>.git
cd Bipolar-Disorder-NLP-Predictor/BipoBot
```

### 2. Create and activate a virtual environment

```bash
py -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
py -m spacy download en_core_web_sm
```

---

## 🔄 Data & Modeling Pipeline

> **Skip this section** if the CSV files and `model/` artifacts are already present in the repo — jump straight to [Running the Dashboards](#-running-the-dashboards).

Run the following scripts **in order** from inside `BipoBot/` with the venv active:

#### Step 1 — Generate synthetic data
```bash
py project_files\data_gen.py
```
Output: `project_files/synthetic_bd_data.csv`

#### Step 2 — Preprocess text
```bash
py project_files\preprocess.py
```
Output: `project_files/preprocessed_bd_data.csv`

#### Step 3 — NLP feature extraction
```bash
py project_files\emotion_detection.py
py project_files\sentiment_analysis.py
py project_files\symptom_extraction.py
py project_files\consolidate_features.py
```
Outputs: `data_with_emotions.csv`, `data_with_sentiment.csv`, `data_with_symptoms.csv`, `final_nlp_features_complete.csv`

#### Step 4 — Train the mood classifier
```bash
py emotion_trajectory_prediction.py
```
Outputs:
- `model/mood_model.pkl` — trained Logistic Regression classifier
- `model/scaler.pkl` — fitted StandardScaler
- `model/label_encoder.pkl` — LabelEncoder for mood classes
- `project_files/final_project_data.csv` — final dataset with `mood` and `predicted_mood`

Training also prints evaluation metrics: **accuracy, F1, precision, and recall**.

---

## 📊 Running the Dashboards

Ensure `project_files/final_project_data.csv` and the `model/` artifacts are present, then run from inside `BipoBot/`:

### 1. Core Dashboard
```bash
py -m streamlit run project_files\dashboard.py
```
- Select a patient from the sidebar
- View mood trajectories, sentiment trends, and symptom counts across sessions

### 2. Healthcare Dashboard
```bash
py -m streamlit run project_files\healthcare_dashboard.py
```
- Actual vs. predicted mood line chart
- Symptom count bar chart
- Session-level metrics designed for clinicians

### 3. NLP Dashboard *(Joker-themed)*
```bash
py -m streamlit run project_files\nlp_dashboard.py
```
- 28-label emotion table per patient session
- Emotion trajectory line chart over sessions
- **Real-time mood prediction** from user-entered dialogue text
- Detected symptom display
- Downloadable per-patient CSV reports

---

## 🤖 How the Model Works

**Input:** Per-session dialogue text from bipolar therapy transcripts

**Features extracted per session:**

| Feature | Description |
|---|---|
| Emotion probabilities | 28 GoEmotions labels |
| Emotion deltas | Session-to-session change per patient |
| Sentiment score | VADER compound score |
| Sentiment delta | Session-to-session change |
| Symptom count | Keyword-matched clinical symptoms |
| Dialogue act | Encoded as: statement=0, question=1, reveal=2, unknown=−1 |

**Classifier:** Logistic Regression on StandardScaler-normalized features

**Output:** `mood ∈ {manic, depressive, stable}` per patient per session

---

## ⚠️ Limitations

- Uses **synthetic data** — real-world clinical performance is unknown
- Symptom extraction relies on **keyword matching**; a clinical NER model would be more robust
- Logistic Regression does not model **temporal dependencies** across sessions

---

## 🚀 Future Work

- Replace Logistic Regression with **BERT or LSTM-based sequence models** for temporal session modeling
- Integrate **ClinicalBERT** for domain-specific symptom extraction
- Add **patient-level risk scores** and longitudinal forecasting
- Build **authentication & role-based access** (clinician vs. researcher views)
- Validate pipeline on **real de-identified clinical transcripts**

---

## 👤 Author

**Rishikesh M Ramasubramaniyan** (22MIA1163)  
*"Predicting Bipolar Disorder Emotion Trajectories using NLP and Streamlit Dashboards"*  
VIT Chennai

---

## 📄 License

This project is free to adapt for academic, research, or portfolio purposes **with appropriate credit** to the original author.