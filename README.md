***

# Bipolar Disorder Emotion Trajectory Prediction using NLP

This project builds an end‑to‑end NLP pipeline to model bipolar disorder mood trajectories from psychotherapy session transcripts. It extracts emotions, symptoms, and dialogue acts from session text, trains a mood classifier, and visualizes patient journeys through interactive Streamlit dashboards. 

## Features

- Synthetic bipolar therapy dialogue generation using GPT‑2. 
- Text preprocessing with spaCy, custom stopwords, speaker detection, and symptom extraction. 
- Emotion extraction (28 GoEmotions labels), sentiment analysis, and dialogue act tagging.
- Feature engineering with emotion deltas, sentiment deltas, symptom counts, and dialogue acts.
- Mood classification (manic / depressive / stable) using Logistic Regression with standardized features. 
- Multiple dashboards:
  - `dashboard.py`: core mood and emotion trajectory dashboard. 
  - `healthcare_dashboard.py`: healthcare‑oriented view with prediction confidence and metrics.
  - `nlp_dashboard.py`: Joker‑themed advanced NLP dashboard with real‑time prediction and patient reports.

## Project Structure

- `data_gen.py` – Generates synthetic bipolar session dialogues and saves `synthetic_bd_data.csv`. 
- `preprocess.py` – Cleans dialogues, lemmatizes text, detects speaker, and writes `preprocessed_bd_data.csv`. 
- *(NLP feature script)* – Produces `final_nlp_features_complete.csv` with emotions, sentiment, symptoms, and dialogue acts. 
- `emotion_trajectory_prediction.py` – Trains the Logistic Regression model, saves `mood_model.pkl`, `scaler.pkl`, `labelencoder.pkl`, and `final_project_data.csv` with predicted moods.
- `dashboard.py` – Basic patient mood & emotion trajectory dashboard. 
- `healthcare_dashboard.py` – Healthcare‑style dashboard with metrics and real‑time prediction. 
- `nlp_dashboard.py` – Full NLP dashboard with emotion tables, trajectories, real‑time text input, and report download.
- `final_nlp_features_complete.csv`, `final_project_data.csv` – Main processed datasets used by the model and dashboards. 

## Tech Stack

- **Language:** Python 3.x  
- **NLP:** spaCy (`en_core_web_sm`), VADER Sentiment. 
- **Modeling:** scikit‑learn (LogisticRegression, StandardScaler, LabelEncoder). 
- **Dashboards:** Streamlit, Plotly Express. 
- **Generation:** Hugging Face `transformers` (GPT‑2 text generation).

## Setup

1. Clone the repository and enter the project folder:

   ```bash
   git clone <your-repo-url>.git
   cd <your-repo-folder>
   ```

2. (Recommended) Create and activate a virtual environment.

3. Install dependencies (example):

   ```bash
   pip install -r requirements.txt
   ```

   If you do not have a `requirements.txt` yet, include at least:

   ```bash
   pip install pandas numpy scikit-learn streamlit plotly spacy transformers vaderSentiment joblib
   python -m spacy download en_core_web_sm
   ```

## Data & Modeling Pipeline

Run these scripts in order:

### 1. Generate synthetic data

```bash
python data_gen.py
```

Outputs: `synthetic_bd_data.csv`. 
### 2. Preprocess text

```bash
python preprocess.py
```

Outputs: `preprocessed_bd_data.csv`. 

### 3. NLP feature extraction

Run your feature‑engineering / NLP script that:

- Reads `preprocessed_bd_data.csv`.
- Runs emotion extraction, sentiment analysis, symptom detection, dialogue act tagging.  
- Writes `final_nlp_features_complete.csv`.

### 4. Train mood trajectory model

```bash
python emotion_trajectory_prediction.py
```

Outputs: 

- `mood_model.pkl` – trained Logistic Regression classifier.  
- `scaler.pkl` – StandardScaler used for features.  
- `labelencoder.pkl` – label encoder for mood classes.  
- `final_project_data.csv` – final dataset with `mood` and `predicted_mood` per patient/session.

## Running the Dashboards

Make sure `final_project_data.csv` and the model artifacts (`mood_model.pkl`, `scaler.pkl`, `labelencoder.pkl`) are present in the project root. 

### 1. Core dashboard

```bash
streamlit run dashboard.py
```

- Select a patient ID from the sidebar.  
- View mood predictions, emotion trajectories (joy, sadness, fear), and symptom counts per session. 

### 2. Healthcare dashboard

```bash
streamlit run healthcare_dashboard.py
```

- Shows mood prediction overview, emotion trends, symptom count bar chart.  
- Includes real‑time mood prediction from new dialogue input and an actual vs. predicted mood comparison chart.

### 3. NLP dashboard

```bash
streamlit run nlp_dashboard.py
```

- Joker‑themed interface with: 
  - 28‑label emotion table for selected patient sessions.  
  - Emotion trajectory line chart over sessions.  
  - Real‑time mood prediction from user‑entered dialogue text.  
  - Detected symptoms and downloadable per‑patient CSV report.

## How the Model Works

- **Input:** Per‑session dialogue text from bipolar therapy transcripts. 
- **NLP features:** 
  - 28 GoEmotions emotion probabilities per session.  
  - Emotion deltas (change in each emotion across sessions per patient).  
  - VADER sentiment compound score and its session‑to‑session delta.  
  - Symptom count from keyword‑based symptom extraction.  
  - Encoded dialogue act (statement = 0, question = 1, reveal = 2, unknown = −1).  
- **Classifier:** Logistic Regression with standardized features, predicting `mood ∈ {manic, depressive, stable}`. 

The script prints final evaluation metrics (accuracy, F1, precision, recall) when you run `emotion_trajectory_prediction.py`. 

## Limitations & Future Work

- Uses synthetic data; performance on real clinical data is unknown. 
- Symptom extraction currently relies on simple keyword matching; could be improved with medical NER and more advanced clinical NLP.
  
- Future extensions:
  - Replace Logistic Regression with transformer‑based sequence models (e.g., BERT, LSTM over sessions).  
  - Add patient‑level risk scores and temporal forecasting.  
  - Add authentication and role‑based access for clinician vs. researcher dashboards.

## Author & Acknowledgement

> **Rishikesh M Ramasubramaniyan (22MIA1163), “Predicting Bipolar Disorder Emotion Trajectories using NLP and Streamlit Dashboards”, VIT Chennai.**

You are free to adapt this project structure and code for academic, research, or portfolio purposes with appropriate credit.

***
