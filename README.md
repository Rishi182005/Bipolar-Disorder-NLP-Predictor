Bipolar Disorder Emotion Trajectory Prediction using NLP
This project builds an end‑to‑end NLP pipeline to model bipolar disorder mood trajectories from psychotherapy session transcripts. It extracts emotions, symptoms, and dialogue acts from session text, trains a mood classifier, and visualizes patient journeys through interactive Streamlit dashboards.

Features
Synthetic bipolar therapy dialogue generation using GPT‑2.
​

Text preprocessing with spaCy, custom stopwords, speaker detection, and symptom extraction.

Emotion extraction (28 GoEmotions labels), sentiment analysis, and dialogue act tagging.

Feature engineering with emotion deltas, sentiment deltas, symptom counts, and dialogue acts, consolidated into a single feature table.

Mood classification (manic / depressive / stable) using Logistic Regression with standardized features.
​

Multiple dashboards (all under project_files/):

dashboard.py: core mood and emotion trajectory dashboard.
​

healthcare_dashboard.py: healthcare‑oriented view with metrics and actual vs. predicted mood trajectories.
​

nlp_dashboard.py: Joker‑themed advanced NLP dashboard with real‑time prediction and patient reports.
​

Project Structure
All main scripts live in the project_files/ folder:

project_files/data_gen.py – Generates synthetic bipolar session dialogues → project_files/synthetic_bd_data.csv.

project_files/preprocess.py – Cleans dialogues, lemmatizes text, detects speaker → project_files/preprocessed_bd_data.csv.

project_files/emotion_detection.py – Adds GoEmotions emotion scores → project_files/data_with_emotions.csv.

project_files/sentiment_analysis.py – Adds VADER sentiment scores → project_files/data_with_sentiment.csv.

project_files/symptom_extraction.py – Adds keyword‑based symptom features → project_files/data_with_symptoms.csv.

project_files/consolidate_features.py – Merges emotions, sentiment, and symptoms → project_files/final_nlp_features_complete.csv.

project_files/dialogue_act_classification.py / project_files/clinicalbert_symptom.py – Additional experimental feature scripts (dialogue acts, ClinicalBERT symptoms).

emotion_trajectory_prediction.py – Trains the Logistic Regression model on the final features, saves:

model/mood_model.pkl

model/scaler.pkl

model/label_encoder.pkl

project_files/final_project_data.csv (with mood and predicted_mood).

Dashboards (all read from project_files/final_project_data.csv and model/):

project_files/dashboard.py.
​

project_files/healthcare_dashboard.py.
​

project_files/nlp_dashboard.py.
​

Key datasets:

project_files/final_nlp_features_complete.csv – Feature table used for training.
​

project_files/final_project_data.csv – Final dataset with true and predicted moods per patient/session, used by dashboards.
​

Tech Stack
Language: Python 3.x

NLP: spaCy (en_core_web_sm), VADER Sentiment.

Modeling: scikit‑learn (LogisticRegression, StandardScaler, LabelEncoder), joblib for model persistence.
​

Dashboards: Streamlit, Plotly Express.

Generation: Hugging Face transformers (GPT‑2 text generation).
​

Setup
Clone the repository and enter the project folder:

bash
git clone <your-repo-url>.git
cd Bipolar-Disorder-NLP-Predictor/BipoBot
(Recommended) Create and activate a virtual environment:

bash
py -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux / macOS
Install dependencies:

bash
pip install -r requirements.txt
py -m spacy download en_core_web_sm
​

Data & Modeling Pipeline
If you already have the CSVs and model files (as in this repo), you can skip to “Running the Dashboards”.
​

To regenerate everything from scratch, run the scripts in this order from inside BipoBot (with venv active):

1. Generate synthetic data
bash
py project_files\data_gen.py
Outputs: project_files/synthetic_bd_data.csv.

2. Preprocess text
bash
py project_files\preprocess.py
Outputs: project_files/preprocessed_bd_data.csv.

3. NLP feature extraction
bash
py project_files\emotion_detection.py
py project_files\sentiment_analysis.py
py project_files\symptom_extraction.py
py project_files\consolidate_features.py
Outputs:

project_files/data_with_emotions.csv.

project_files/data_with_sentiment.csv.

project_files/data_with_symptoms.csv.

project_files/final_nlp_features_complete.csv.

4. Train mood trajectory model
bash
py emotion_trajectory_prediction.py
Outputs:

model/mood_model.pkl – trained Logistic Regression classifier.

model/scaler.pkl – StandardScaler used for features.

model/label_encoder.pkl – label encoder for mood classes.

project_files/final_project_data.csv – final dataset with mood and predicted_mood per patient/session.

Running the Dashboards
Make sure project_files/final_project_data.csv and the model artifacts in the model/ folder are present.

From inside BipoBot (venv active):

1. Core dashboard
bash
py -m streamlit run project_files\dashboard.py
Select a patient ID from the sidebar.

View mood trajectories, sentiment trends, and symptom counts across sessions.
​

2. Healthcare dashboard
bash
py -m streamlit run project_files\healthcare_dashboard.py
Shows mood prediction overview and actual vs. predicted mood line chart.

Includes symptom count bar chart and key session‑level metrics for clinicians.
​

3. NLP dashboard
bash
py -m streamlit run project_files\nlp_dashboard.py
Joker‑themed interface with:

28‑label emotion table for selected patient sessions.

Emotion trajectory line chart over sessions.

Real‑time mood prediction from user‑entered dialogue text.

Detected symptoms and downloadable per‑patient CSV report.
​

How the Model Works
Input: Per‑session dialogue text from bipolar therapy transcripts (synthetic in this project).!
​

NLP features:

28 GoEmotions emotion probabilities per session.

Emotion deltas across sessions per patient.

VADER sentiment compound score and its session‑to‑session delta.

Symptom count from keyword‑based symptom extraction.

Encoded dialogue act (statement = 0, question = 1, reveal = 2, unknown = −1).

Classifier: Logistic Regression with standardized features, predicting 
m
o
o
d
∈
{
manic
,
depressive
,
stable
}
mood∈{manic,depressive,stable}.
​

emotion_trajectory_prediction.py prints evaluation metrics (accuracy, F1, precision, recall) when you train the model.
​

Limitations & Future Work
Uses synthetic data; performance on real clinical data is unknown.
​

Symptom extraction currently relies on simple keyword matching; could be improved with clinical NER and domain‑specific models.

Potential extensions:

Replace Logistic Regression with transformer‑based sequence models (e.g., BERT, LSTM over sessions).!
​

Add patient‑level risk scores and temporal forecasting.

Add authentication and role‑based access for clinician vs. researcher dashboards.

Author & Acknowledgement
Rishikesh M Ramasubramaniyan (22MIA1163), “Predicting Bipolar Disorder Emotion Trajectories using NLP and Streamlit Dashboards”, VIT Chennai.
​

You are free to adapt this project structure and code for academic, research, or portfolio purposes with appropriate credit.