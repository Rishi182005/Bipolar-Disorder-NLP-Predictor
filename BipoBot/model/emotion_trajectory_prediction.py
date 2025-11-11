# emotion_trajectory_prediction.py - FINAL VERSION with 28 Emotions + Scaling + Stratified Split

import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load data
df = pd.read_csv('final_nlp_features_complete.csv')

# Parse string dicts/lists
df['emotion_dict'] = df['emotion_dict'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df['sentiment_scores'] = df['sentiment_scores'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df['symptom_list'] = df['symptom_list'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

# Extract ALL 28 emotions from GoEmotions
all_emotion_keys = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 
    'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 
    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 
    'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]
for key in all_emotion_keys:
    df[f'emotion_{key}'] = df['emotion_dict'].apply(lambda d: d.get(key, 0.0))

# Sentiment
df['sentiment_compound'] = df['sentiment_scores'].apply(lambda d: d.get('compound', 0.0))

# Symptom count
df['symptom_count'] = df['symptom_list'].apply(len)

# Dialogue act encoding
act_map = {'statement': 0, 'question': 1, 'reveal': 2, 'unknown': -1}
df['dialogue_act_encoded'] = df['dialogue_act'].map(act_map).fillna(-1)

# Sort by patient and session
df = df.sort_values(['patient_id', 'session'])

# Compute Deltas for ALL 28 emotions + sentiment
for key in all_emotion_keys:
    df[f'{key}_delta'] = df.groupby('patient_id')[f'emotion_{key}'].diff().fillna(0)
df['sentiment_delta'] = df.groupby('patient_id')['sentiment_compound'].diff().fillna(0)

# Features (28 emotions + deltas + sentiment + symptoms + dialogue)
emotion_features = [f'emotion_{key}' for key in all_emotion_keys]
delta_features = [f'{key}_delta' for key in all_emotion_keys]
features = emotion_features + delta_features + [
    'sentiment_compound', 'sentiment_delta', 'symptom_count', 'dialogue_act_encoded'
]
X = df[features].fillna(0)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Target
le = LabelEncoder()
y = le.fit_transform(df['mood'])

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print("FINAL EVALUATION METRICS:")
print(f"Accuracy: {accuracy:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Save model and scaler for dashboard
import joblib
joblib.dump(model, 'mood_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')

# Add predictions
df['predicted_mood'] = le.inverse_transform(model.predict(X_scaled))

# Save final data
df.to_csv('final_project_data.csv', index=False)
print("FINAL DATA SAVED: 'final_project_data.csv'")
print("MODELS SAVED: mood_model.pkl, scaler.pkl, label_encoder.pkl")
print(df[['patient_id', 'session', 'mood', 'predicted_mood']].head(10))
