# emotion_detection.py - Enhanced with 28 Emotions from GoEmotions (Public Model)

import pandas as pd
from transformers import pipeline

# Load data
df = pd.read_csv('preprocessed_bd_data.csv')

# Use public GoEmotions model (28 emotions, multi-label)
emotion_classifier = pipeline('text-classification', 
                              model='SamLowe/roberta-base-go_emotions', 
                              top_k=None,  # Get scores for all 28 emotions
                              device=-1)  # CPU (change to 0 for GPU if available)

def detect_emotions(text):
    if pd.isna(text) or not text.strip():
        # Default zero for all 28 emotions
        labels = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 
                  'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 
                  'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 
                  'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']
        return {label: 0.0 for label in labels}
    try:
        results = emotion_classifier(text)[0]
        return {item['label']: round(item['score'], 4) for item in results}
    except Exception as e:
        print(f"Error processing: {e}")
        return {label: 0.0 for label in [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 
            'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 
            'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 
            'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]}

# Apply to clean_dialogue
df['emotion_dict'] = df['clean_dialogue'].apply(detect_emotions)

# Save
df.to_csv('data_with_emotions.csv', index=False)
print("Emotion detection complete with 28 emotions! Saved to 'data_with_emotions.csv'.")
print(df[['clean_dialogue', 'emotion_dict']].head(5))
