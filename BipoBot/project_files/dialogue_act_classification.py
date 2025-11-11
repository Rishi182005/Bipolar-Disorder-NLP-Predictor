# dialogue_act_classification.py - Enhanced with Hybrid BERT + Heuristic Rules

import pandas as pd
from transformers import pipeline
import re

# Load data
df = pd.read_csv('data_with_symptoms.csv')

# Load lightweight BERT for intent (public, fast)
classifier = pipeline("text-classification", 
                      model="facebook/bart-large-mnli", 
                      device=-1)  # CPU

# Heuristic keywords
question_keywords = ['what', 'how', 'why', 'when', 'where', 'do you', 'are you', '?']
reveal_keywords = ['i feel', 'i am', 'my', 'me', 'myself', 'think', 'believe']
statement_keywords = ['the', 'it is', 'this is', 'yes', 'no', 'okay']

def classify_dialogue_act(text):
    if pd.isna(text) or not text.strip():
        return 'unknown'
    
    text_lower = text.lower()
    
    # 1. Question detection (high precision)
    if any(k in text_lower for k in question_keywords) or text_lower.endswith('?'):
        return 'question'
    
    # 2. Reveal (patient sharing)
    if any(k in text_lower for k in reveal_keywords):
        return 'reveal'
    
    # 3. Statement (default or doctor note)
    if any(k in text_lower for k in statement_keywords):
        return 'statement'
    
    # 4. BERT fallback for ambiguity
    try:
        result = classifier(text, candidate_labels=["question", "statement", "reveal"])
        return result['labels'][0]
    except:
        return 'statement'

# Apply
df['dialogue_act'] = df['clean_dialogue'].apply(classify_dialogue_act)

# Save
df.to_csv('final_nlp_features_complete.csv', index=False)
print("Dialogue act classification complete! Saved to 'final_nlp_features_complete.csv'.")
print(df[['clean_dialogue', 'dialogue_act']].head(10))
