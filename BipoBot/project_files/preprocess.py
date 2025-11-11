# preprocess.py - FIXED VERSION (Handles NaN + Version Warnings)

import pandas as pd
import spacy
import string
from spacy.lang.en.stop_words import STOP_WORDS

# Load spaCy (ignore version warning)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If model not found, download it
    import subprocess
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Enhanced stop words
STOP_WORDS.update(["patient", "session", "therapy", "bipolar", "describe", "symptom", "mood"])

def clean_text(text):
    # Handle NaN and non-string values
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return ""
    
    # Lowercase and remove punctuation
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    
    # Tokenize and lemmatize
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.text not in STOP_WORDS and not token.is_space]
    return " ".join(tokens)

def detect_speaker(text):
    # Handle NaN and non-string values
    if pd.isna(text) or not isinstance(text, str):
        return "unknown"
    
    text_lower = text.lower()
    if any(word in text_lower for word in ["doctor", "therapist"]):
        return "doctor"
    elif any(word in text_lower for word in ["patient", "i ", "my ", "me "]):
        return "patient"
    return "unknown"

# Load data
df = pd.read_csv('synthetic_bd_data(1).csv')

# Apply preprocessing
df['clean_dialogue'] = df['dialogue'].apply(clean_text)
df['speaker'] = df['dialogue'].apply(detect_speaker)

# Save
df.to_csv('preprocessed_bd_data.csv', index=False)
print("✅ Preprocessing COMPLETE! Saved to 'preprocessed_bd_data.csv'")
print("\nFirst 5 rows:")
print(df[['dialogue', 'clean_dialogue', 'speaker']].head())
print(f"\nEmpty dialogues: {df['dialogue'].isna().sum()}")
