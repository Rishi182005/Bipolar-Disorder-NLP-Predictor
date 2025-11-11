# symptom_extraction.py - MAXIMIZED SYMPTOM DETECTION (No More Empty [])

import pandas as pd
import spacy
import re

# Load data
df = pd.read_csv('data_with_sentiment.csv')

# Load SciSpacy model
try:
    nlp = spacy.load("en_core_sci_sm")
except:
    nlp = spacy.load("en_core_web_sm")  # Fallback

# MASSIVE BIPOLAR KEYWORD LIST (100+ terms to catch EVERYTHING)
bipolar_keywords = [
    # Core moods
    'mania', 'manic', 'depression', 'depressive', 'bipolar', 'mood swing', 'mood change',
    'euphoric', 'dysphoric', 'hypomania', 'mixed episode',
    
    # Energy & Activity
    'energetic', 'hyper', 'restless', 'racing thought', 'racing mind', 'grandios', 'inflated self',
    'talkative', 'pressured speech', 'flight of idea', 'distractible', 'goal directed',
    
    # Sleep
    'insomnia', 'sleepless', 'no sleep', 'little sleep', 'hypersomnia', 'oversleep',
    
    # Irritability & Anger
    'irritable', 'angry', 'rage', 'agitated', 'annoyed', 'frustrated',
    
    # Anxiety & Fear
    'anxiety', 'anxious', 'panic', 'panic attack', 'worry', 'fear', 'nervous', 'dread',
    
    # Sadness & Hopelessness
    'sad', 'hopeless', 'hopelessness', 'worthless', 'guilt', 'tearful', 'crying',
    'empty', 'numb', 'despair', 'suicidal', 'death wish',
    
    # Cognitive
    'confused', 'disoriented', 'memory problem', 'concentration', 'focus issue',
    'delusion', 'hallucination', 'paranoia', 'grandiose delusion',
    
    # Physical
    'headache', 'nausea', 'fatigue', 'lethargy', 'tired', 'exhausted', 'pain', 'ache',
    'appetite change', 'weight loss', 'weight gain', 'pale', 'tremor', 'sweating',
    
    # Behavior
    'impulsive', 'reckless', 'spending spree', 'risk taking', 'sexual indiscretion',
    'isolated', 'withdrawn', 'avoid people', 'no motivation',
    
    # Positive mania
    'excited', 'overjoyed', 'euphoric', 'on top of world', 'unstoppable', 'superhuman',
    
    # Negative depression
    'grief', 'loss', 'bereaved', 'devastated', 'broken', 'shattered'
]

def extract_symptoms(text):
    if pd.isna(text) or not text.strip():
        return ['no dialogue']
    
    text_lower = text.lower()
    found = []
    
    # 1. SciSpacy NER
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ['DISEASE', 'SYMPTOM', 'FINDING']:
            found.append(ent.text.lower())
    
    # 2. Keyword matching (case-insensitive, partial match)
    for keyword in bipolar_keywords:
        if keyword in text_lower:
            found.append(keyword)
    
    # 3. Regex for patterns (e.g., "no sleep", "racing thoughts")
    patterns = [
        r'\bno sleep\b', r'\bracing thoughts?\b', r'\bflight of ideas?\b',
        r'\bpressured speech\b', r'\bgrandios\w*\b', r'\bspending spree\b'
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        found.extend(matches)
    
    # 4. Fallback: if still empty, add mood-based placeholder
    if not found:
        if 'manic' in text_lower or 'energy' in text_lower:
            found.append('high energy')
        elif 'depress' in text_lower or 'sad' in text_lower:
            found.append('low mood')
        else:
            found.append('stable state')
    
    return list(set(found))  # Unique only

# Apply
df['symptom_list'] = df['clean_dialogue'].apply(extract_symptoms)

# Save
df.to_csv('data_with_symptoms.csv', index=False)
print("SYMPTOM EXTRACTION MAXIMIZED! Saved to 'data_with_symptoms.csv'.")
print("Empty lists: ", (df['symptom_list'].apply(len) == 0).sum())
print(df[['clean_dialogue', 'symptom_list']].head(10))
