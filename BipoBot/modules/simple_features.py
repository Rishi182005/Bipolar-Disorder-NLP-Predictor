from .emotions_live import LABELS
import numpy as np

ACT_MAP = {'statement':0, 'question':1, 'reveal':2, 'unknown':-1}

def classify_act(text: str) -> str:
    L = text.lower().strip()
    if not L: return 'unknown'
    if L.endswith('?') or any(k in L for k in ['what','how','why','when','where','do you','are you','?']):
        return 'question'
    if any(k in L for k in ['i feel','i am',' my ',' me ','myself','think','believe']):
        return 'reveal'
    if any(k in L for k in ['the ','it is','this is','yes','no','okay']):
        return 'statement'
    return 'statement'

def extract_symptoms(text: str):
    t = text.lower()
    keys = ['insomnia','racing thoughts','racing thought','hopeless','energetic','unstop','cry','sad','fear','panic']
    found = [k for k in keys if k in t]
    if not found:
        if 'manic' in t or 'energy' in t: return ['high energy']
        if 'depress' in t or 'sad' in t: return ['low mood']
        return ['stable state']
    return list(sorted(set(found)))

def build_features(emotions: dict, sentiment: dict, symptoms: list, act: str,
                   prev_emotions: dict|None=None, prev_sent: float|None=None):
    emo = [emotions.get(k,0.0) for k in LABELS]
    deltas = [0.0]*len(LABELS) if prev_emotions is None else [
        emotions.get(k,0.0) - prev_emotions.get(k,0.0) for k in LABELS
    ]
    sent = sentiment.get('compound', 0.0)
    sent_delta = 0.0 if prev_sent is None else (sent - prev_sent)
    symptom_count = len(symptoms or [])
    act_enc = ACT_MAP.get(act, -1)
    return np.array(emo + deltas + [sent, sent_delta, symptom_count, act_enc], dtype=float)
