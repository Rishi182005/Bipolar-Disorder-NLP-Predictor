from transformers import pipeline
from functools import lru_cache

LABELS = ['admiration','amusement','anger','annoyance','approval','caring','confusion',
          'curiosity','desire','disappointment','disapproval','disgust','embarrassment',
          'excitement','fear','gratitude','grief','joy','love','nervousness','optimism',
          'pride','realization','relief','remorse','sadness','surprise','neutral']

@lru_cache(maxsize=1)
def _clf():
    return pipeline('text-classification', model='SamLowe/roberta-base-go_emotions', top_k=None, device=-1)

def detect_emotions(text: str) -> dict:
    if not text.strip():
        return {k: 0.0 for k in LABELS}
    try:
        res = _clf()(text)[0]
        return {item['label']: round(item['score'],4) for item in res}
    except:
        return {k: 0.0 for k in LABELS}
