# sentiment_analysis.py - Enhanced VADER with Booster Words for Mental Health

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load data from emotion detection
df = pd.read_csv('data_with_emotions.csv')

# Enhanced VADER analyzer with custom lexicon for bipolar terms
analyzer = SentimentIntensityAnalyzer()

# Add bipolar-specific boosters (increase intensity for key words)
custom_lexicon = {
    'mania': 2.5, 'depressive': -2.5, 'hopeless': -2.8, 'energetic': 2.0,
    'insomnia': -1.8, 'irritable': -2.0, 'euphoric': 2.2, 'dysphoric': -2.2,
    'panic': -2.5, 'excitement': 2.0, 'grief': -2.8, 'relief': 1.8
}
analyzer.lexicon.update(custom_lexicon)

def analyze_sentiment(text):
    if pd.isna(text) or not text.strip():
        return {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}
    scores = analyzer.polarity_scores(text)
    return {k: round(v, 4) for k, v in scores.items()}

# Apply to clean_dialogue
df['sentiment_scores'] = df['clean_dialogue'].apply(analyze_sentiment)

# Save
df.to_csv('data_with_sentiment.csv', index=False)
print("Sentiment analysis complete with enhanced VADER! Saved to 'data_with_sentiment.csv'.")
print(df[['clean_dialogue', 'sentiment_scores']].head(5))
