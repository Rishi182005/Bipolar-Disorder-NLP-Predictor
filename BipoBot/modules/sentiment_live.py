from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
analyzer.lexicon.update({
    'mania': 2.5, 'depressive': -2.5, 'hopeless': -2.8, 'energetic': 2.0,
    'insomnia': -1.8, 'irritable': -2.0, 'euphoric': 2.2, 'dysphoric': -2.2,
    'panic': -2.5, 'excitement': 2.0, 'grief': -2.8, 'relief': 1.8
})
def analyze_sentiment(text: str) -> dict:
    if not text.strip():
        return {'neg':0.0,'neu':0.0,'pos':0.0,'compound':0.0}
    s = analyzer.polarity_scores(text)
    return {k: round(v,4) for k,v in s.items()}
