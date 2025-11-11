import pandas as pd

# Load both datasets
df_emotions = pd.read_csv('data_with_emotions.csv')
df_sentiment = pd.read_csv('data_with_sentiment.csv')

# Merge with a left join to keep all rows from data_with_emotions and add sentiment_scores
df_final = df_emotions.merge(df_sentiment[['clean_dialogue', 'sentiment_scores']], 
                             on='clean_dialogue', 
                             how='left', 
                             suffixes=('_emotions', '_sentiment'))

# Fill any missing sentiment_scores with a default dictionary
df_final['sentiment_scores'] = df_final['sentiment_scores'].fillna({'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0})

# Ensure emotion_dict is present (should already be from df_emotions)
if 'emotion_dict' not in df_final.columns:
    print("Warning: emotion_dict not found. Using data from data_with_emotions.csv.")
    df_final['emotion_dict'] = df_emotions['emotion_dict']

# Save the consolidated dataset
df_final.to_csv('final_nlp_features.csv', index=False)
print("Features consolidated! Saved to 'final_nlp_features.csv'.")
print(df_final[['clean_dialogue', 'emotion_dict', 'sentiment_scores']].head(5))  # Preview
