# data_gen.py (IMPROVED)
import pandas as pd
from transformers import pipeline
import random

generator = pipeline('text-generation', model='gpt2', truncation=True, pad_token_id=50256)

moods = ['manic', 'depressive', 'stable']
mood_prompts = {
    'manic': ["I'm full of energy, ideas racing, haven't slept in days. ",
              "Everything feels electric! I'm going to start 10 projects today. ",
              "I feel invincible, like I could conquer the world!"],
    'depressive': ["I can't get out of bed. Everything feels heavy and pointless. ",
                   "No energy, no hope. I just want to disappear. ",
                   "Tears come for no reason. The world is gray."],
    'stable': ["I'm doing okay today. Mood feels balanced. ",
               "Therapy is helping. I slept well and ate breakfast. ",
               "No extreme highs or lows. Just steady."]
}

def generate_dialogue(mood):
    prompt = random.choice(mood_prompts[mood])
    out = generator(prompt, max_length=100, num_return_sequences=1, temperature=0.9)
    return out[0]['generated_text'][len(prompt):].strip()

# Generate 50 patients, 6-10 sessions
data = []
for pid in range(50):
    sessions = random.randint(6, 10)
    mood_history = []
    for sess in range(1, sessions + 1):
        # Add mood persistence (realistic)
        mood = random.choices(moods, weights=[0.3, 0.3, 0.4], k=1)[0]
        if mood_history and random.random() < 0.7:
            mood = mood_history[-1]
        mood_history.append(mood)
        
        dialogue = generate_dialogue(mood)
        data.append({
            'patient_id': pid,
            'session': sess,
            'dialogue': dialogue,
            'mood': mood,
            'progress_score': round(random.uniform(0.3, 0.9), 2),
            'decision': random.choice(['Monitor', 'Therapy', 'Medication Review'])
        })

df = pd.DataFrame(data)
df.to_csv('synthetic_bd_data.csv', index=False)
print(f"Generated {len(df)} records → synthetic_bd_data.csv")
