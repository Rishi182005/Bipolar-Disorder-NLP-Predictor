from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('final_nlp_features_complete.csv')
df['symptom_list'] = df['symptom_list'].apply(eval)  # Convert string to list

# Prepare labels (simplified: 1 if symptoms, 0 if empty)
df['has_symptoms'] = df['symptom_list'].apply(lambda x: 1 if x else 0)

# Tokenize and align
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
def tokenize_data(texts, labels):
    encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=128, return_tensors="pt")
    return encodings, np.array(labels)

texts = df['clean_dialogue']
labels = df['has_symptoms']
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)
train_encodings, train_labels = tokenize_data(train_textso, train_labels)
val_encodings, val_labels = tokenize_data(val_texts, val_labels)

# Model
model = AutoModelForTokenClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=2)

# Training
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_encodings,
    eval_dataset=val_encodings
)
trainer.train()

# Save model
model.save_pretrained("clinicalbert_symptom_model")
tokenizer.save_pretrained("clinicalbert_symptom_model")
