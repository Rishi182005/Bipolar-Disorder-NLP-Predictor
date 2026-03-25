# Bipolar Disorder Prediction using NLP Emotion Trajectories

## Overview

This project develops an NLP pipeline to analyze doctor-patient dialogue data and predict mood shifts in bipolar disorder using emotion trajectory modeling.

## Key Idea

Track emotional evolution across multiple sessions to identify potential mood swings earlier than traditional static analysis.

## Pipeline

Text Input → NLP Preprocessing → Emotion Extraction (DistilBERT) → Feature Engineering → Temporal Modeling → Prediction

## Technologies

- Python
- DistilBERT / BioBERT
- HuggingFace Transformers
- Scikit-learn
- Deep Learning models

## Dataset

- Synthetic dataset (200–300 simulated patients)
- Multi-session dialogue sequences

## Engineering Highlights

- Sequential emotion trajectory modeling
- Transformer-based feature extraction
- Healthcare-focused NLP pipeline

## Metrics

- F1 Score evaluation
- MAE for prediction performance

