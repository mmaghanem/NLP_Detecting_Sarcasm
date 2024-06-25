# NLP: Detecting Sarcasm in Text Data from Social Media

"Detecting Sarcasm in Text Data from Social Media." The project Developed advanced NLP models using BERT and concatenated Word2Vec embeddings to detect sarcasm in social media posts, optimizing with Optuna for enhanced performance. Demonstrated proficiency in contextual language understanding, semantic analysis, and hyperparameter tuning for accurate sarcasm classification.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Preprocessing](#data-preprocessing)
3. [Modeling and Training](#modeling-and-training)
4. [Evaluation and Analysis](#evaluation-and-analysis)
5. [Skills and Technologies Applied](#skills-and-technologies-applied)
6. [Benefits of the Project](#benefits-of-the-project)

## Introduction

The project aims to classify tweets as sarcastic or not using state-of-the-art NLP models, including BERT, BERTweet, and ALBERT, enhanced with Word2Vec embeddings. The goal is to improve the precision and recall in sarcasm detection, providing valuable insights for applications in social media analysis and customer feedback interpretation.

## Data Preprocessing

### Methods and Techniques

- **Data Cleaning**: Remove URLs, user mentions, and excessive whitespace.
- **Tokenization**: Use BERT and Word2Vec tokenizers to prepare the text for model input.
- **Padding and Truncation**: Ensure uniform input length for batch processing.

```python
import pandas as pd
import re
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
import torch

# Load data
data = pd.read_csv('train_df.csv')

# Clean data
def clean_tweet(tweet):
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    return tweet

data['cleaned_text'] = data['text'].apply(clean_tweet)

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = data['cleaned_text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

# Padding
tokens_padded = pad_sequence([torch.tensor(t) for t in tokens], batch_first=True, padding_value=tokenizer.pad_token_id)
```

## Modeling and Training
### Baseline Models
   - **BERT:** A transformer model that captures contextual relationships in text.
   - **BERTweet:** A BERT model pre-trained on English tweets.
   - **ALBERT:** A lighter version of BERT optimized for faster training.
```python
from transformers import BertModel, BertConfig

# Load pre-trained BERT model
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Example forward pass
inputs = torch.tensor(tokens_padded[:5])
outputs = bert_model(inputs)
pooled_output = outputs[1]  # Pooled output for classification
```

### Custom Models
   - **Baseline + Word2Vec:** Train custom Word2Vec embeddings and concatenate with BERT embeddings for enhanced feature representation.
```python
from gensim.models import Word2Vec

# Train Word2Vec model
sentences = [tweet.split() for tweet in data['cleaned_text']]
w2v_model = Word2Vec(sentences, vector_size=300, window=5, min_count=1, workers=4)

# Get Word2Vec embeddings
def get_w2v_embeddings(tokens, w2v_model):
    return [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]

data['w2v_embeddings'] = data['cleaned_text'].apply(lambda x: get_w2v_embeddings(x.split(), w2v_model))
```

### Fine-Tuning and Hyperparameter Optimization
   - **Optuna:** Used for hyperparameter tuning to optimize model performance.
```python
import optuna
from optuna.trial import TrialState
from transformers import AdamW

def objective(trial):
    # Define model, optimizer, and other hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    model = BertModel.from_pretrained('bert-base-uncased')
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Training loop (simplified)
    for epoch in range(3):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch['input_ids'])
            loss = criterion(outputs, batch['labels'])
            loss.backward()
            optimizer.step()
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

## Evaluation and Analysis
### Metrics
   - **Accuracy, Precision, Recall, F1-Score:** Evaluate model performance on test data.
   - **Confusion Matrix:** Visualize true positives, false positives, true negatives, and false negatives.
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Predictions and evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
```

### Embedding Analysis
   - **PCA and Cosine Similarity:** Analyze embeddings to understand the feature space and similarity between classes.
```python
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objs as go

# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(embeddings)

# Cosine similarity
cos_sim = cosine_similarity(embeddings)

# Visualization
fig = go.Figure(data=go.Scatter(x=pca_result[:, 0], y=pca_result[:, 1], mode='markers'))
fig.show()
```

## Skills and Technologies Applied
### Python Programming
   - **Data Processing:** Used pandas for data manipulation and re for text cleaning.
   - **NLP:** Implemented tokenization, embedding, and text classification using transformers and gensim.
   - **Deep Learning:** Built and trained models using torch and torch.nn.
### Model Optimization
   - **Hyperparameter Tuning:** Applied optuna for optimizing model hyperparameters to enhance performance.
   - **Model Evaluation:** Used scikit-learn metrics for evaluating model accuracy, precision, recall, and F1-score.

## Benefits of the Project
1. **Enhanced Sarcasm Detection:**
   - Improved detection accuracy by integrating contextual and word-level embeddings.
2. **Application of Advanced NLP Techniques:**
   - Demonstrated the use of state-of-the-art models and custom embeddings to solve complex NLP tasks.
3. **Comprehensive Model Evaluation:**
   - Provided detailed analysis and visualization of model performance and embeddings.
