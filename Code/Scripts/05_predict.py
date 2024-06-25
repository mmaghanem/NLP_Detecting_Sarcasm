"""
PREDICT WITH FINE-TUNED MODEL
with Optuna best hyperparameters and loaded embeddings from w2v and BERT
"""

# Import packages

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import AdamW, Adam, SGD
from torch.nn.utils.rnn import pad_sequence

# Gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Bert
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader

# Admin
import os
import time
from tqdm.auto import tqdm
import re

# Data
import pandas as pd
import numpy as np
import random

# Gradients
import csv
import pickle

# Optuna
import optuna
from optuna.pruners import MedianPruner

# Custom modules
from global_vars import GVD, FPATH
from project_utils import CustomDataset,  determine_category, BERT, BERTweet, ALBERT, CNNForWord2VecBERT, CNNForWord2VecBERTFT, CNNForWord2VecBERTweet, CNNForWord2VecBERTweetFT, CNNForWord2VecALBERT, CNNForWord2VecALBERTFT


project_dict = [
  {'MODEL': BERT, 'NAME':'CNNForBERT', 'W2V': False, 'FINETUNING':False, 'BASELINE': True, 'CONFIG':'bert-base-uncased'},
  {'MODEL': CNNForWord2VecBERT, 'NAME':'CNNForWord2VecBERT', 'W2V': True, 'FINETUNING':False, 'BASELINE': False, 'CONFIG':'bert-base-uncased'},
  {'MODEL': CNNForWord2VecBERTFT, 'NAME':'CNNForWord2VecBERTFT', 'W2V': True, 'FINETUNING':True, 'BASELINE': False, 'CONFIG':'bert-base-uncased'},
  {'MODEL': BERTweet, 'NAME':'CNNForBERTweet', 'W2V': False, 'FINETUNING':False, 'BASELINE': True, 'CONFIG':'vinai/bertweet-base'},
  {'MODEL': CNNForWord2VecBERTweet, 'NAME':'CNNForWord2VecBERTweet', 'W2V': True, 'FINETUNING':False, 'BASELINE': False, 'CONFIG':'vinai/bertweet-base'},
  {'MODEL': CNNForWord2VecBERTweetFT, 'NAME':'CNNForWord2VecBERTweetFT', 'W2V': True, 'FINETUNING':True, 'BASELINE': False, 'CONFIG':'vinai/bertweet-base'},
  {'MODEL': ALBERT, 'NAME':'CNNForALBERT', 'W2V': False, 'FINETUNING':False, 'BASELINE': True, 'CONFIG':'albert/albert-base-v2'},
  {'MODEL': CNNForWord2VecALBERT, 'NAME':'CNNForWord2VecALBERT', 'W2V': True, 'FINETUNING':False, 'BASELINE': False, 'CONFIG':'albert/albert-base-v2'},
  {'MODEL': CNNForWord2VecALBERTFT, 'NAME':'CNNForWord2VecALBERTFT', 'W2V': True, 'FINETUNING':True, 'BASELINE': False, 'CONFIG':'albert/albert-base-v2'}
]


def predict(**kwargs):

  # Set the kwargs
  CONFIG = kwargs['CONFIG']
  MODEL = kwargs['MODEL']
  NAME = kwargs['NAME']
  FNAME = NAME + '.pth'
  FNAME_RESULTS = NAME + '.pkl'
  W2V = kwargs['W2V']
  FINETUNING = kwargs['FINETUNING']
  FNAME_OPTUNA_STUDY = f'optuna_study_{NAME}.pkl'

  # Instantiate study object to get best_hyperparams  
  with open(FPATH + FNAME_OPTUNA_STUDY, 'rb') as f:
      study = pickle.load(f)
  best_hyperparams = study.best_params

  # Set the hyperparameters safely using .get()
  batch_size = best_hyperparams.get('batch_size')
  max_len = best_hyperparams.get('max_len')
  dropout_rate = best_hyperparams.get('dropout_rate')
  hidden_dim = best_hyperparams.get('hidden_dim')


  # Instantiate the CNN hyperparameters
  filter_sizes = [3, 4, 5]
  num_filters = 100
  embedding_dim = 768

  # Instantiate w2v model
  if W2V:
    WORD2VEC_MODEL = Word2Vec.load(FPATH + 'word2vec_model.model')
  else:
    WORD2VEC_MODEL = None

  # Instantiate model
  if W2V:
    vocab_size = len(WORD2VEC_MODEL.wv.index_to_key)
    word2vec_weights = WORD2VEC_MODEL.wv.vectors
    if FINETUNING:
      hidden_dim = hidden_dim
      freeze = True
      model = MODEL(embedding_dim=embedding_dim, 
                    num_filters=num_filters,
                    filter_sizes=filter_sizes,
                    dropout_rate=dropout_rate,
                    hidden_dim=hidden_dim,
                    vocab_size=vocab_size,
                    word2vec_weights=word2vec_weights,
                    freeze=freeze)
    else:
      hidden_dim = None
      freeze = None
      model = MODEL(embedding_dim=embedding_dim, 
                    num_filters=num_filters,
                    filter_sizes=filter_sizes,
                    dropout_rate=dropout_rate,
                    vocab_size=vocab_size,
                    word2vec_weights=word2vec_weights)
  else:
    model = MODEL()
  
  # Load model
  model.load_state_dict(torch.load(FPATH + FNAME))

  # Move the model to the GPU
  if torch.cuda.is_available():
      model = model.to('cuda')

  # Instantiate tokenizer
  tokenizer = AutoTokenizer.from_pretrained(CONFIG)

  # Instantiate test set
  test_df = pd.read_csv(FPATH + 'test_df.csv')

  # Create custom dataset and dataloader
  test_set = CustomDataset(test_df, tokenizer, max_len, word2vec_model=WORD2VEC_MODEL)
  test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

  # Create the loss function
  loss_function = nn.BCEWithLogitsLoss()

  # Define threshold
  threshold = 0.5

  # for epoch in range(epochs):
  model.eval()
  true_labels, predictions, probabilities, embeddings = [], [], [], []

  # Prediction loop
  total_test_iterations = len(test_loader)
  total_loss = 0
  with torch.no_grad():
    for i, data in tqdm(enumerate(test_loader,0),total=total_test_iterations, desc="Pred"):

      input_ids = data['input_ids'].to('cuda')
      attention_mask = data['attention_mask'].to('cuda')
      targets = data['targets'].to('cuda')
      
      if W2V:
          word_indices = data['word_indices'].to('cuda')
          outputs, x_combined = model(input_ids, attention_mask, word_indices, prediction = True)

      else:
          outputs, x_combined = model(input_ids, attention_mask, prediction = True)
      
      # Calculate loss
      loss = loss_function(outputs, targets.unsqueeze(1))
      
      # Get results
      y_prob = torch.sigmoid(outputs).cpu().detach().numpy()
      y_pred = (y_prob > threshold).astype(int)
      targets = targets.cpu().detach().numpy()
      total_loss += loss

      # Extend lists
      true_labels.extend(targets.tolist())
      predictions.extend(y_pred.tolist())
      probabilities.extend(y_prob.tolist())
      embeddings.extend(x_combined.cpu().numpy())

  # Update dataframe
  test_df['label'] = true_labels
  test_df['pred'] = predictions
  test_df['probs'] = probabilities
  test_df['embeddings'] = embeddings

  # Get value from lists
  test_df['pred'] = test_df['pred'].apply(lambda x: x[0])
  test_df['probs'] = test_df['probs'].apply(lambda x: x[0])

  # Get results
  test_df['category'] = test_df.apply(determine_category, axis=1)
  
  # Pickle results
  test_df.to_pickle(FPATH + NAME + '_test_df.pkl')

  print(f'Done pickling results for {NAME}')

  return test_df

# Run function
for project in project_dict:
  predict(**project)


