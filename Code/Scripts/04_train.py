"""
TRAIN MODEL
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
from project_utils import CustomDataset, BERT, BERTweet, ALBERT, CNNForWord2VecBERT, CNNForWord2VecBERTFT, CNNForWord2VecBERTweet, CNNForWord2VecBERTweetFT, CNNForWord2VecALBERT, CNNForWord2VecALBERTFT


project_dict = [
  {'MODEL':BERT, 'NAME':'CNNForBERT', 'W2V': False, 'FINETUNING':False, 'BASELINE': True, 'CONFIG':'bert-base-uncased'},
  {'MODEL':CNNForWord2VecBERT, 'NAME':'CNNForWord2VecBERT', 'W2V': True, 'FINETUNING':False, 'BASELINE': False, 'CONFIG':'bert-base-uncased'},
  {'MODEL':CNNForWord2VecBERTFT, 'NAME':'CNNForWord2VecBERTFT', 'W2V': True, 'FINETUNING':True, 'BASELINE': False, 'CONFIG':'bert-base-uncased'},
  {'MODEL':BERTweet, 'NAME':'CNNForBERTweet', 'W2V': False, 'FINETUNING':False, 'BASELINE': True, 'CONFIG':'vinai/bertweet-base'},
  {'MODEL':CNNForWord2VecBERTweet, 'NAME':'CNNForWord2VecBERTweet', 'W2V': True, 'FINETUNING':False, 'BASELINE': False, 'CONFIG':'vinai/bertweet-base'},
  {'MODEL':CNNForWord2VecBERTweetFT, 'NAME':'CNNForWord2VecBERTweetFT', 'W2V': True, 'FINETUNING':True, 'BASELINE': False, 'CONFIG':'vinai/bertweet-base'},
  {'MODEL':ALBERT, 'NAME':'CNNForALBERT', 'W2V': False, 'FINETUNING':False, 'BASELINE': True, 'CONFIG':'albert/albert-base-v2'},
  {'MODEL':CNNForWord2VecALBERT, 'NAME':'CNNForWord2VecALBERT', 'W2V': True, 'FINETUNING':False, 'BASELINE': False, 'CONFIG':'albert/albert-base-v2'},
  {'MODEL':CNNForWord2VecALBERTFT, 'NAME':'CNNForWord2VecALBERTFT', 'W2V': True, 'FINETUNING':True, 'BASELINE': False, 'CONFIG':'albert/albert-base-v2'}
]

def train_model(**kwargs):
  """ 
  1/4 Load variables
  """
  # Instantiate model and model conditions
  NAME = kwargs['NAME']
  MODEL = kwargs['MODEL']
  W2V = kwargs['W2V']
  FINETUNING = kwargs['FINETUNING']
  BASELINE = kwargs['BASELINE']
  CONFIG = kwargs['CONFIG']

  # Instantiate file names
  FNAME_OPTUNA_STUDY = f'optuna_study_{NAME}.pkl'

  # Instantiate study object to get best_hyperparams
  with open(FPATH + FNAME_OPTUNA_STUDY, 'rb') as f:
      study = pickle.load(f)
  best_hyperparams = study.best_params

  # Instantiate dataframes for train and validation
  train_df = pd.read_csv(FPATH + 'train_df.csv')
  val_df = pd.read_csv(FPATH + 'val_df.csv')

  # Instantiate the CNN hyperparameters
  filter_sizes = [3, 4, 5]
  num_filters = 100
  embedding_dim = 768

  # Set the hyperparameters
  batch_size = best_hyperparams['batch_size']
  learning_rate = best_hyperparams['learning_rate']
  epochs = best_hyperparams['epochs']
  dropout_rate = best_hyperparams['dropout_rate']
  optimizer = best_hyperparams['optimizer']
  weight_decay = best_hyperparams['weight_decay']
  max_len = best_hyperparams['max_len']
  hidden_dim = best_hyperparams.get('hidden_dim', None)

  # Instantiate the dataset with the BERT tokenizer and embeddings
  bert_model = AutoModel.from_pretrained(CONFIG)
  tokenizer = AutoTokenizer.from_pretrained(CONFIG)

  # Ensure bert_model is in eval mode and move to GPU if available
  bert_model.eval()
  if torch.cuda.is_available():
      bert_model = bert_model.to('cuda')

  # Instantiate w2v model
  if W2V:
    WORD2VEC_MODEL = Word2Vec.load(FPATH + 'word2vec_model.model')
  else:
    WORD2VEC_MODEL = None
  
  # Instantiate training and validation sets
  training_set = CustomDataset(train_df, tokenizer, max_len, WORD2VEC_MODEL)
  val_set = CustomDataset(val_df, tokenizer, max_len, WORD2VEC_MODEL)

  # Create the dataloaders
  training_loader = DataLoader(training_set, batch_size, shuffle=True)
  val_loader = DataLoader(val_set, batch_size, shuffle=False)

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

  # Move the model to the GPU
  if torch.cuda.is_available():
      model = model.to('cuda')

  # Create the optimizer
  if optimizer == 'AdamW':
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
  elif optimizer == 'Adam':
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
  elif optimizer == 'SGD':
    optimizer = SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
  else:
    raise ValueError("Invalid optimizer")

  # Create the loss function
  loss_function = nn.BCEWithLogitsLoss()

  """
  2/4 Train
  """
  # Expand number of steps back before taking average loss
  accumulation_steps = 3
  print(f'{NAME} Start!')
  # Training loop with metrics calculation
  for epoch in range(epochs):
      model.train()
      train_targets = []
      train_outputs = []

      # Training phase
      total_train_iterations = len(training_loader)
      total_loss = 0
      for i, data in tqdm(enumerate(training_loader,0),total=total_train_iterations, desc="Training"):
          
          input_ids = data['input_ids'].to(bert_model.device)
          attention_mask = data['attention_mask'].to(bert_model.device)
          targets = data['targets'].to(bert_model.device)

          # Forward pass
          if W2V:
            word_indices = data['word_indices'].to(bert_model.device)
            outputs = model(input_ids, attention_mask, word_indices)
          else:
            outputs = model(input_ids, attention_mask)
          
          # Get loss
          loss = loss_function(outputs, targets.unsqueeze(1))
          loss.backward()
          if (i + 1) % accumulation_steps == 0:  
              optimizer.step()  
              optimizer.zero_grad()

          # Store targets and outputs for evaluation
          train_targets.extend(targets.cpu().detach().numpy().tolist())
          train_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

      print(f"Epoch {epoch+1}/{epochs}, Training Loss: {total_loss/total_train_iterations}")
      
  print()
  # Assuming `model` is your trained model
  torch.save(model.state_dict(), FPATH + f'{NAME}.pth')

  print(f'{NAME} Done!')

# Run function
for project in project_dict:
  train_model(**project)
