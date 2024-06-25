"""
PRE-PROCESSING
Generate embeddings and train_test_split data
for files downstream
"""

#============= Connect to dir and modules =================

# Update working directory
from google.colab import drive
import os

# # Mount drive
# drive.mount('/content/drive')

# # Change workdir
# os.chdir('./drive/MyDrive/266_project/project') #

from global_vars import GVD, FPATH
from project_utils import prepare_word2vec_sequences

#============= Import packages =================

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

# Sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, roc_auc_score, precision_recall_curve, classification_report

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
import pickle

# Gradients
import csv

#============= Global Variables =================

# Instantiate global variables
word2vec_model = FPATH + GVD['word2vec_model']
text_for_embeddings = FPATH + GVD['text_for_embeddings']
word2vec_embeddings = FPATH + GVD['word2vec_embeddings']
bert_embeddings = FPATH + GVD['bert_embeddings']

#============= Generate embeddings with W2V =================

# Import df
df = pd.read_csv(FPATH + GVD['text_for_embeddings'], encoding='utf-8')#, skiprows=[34331,67469,98406], error_bad_lines=False)

# Print pivot table
pd.pivot_table(df, index=['source'], columns='label', values=['text'], aggfunc='count')

# Instantiate tweets
tweets = df[df['source'].isin(['t3','t5'])].copy()

#Save
if not os.path.exists(FPATH + GVD['word2vec_model']):
  # Get padded sequences and word2vec model
  word2vec_embeddings, word2vec_model = prepare_word2vec_sequences(df)
  
  # Save
  word2vec_model.save(FPATH + GVD["word2vec_model"])
  torch.save(word2vec_embeddings, FPATH + GVD['word2vec_embeddings'])

  print(f"vocab_size: {vocab_size}")
  print(f"word2vec_embeddings.shape: {word2vec_embeddings.shape}")

# ============== Generate Bert embeddings ===============

# Instantiate the model and generate bert embeddings
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_embeddings = []
for text in tweets['text']:
    tokens = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length = 512,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )
    bert_embeddings.append(tokens)

# Pickle Bert embeddings
with open(FPATH + GVD['bert_embeddings'], 'wb') as f:
    pickle.dump(bert_embeddings, f)

# ============= Sample for train, val and test ============

# Import data sets
train_df = pd.read_csv(FPATH + GVD['train_df'])
val_df = pd.read_csv(FPATH + GVD['val_df'])
test_df = pd.read_csv(FPATH + GVD['test_df'])

# Set sample size
train_nsample, val_nsample, test_nsample = 2500, 600, 1500

# Get samples
train_df_sample = pd.concat([train_df[train_df['label']==0].sample(n=train_nsample, random_state=42),train_df[train_df['label']==1].sample(n=train_nsample, random_state=42)])
val_df_sample = pd.concat([val_df[val_df['label']==0].sample(n=val_nsample, random_state=42),val_df[val_df['label']==1].sample(n=val_nsample, random_state=42)])
test_df_sample = pd.concat([test_df[test_df['label']==0].sample(n=test_nsample, random_state=42),test_df[test_df['label']==1].sample(n=test_nsample, random_state=42)])

# Reset resample
train_df_sample.reset_index(drop=True, inplace=True)
val_df_sample.reset_index(drop=True, inplace=True)
test_df_sample.reset_index(drop=True, inplace=True)

# Save
train_df_sample.to_csv(FPATH + GVD['train_df_sample'])
val_df_sample.to_csv(FPATH + GVD['val_df_sample'])
test_df_sample.to_csv(FPATH + GVD['test_df_sample'])

#============= Fin =================

print(train_df_sample['label'].value_counts(normalize=True) * 100)
print(train_df_sample['label'].value_counts())
print(val_df_sample['label'].value_counts(normalize=True) * 100)
print(val_df_sample['label'].value_counts())
print(test_df_sample['label'].value_counts(normalize=True) * 100)
print(test_df_sample['label'].value_counts())

print(train_df_sample.shape)
print(val_df_sample.shape)
print(test_df_sample.shape)