"""
MODEL HYPERPARAMETER OPTIMIZATION
with Optuna and loaded embeddings from w2v
and BERT
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

# Calculate metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, roc_auc_score, precision_recall_curve, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve

# Custom modules
from global_vars import GVD, FPATH
from project_utils import CustomDataset, BERT, BERTweet, ALBERT, CNNForWord2VecBERT, CNNForWord2VecBERTFT, CNNForWord2VecBERTweet, CNNForWord2VecBERTweetFT, CNNForWord2VecALBERT, CNNForWord2VecALBERTFT



""" 
1/4 Load embeddings
"""

# Load w2v embeddings and bert embedding
word2vec_embeddings = torch.load(FPATH + 'word2vec_embeddings.pt')
# bert_embeddings = FPATH + GVD['bert_embeddings']


"""
2/4 Instantiate objective function
"""
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

# Function to calculate metrics
def calculate_metrics(targets, outputs):
    accuracy = accuracy_score(targets, outputs)
    precision = precision_score(targets, outputs)
    recall = recall_score(targets, outputs)
    f1 = f1_score(targets, outputs)
    precision_vals, recall_vals, _ = precision_recall_curve(targets, outputs)
    pr_auc = auc(recall_vals, precision_vals)
    roc_auc = roc_auc_score(targets, outputs)
    return accuracy, precision, recall, f1, pr_auc, roc_auc

def objective(**kwargs):

  # Instantiate model and model conditions
  NAME = kwargs['NAME']
  MODEL = kwargs['MODEL']
  W2V = kwargs['W2V']
  FINETUNING = kwargs['FINETUNING']
  BASELINE = kwargs['BASELINE']
  CONFIG = kwargs['CONFIG']

  parameters = {
      'batch_size': trial.suggest_int('batch_size', 2, 4),
      'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
      'epochs': trial.suggest_int('epochs', 3, 5),
      'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
      'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5, log=True),
      'optimizer': trial.suggest_categorical('optimizer', ['AdamW', 'Adam','SGD']),
      'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
      'max_len': trial.suggest_int('max_len', 50, 100),
      'unfreeze_epoch': trial.suggest_int('unfreeze_epoch', 0, 5),
      'freeze_bert': trial.suggest_categorical('freeze_bert', [True, False]),
      'hidden_dim': trial.suggest_int('hidden_dim', 64, 256)
    }

  # Set the parameters
  batch_size = parameters['batch_size']
  learning_rate = parameters['learning_rate']
  epochs = parameters['epochs']
  dropout_rate = parameters['dropout_rate']
  optimizer = parameters['optimizer']
  weight_decay = parameters['weight_decay']
  max_len = parameters['max_len']
  unfreeze_epoch = parameters['unfreeze_epoch']
  freeze_bert = parameters['freeze_bert']
  hidden_dim = parameters['hidden_dim']


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

  # Define the parameters
  train_params = {'batch_size': batch_size,'shuffle': True}
  filter_sizes = [3, 4, 5] # We should add it to the Parameters
  num_filters = 100  # We should add it to the Parameters
  embedding_dim = 768
  vocab_size = len(WORD2VEC_MODEL.wv.index_to_key)

  # Instantiate dataframes for train and validation
  train_df = pd.read_csv(FPATH + 'train_df.csv')
  val_df = pd.read_csv(FPATH + 'val_df.csv')

  # Instantiate the CNN hyperparameters
  filter_sizes = [3, 4, 5]
  num_filters = 100
  embedding_dim = 768

  # Pass train and test to dataloader
  if W2V:
    training_set = CustomDataset(train_df, tokenizer, WORD2VEC_MODEL, max_len) # Move the Data rows outside the DataCustom
    val_set = CustomDataset(val_df, tokenizer, WORD2VEC_MODEL, max_len)  # Move the Data rows outside the DataCustom
  else:
    training_set = CustomDataset(train_df, tokenizer, max_len)
    val_set = CustomDataset(val_df, tokenizer, max_len)
  # Create the dataloaders
  training_loader = DataLoader(training_set, **train_params)
  val_loader = DataLoader(val_set, **train_params)

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

  # Instantiate pruner
  pruner = MedianPruner()

  # Initialize lists to store metrics
  metrics = {
      'train': {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'pr_auc': [], 'roc_auc': []},
      'val': {'loss':[], 'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'pr_auc': [], 'roc_auc': []}
  }

  # Define threshold
  threshold = 0.5

  # Expand number of steps back before taking average loss
  accumulation_steps = 3

  # Training loop with metrics calculation
  for epoch in range(epochs):
      model.train()
      train_targets = []
      train_outputs = []

      # Unfreeze the BERT model -> unfreeze_epoch is the epoch to unfreeze BERT
      if epoch == unfreeze_epoch:
          for param in model.bert_embeddings.parameters():
              param.requires_grad = True
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

          train_targets.extend(targets.cpu().detach().numpy().tolist())
          train_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

      # Calculate and store training metrics
      train_outputs_bin = np.array(train_outputs) >= threshold
      train_acc, train_prec, train_rec, train_f1, train_pr_auc, train_roc_auc = calculate_metrics(np.array(train_targets), train_outputs_bin)
      total_loss += loss.item()
      train_loss = total_loss / len(training_loader)
      metrics['train']['loss'].append(round(train_loss,4))
      metrics['train']['accuracy'].append(round(train_acc,4))
      metrics['train']['precision'].append(round(train_prec,4))
      metrics['train']['recall'].append(round(train_rec,4))
      metrics['train']['f1'].append(round(train_f1,4))
      metrics['train']['pr_auc'].append(round(train_pr_auc,4))
      metrics['train']['roc_auc'].append(round(train_roc_auc,4))

     # Validation phase
      model.eval()
      val_targets = []
      val_outputs = []
      val_loss_accumulated = 0.0  # To accumulate loss over all validation batches

      with torch.no_grad():
          total_val_iterations = len(val_loader)
          for data in tqdm(val_loader, total=total_val_iterations, desc="Validation"):
              word_indices = data['word_indices'].to('cuda', dtype=torch.long)
              bert_inputs = {key: value.to('cuda') for key, value in data['bert_inputs'].items()}
              targets = data['targets'].to('cuda', dtype=torch.float)

              # Forward pass
              outputs = model(word_indices, bert_inputs)  # Assuming model outputs logits # , w2vec_embeddings, bert_embedings
              loss = loss_function(outputs, targets.unsqueeze(1))
              val_loss_accumulated += loss.item()

              outputs = torch.sigmoid(outputs).squeeze()  # Apply sigmoid once to get probabilities
              val_targets.extend(targets.cpu().detach().numpy())
              # Assuming outputs could be a scalar or an array, ensure it's always treated as an iterable
              outputs_np = outputs.cpu().detach().numpy()  # Convert to numpy array

              # If outputs_np is a scalar (0-d array), convert it into a 1-d array with a single value
              if outputs_np.ndim == 0:
                  outputs_np = np.expand_dims(outputs_np, axis=0)

              val_outputs.extend(outputs_np)
              # val_outputs.extend(outputs.cpu().detach().numpy())

      # Calculate average validation loss
      val_loss = val_loss_accumulated / total_val_iterations

      # Convert outputs to binary predictions based on the threshold
      val_outputs_bin = np.array(val_outputs) >= threshold

      # Now calculate and print metrics using val_targets and val_outputs_bin
      val_acc, val_prec, val_rec, val_f1, val_pr_auc, val_roc_auc = calculate_metrics(np.array(val_targets), val_outputs_bin)
      metrics['val']['loss'].append(round(val_loss,4))
      metrics['val']['accuracy'].append(round(val_acc,4))
      metrics['val']['precision'].append(round(val_prec,4))
      metrics['val']['recall'].append(round(val_rec,4))
      metrics['val']['f1'].append(round(val_f1,4))
      metrics['val']['pr_auc'].append(round(val_pr_auc,4))
      metrics['val']['roc_auc'].append(round(val_roc_auc,4))

      print(f"Epoch {epoch+1}/{epochs} - Train Metrics: Loss: {train_loss}, Accuracy: {train_acc}, Precision: {train_prec}, Recall: {train_rec}, F1: {train_f1}, PR AUC: {train_pr_auc}, ROC AUC: {train_roc_auc}")
      print(f"Epoch {epoch+1}/{epochs} - Val Metrics: Loss: {val_loss},  Accuracy: {val_acc}, Precision: {val_prec}, Recall: {val_rec}, F1: {val_f1}, PR AUC: {val_pr_auc}, ROC AUC: {val_roc_auc}")
      trial.report(val_f1, epoch)
      if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

      # At the end of your objective function, before returning the optimization metric
      trial.set_user_attr("train_loss", train_loss)
      trial.set_user_attr("train_accuracy", train_acc)
      trial.set_user_attr("train_precision", train_prec)
      trial.set_user_attr("train_recall", train_rec)
      trial.set_user_attr("train_f1", train_f1)
      trial.set_user_attr("train_pr_auc", train_pr_auc)
      trial.set_user_attr("train_roc_auc", train_roc_auc)

      trial.set_user_attr("val_loss", val_loss)
      trial.set_user_attr("val_accuracy", val_acc)
      trial.set_user_attr("val_precision", val_prec)
      trial.set_user_attr("val_recall", val_rec)
      trial.set_user_attr("val_f1", val_f1)
      trial.set_user_attr("val_pr_auc", val_pr_auc)
      trial.set_user_attr("val_roc_auc", val_roc_auc)

  return np.max(metrics['val']['f1'])

"""
3/4 Run Optuna study
"""

# Empty cash
torch.cuda.empty_cache()

# Run trials
study = optuna.create_study(direction='maximize', pruner=MedianPruner())
study.optimize(objective, n_trials=10)

"""
4/4 Return and pickle results
"""

# Get the best hyperparameters
best_params = study.best_params
print(best_params)

# Pickle the study
with open(GVD['optuna_study_w2v_bert'], 'wb') as f:
    pickle.dump(study, f)

print('Done!')

# Run function
for project in project_dict:
  objective(**project)
