"""
PROJECT UTILS
Functions and Classes to be used across the project
"""


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
from transformers import AutoTokenizer, AutoModel, BertModel, get_linear_schedule_with_warmup
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

# Optuna

import optuna
from optuna.pruners import MedianPruner

# Visualizations
import matplotlib.pyplot as plt

# Calculate metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve


"""
GENERATE EMBEDDINGS
Functions to create w2v and bert embeddings at outset of project
"""


# Combined Cleaning and Preprocessing Function
def clean_and_preprocess_tweets(df):
    def clean_tweet(tweet):
        # Remove URLs
        tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
        # Remove user mentions
        tweet = re.sub(r'@\w+', '', tweet)
        # Remove excessive whitespace
        tweet = re.sub(r'\s+', ' ', tweet).strip()
        return tweet

    # Apply cleaning and simple preprocessing
    return df['text'].apply(lambda x: simple_preprocess(clean_tweet(x)))

# Streamlined Training and Sequence Preparation
def prepare_word2vec_sequences(df, max_len=None):
    # Step 1: Clean and preprocess tweets
    tweets_preprocessed = clean_and_preprocess_tweets(df)

    # Step 2: Train Word2Vec
    word2vec_model = Word2Vec(sentences=tweets_preprocessed, vector_size=768, window=5, min_count=1, workers=4)

    # Step 3: Create word to index mapping
    word_index = {word: i for i, word in enumerate(word2vec_model.wv.index_to_key)}

    # Step 4: Convert tweets to sequences of indices
    sequences = [[word_index.get(word, 0) for word in tweet] for tweet in tweets_preprocessed]

    # Convert sequences to PyTorch tensors before padding
    sequences_tensors = [torch.tensor(seq) for seq in sequences]

    # Step 5: Pad sequences
    max_len = max(len(seq) for seq in sequences_tensors)

    # Step 6: Make padded sequence
    padded_sequences = pad_sequence(sequences_tensors, batch_first=True, padding_value=0)

    # Dimensions
    word2vec_dim = 300
    bert_dim = 768

    # Linear transformation layer
    projection_layer = nn.Linear(in_features=word2vec_dim, out_features=bert_dim)
    torch.nn.init.xavier_uniform_(projection_layer.weight)

    # Example Word2Vec embeddings tensor ([batch_size, sequence_length, word2vec_dim])
    word2vec_embeddings = torch.randn(len(padded_sequences), max_len, word2vec_dim)

    # Project embeddings
    projected_embeddings = projection_layer(word2vec_embeddings)

    return projected_embeddings, word2vec_model


"""
PREPARING DATA FOR MODEL
"""

# Create a class for the dataset
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, word2vec_model=None):
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.label
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.word2vec_model = word2vec_model

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        words = simple_preprocess(text) 
        targets = self.targets[index]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        if self.word2vec_model:
          # Prepare Word2Vec embeddings
          word_indices = [self.word2vec_model.wv.key_to_index.get(word, 0) for word in words]
          
          # Ensure word_indices does not exceed max_len
          word_indices = word_indices[:self.max_len]
          # Pad word_indices to ensure it has length of max_len
          word_indices = np.pad(word_indices, (0, self.max_len - len(word_indices)), mode='constant', constant_values=0)
          word_indices = torch.tensor(word_indices, dtype=torch.long)

          return {
              'input_ids': inputs['input_ids'].flatten(),
              'attention_mask': inputs['attention_mask'].flatten(),
              'word_indices': word_indices,
              'targets': torch.tensor(targets, dtype=torch.float),
            }
        else:
          return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'targets': torch.tensor(targets, dtype=torch.float),
          }


"""
MODELS
"""

class Attention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, embeddings):
        # embeddings shape: [batch_size, max_seq_length, embedding_dim]
        attention_weights = self.attention(embeddings)  # Compute attention weights
        attention_weights = torch.softmax(attention_weights, dim=1)  # Softmax over max_seq_length dimension
        attended_embeddings = embeddings * attention_weights  # Apply weights
        attended_embeddings = torch.sum(attended_embeddings, dim=1)  # Sum over the sequence
        return attended_embeddings

class BERT(nn.Module):
    """Model_1.0"""
    def __init__(self):
        super(BERT, self).__init__()

        # BERT Embedding Layer
        self.bert = AutoModel.from_pretrained('bert-base-uncased')

        # Fully connected layer: Adjust according to your task
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, prediction = False):

        # Process BERT embeddings
        bert_embeddings = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Global Average Pooling
        x_bert = bert_embeddings.pooler_output

        # Apply fully connected layer
        x = self.fc(x_bert)

        if prediction:
            return x, x_bert
        else:
            return x

class CNNForWord2VecBERT(nn.Module):
    """Model_1.1"""
    def __init__(self, word2vec_weights, vocab_size, embedding_dim, num_filters, filter_sizes, dropout_rate):
        super(CNNForWord2VecBERT, self).__init__()

        # WORD2VEC Embedding layer
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(word2vec_weights))

        # Convolutional layers: Adjusted for embedding dimensions
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(k, embedding_dim), padding=(k - 1, 0)) for k in filter_sizes
        ])

        # Batch normalization layers: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d
        self.conv_bn = nn.ModuleList([
            nn.BatchNorm2d(num_filters) for _ in filter_sizes
        ])

        # Global Average Pooling layer for CNN features
        self.cnn_global_avg_pool = nn.AdaptiveAvgPool2d((1, num_filters))

        # BERT Embedding Layer
        self.bert = AutoModel.from_pretrained('bert-base-uncased')

        # global average pooling layer for BERT embeddings
        self.bert_global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layer: Adjust according to your task
        self.fc = nn.Linear(num_filters * len(filter_sizes) + embedding_dim, 1) # The "* 2" accounts for concatenation of avg and max pooling features

    def forward(self, input_ids, attention_mask, word_indices, prediction=False):
        # x shape: [batch_size, max_sequence_length, embedding_dim]

        # Word2Vec Embeddings

        # Convert ids to embeddings
        x = self.embedding(word_indices)

        # Add a channel dimension: [batch_size, 1, max_sequence_length, embedding_dim]
        x = x.unsqueeze(1)

        # Apply convolutions and ReLU
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]

        # Apply global average pooling
        x = [self.cnn_global_avg_pool(xi).squeeze(2) for xi in x]

        # Concatenate along the filter dimension
        x = torch.cat(x, 1)

        # Flatten
        x = x.view(x.size(0), -1)

        # Process BERT embeddings
        bert_embeddings = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x_bert = bert_embeddings.last_hidden_state


        # Apply mean pooling
        x_bert = x_bert.mean(dim=1)

        # Concatenate Word2Vec and BERT embeddings
        x_combined = torch.cat((x, x_bert), 1)

        # Apply dropout
        x_combined = self.dropout(x_combined)

        # Apply fully connected layer
        x = self.fc(x_combined)

        if prediction:
          return x, x_combined
        else:
          return x

class CNNForWord2VecBERTFT(nn.Module):
    """Model_1.2"""
    def __init__(self, word2vec_weights, vocab_size, embedding_dim, num_filters, filter_sizes, dropout_rate, hidden_dim, freeze=True):
        super(CNNForWord2VecBERTFT, self).__init__()

        # WORD2VEC Embedding layer
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(word2vec_weights))

        # Convolutional layers: Adjusted for embedding dimensions
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(k, embedding_dim), padding=(k - 1, 0)) for k in filter_sizes
        ])

        # Batch normalization layers: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d
        self.conv_bn = nn.ModuleList([
            nn.BatchNorm2d(num_filters) for _ in filter_sizes
        ])

        # Global Average Pooling layer for CNN features
        self.cnn_global_avg_pool = nn.AdaptiveAvgPool2d((1, num_filters))

        # BERT Embedding Layer
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.bert_attention = Attention(embedding_dim, hidden_dim)  # Initialize attention for BERT embeddings

        # Freeze BERT layer
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

        # global average pooling layer for BERT embeddings
        self.bert_global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layer: Adjust according to your task
        self.fc = nn.Linear(num_filters * len(filter_sizes) + embedding_dim, 1) # The "* 2" accounts for concatenation of avg and max pooling features

    def forward(self, input_ids, attention_mask, word_indices, prediction=False):
        # x shape: [batch_size, max_sequence_length, embedding_dim]

        # Word2Vec Embeddings

        # Convert ids to embeddings
        x = self.embedding(word_indices)

        # Add a channel dimension: [batch_size, 1, max_sequence_length, embedding_dim]
        x = x.unsqueeze(1)

        # Apply convolutions and ReLU
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]

        # Apply global average pooling
        x = [self.cnn_global_avg_pool(xi).squeeze(2) for xi in x]

        # Concatenate along the filter dimension
        x = torch.cat(x, 1)

        # Flatten
        x = x.view(x.size(0), -1)

        # Process BERT embeddings
        bert_embeddings = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x_bert = bert_embeddings.last_hidden_state

        # Apply attention in BERT embeddings
        x_bert = self.bert_attention(x_bert)

        # Add a dimension to match the expected input shape
        x_bert = x_bert.unsqueeze(-1)

        # BERT Global average pooling
        x_bert = self.bert_global_avg_pool(x_bert).squeeze(-1)

        # Concatenate Word2Vec and BERT embeddings
        x_combined = torch.cat((x, x_bert), 1)

        # Apply dropout
        x_combined = self.dropout(x_combined)

        # Apply fully connected layer
        x = self.fc(x_combined)

        if prediction:
          return x, x_combined
        else:
          return x

class BERTweet(nn.Module):
    """Model_2.0"""
    def __init__(self):
        super(BERTweet, self).__init__()

        # BERTweet Embedding Layer
        self.bertweet = AutoModel.from_pretrained('vinai/bertweet-base')

        # Fully connected layer: Adjust according to your task
        self.fc = nn.Linear(self.bertweet.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, prediction = False):

        # Process BERTweet embeddings
        bertweet_embeddings = self.bertweet(input_ids=input_ids, attention_mask=attention_mask)

        # Global Average Pooling
        x_bertweet = bertweet_embeddings.pooler_output

        # Apply fully connected layer
        x = self.fc(x_bertweet)

        if prediction:
            return x, x_bertweet
        else:
            return x

class CNNForWord2VecBERTweet(nn.Module):
    """Model_2.1"""
    def __init__(self, word2vec_weights, vocab_size, embedding_dim, num_filters, filter_sizes, dropout_rate):
        super(CNNForWord2VecBERTweet, self).__init__()

        # WORD2VEC Embedding layer
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(word2vec_weights))

        # Convolutional layers: Adjusted for embedding dimensions
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(k, embedding_dim), padding=(k - 1, 0)) for k in filter_sizes
        ])

        # Batch normalization layers: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d
        self.conv_bn = nn.ModuleList([
            nn.BatchNorm2d(num_filters) for _ in filter_sizes
        ])

        # Global Average Pooling layer for CNN features
        self.cnn_global_avg_pool = nn.AdaptiveAvgPool2d((1, num_filters))

        # BERTweet Embedding Layer
        self.bertweet = AutoModel.from_pretrained('vinai/bertweet-base')

        # global average pooling layer for BERTweet embeddings
        self.bertweet_global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layer: Adjust according to your task
        self.fc = nn.Linear(num_filters * len(filter_sizes) + embedding_dim, 1) # The "* 2" accounts for concatenation of avg and max pooling features

    def forward(self, input_ids, attention_mask, word_indices, prediction = False):
        # x shape: [batch_size, max_sequence_length, embedding_dim]

        # Word2Vec Embeddings

        # Convert ids to embeddings
        x = self.embedding(word_indices)

        # Add a channel dimension: [batch_size, 1, max_sequence_length, embedding_dim]
        x = x.unsqueeze(1)

        # Apply convolutions and ReLU
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]

        # Apply global average pooling
        x = [self.cnn_global_avg_pool(xi).squeeze(2) for xi in x]

        # Concatenate along the filter dimension
        x = torch.cat(x, 1)

        # Flatten
        x = x.view(x.size(0), -1)

        # Process BERTweet embeddings
        bertweet_embeddings = self.bertweet(input_ids=input_ids, attention_mask=attention_mask)
        x_bertweet = bertweet_embeddings.last_hidden_state


        # Apply mean pooling
        x_bertweet = x_bertweet.mean(dim=1)

        # Concatenate Word2Vec and BERTweet embeddings
        x_combined = torch.cat((x, x_bertweet), 1)

        # Apply dropout
        x_combined = self.dropout(x_combined)

        # Apply fully connected layer
        x = self.fc(x_combined)

        if prediction:
          return x, x_combined
        else:
          return x

class CNNForWord2VecBERTweetFT(nn.Module):
    """Model_2.2"""
    def __init__(self, word2vec_weights, vocab_size, embedding_dim, num_filters, filter_sizes, dropout_rate, hidden_dim, freeze=True):
        super(CNNForWord2VecBERTweetFT, self).__init__()

        # WORD2VEC Embedding layer
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(word2vec_weights))

        # Convolutional layers: Adjusted for embedding dimensions
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(k, embedding_dim), padding=(k - 1, 0)) for k in filter_sizes
        ])

        # Batch normalization layers: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d
        self.conv_bn = nn.ModuleList([
            nn.BatchNorm2d(num_filters) for _ in filter_sizes
        ])

        # Global Average Pooling layer for CNN features
        self.cnn_global_avg_pool = nn.AdaptiveAvgPool2d((1, num_filters))

        # BERT Embedding Layer
        self.bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
        self.bertweet_attention = Attention(embedding_dim, hidden_dim)  # Initialize attention for BERT embeddings

        # Freeze BERT layer
        if freeze:
            for param in self.bertweet.parameters():
                param.requires_grad = False

        # global average pooling layer for BERT embeddings
        self.bertweet_global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layer: Adjust according to your task
        self.fc = nn.Linear(num_filters * len(filter_sizes) + embedding_dim, 1) # The "* 2" accounts for concatenation of avg and max pooling features

    def forward(self, input_ids, attention_mask, word_indices, prediction = False):
        # x shape: [batch_size, max_sequence_length, embedding_dim]

        # Word2Vec Embeddings

        # Convert ids to embeddings
        x = self.embedding(word_indices)

        # Add a channel dimension: [batch_size, 1, max_sequence_length, embedding_dim]
        x = x.unsqueeze(1)

        # Apply convolutions and ReLU
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]

        # Apply global average pooling
        x = [self.cnn_global_avg_pool(xi).squeeze(2) for xi in x]

        # Concatenate along the filter dimension
        x = torch.cat(x, 1)

        # Flatten
        x = x.view(x.size(0), -1)

        # Process BERTweet embeddings
        bertweet_embeddings = self.bertweet(input_ids=input_ids, attention_mask=attention_mask)
        x_bertweet = bertweet_embeddings.last_hidden_state

        # Apply attention in BERT embeddings
        x_bertweet = self.bertweet_attention(x_bertweet)

        # Add a dimension to match the expected input shape
        x_bertweet = x_bertweet.unsqueeze(-1)

        # BERT Global average pooling
        x_bertweet = self.bertweet_global_avg_pool(x_bertweet).squeeze(-1)

        # Concatenate Word2Vec and BERT embeddings
        x_combined = torch.cat((x, x_bertweet), 1)

        # Apply dropout
        x_combined = self.dropout(x_combined)

        # Apply fully connected layer
        x = self.fc(x_combined)

        if prediction:
          return x, x_combined
        else:
          return x

class ALBERT(nn.Module):
    """Model_3.0"""
    def __init__(self):
        super(ALBERT, self).__init__()

        # ALBERT Embedding Layer
        self.ALBERT = AutoModel.from_pretrained('albert/albert-base-v2')

        # Fully connected layer: Adjust according to your task
        self.fc = nn.Linear(self.ALBERT.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, prediction = False):

        # Process ALBERT embeddings
        ALBERT_embeddings = self.ALBERT(input_ids=input_ids, attention_mask=attention_mask)

        # Global Average Pooling
        x_ALBERT = ALBERT_embeddings.pooler_output

        # Apply fully connected layer
        x = self.fc(x_ALBERT)

        if prediction:
            return x, x_ALBERT
        else:
            return x

class CNNForWord2VecALBERT(nn.Module):
    """Model_3.1"""
    def __init__(self, word2vec_weights, vocab_size, embedding_dim, num_filters, filter_sizes, dropout_rate):
        super(CNNForWord2VecALBERT, self).__init__()

        # WORD2VEC Embedding layer
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(word2vec_weights))

        # Convolutional layers: Adjusted for embedding dimensions
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(k, embedding_dim), padding=(k - 1, 0)) for k in filter_sizes
        ])

        # Batch normalization layers: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d
        self.conv_bn = nn.ModuleList([
            nn.BatchNorm2d(num_filters) for _ in filter_sizes
        ])

        # Global Average Pooling layer for CNN features
        self.cnn_global_avg_pool = nn.AdaptiveAvgPool2d((1, num_filters))

        # ALBERT Embedding Layer
        self.ALBERT = AutoModel.from_pretrained('albert/albert-base-v2')

        # global average pooling layer for ALBERT embeddings
        self.ALBERT_global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layer: Adjust according to your task
        self.fc = nn.Linear(num_filters * len(filter_sizes) + embedding_dim, 1) # The "* 2" accounts for concatenation of avg and max pooling features

    def forward(self, input_ids, attention_mask, word_indices, prediction = False):
        # x shape: [batch_size, max_sequence_length, embedding_dim]

        # Word2Vec Embeddings

        # Convert ids to embeddings
        x = self.embedding(word_indices)

        # Add a channel dimension: [batch_size, 1, max_sequence_length, embedding_dim]
        x = x.unsqueeze(1)

        # Apply convolutions and ReLU
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]

        # Apply global average pooling
        x = [self.cnn_global_avg_pool(xi).squeeze(2) for xi in x]

        # Concatenate along the filter dimension
        x = torch.cat(x, 1)

        # Flatten
        x = x.view(x.size(0), -1)

        # Process ALBERT embeddings
        ALBERT_embeddings = self.ALBERT(input_ids=input_ids, attention_mask=attention_mask)
        x_ALBERT = ALBERT_embeddings.last_hidden_state


        # Apply mean pooling
        x_ALBERT = x_ALBERT.mean(dim=1)

        # Concatenate Word2Vec and ALBERT embeddings
        x_combined = torch.cat((x, x_ALBERT), 1)

        # Apply dropout
        x_combined = self.dropout(x_combined)

        # Apply fully connected layer
        x = self.fc(x_combined)

        if prediction:
          return x, x_combined
        else:
          return x

class CNNForWord2VecALBERTFT(nn.Module):
    """Model_3.2"""
    def __init__(self, word2vec_weights, vocab_size, embedding_dim, num_filters, filter_sizes, dropout_rate, hidden_dim, freeze=True):
        super(CNNForWord2VecALBERTFT, self).__init__()

        # WORD2VEC Embedding layer
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(word2vec_weights))

        # Convolutional layers: Adjusted for embedding dimensions
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(k, embedding_dim), padding=(k - 1, 0)) for k in filter_sizes
        ])

        # Batch normalization layers: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d
        self.conv_bn = nn.ModuleList([
            nn.BatchNorm2d(num_filters) for _ in filter_sizes
        ])

        # Global Average Pooling layer for CNN features
        self.cnn_global_avg_pool = nn.AdaptiveAvgPool2d((1, num_filters))

        # ALBERT Embedding Layer
        self.ALBERT = AutoModel.from_pretrained('albert/albert-base-v2')
        self.ALBERT_attention = Attention(embedding_dim, hidden_dim)  # Initialize attention for ALBERT embeddings

        # Freeze ALBERT layer
        if freeze:
            for param in self.ALBERT.parameters():
                param.requires_grad = False

        # global average pooling layer for ALBERT embeddings
        self.ALBERT_global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layer: Adjust according to your task
        self.fc = nn.Linear(num_filters * len(filter_sizes) + embedding_dim, 1) # The "* 2" accounts for concatenation of avg and max pooling features

    def forward(self, input_ids, attention_mask, word_indices, prediction = False):
        # x shape: [batch_size, max_sequence_length, embedding_dim]

        # Word2Vec Embeddings

        # Convert ids to embeddings
        x = self.embedding(word_indices)

        # Add a channel dimension: [batch_size, 1, max_sequence_length, embedding_dim]
        x = x.unsqueeze(1)

        # Apply convolutions and ReLU
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]

        # Apply global average pooling
        x = [self.cnn_global_avg_pool(xi).squeeze(2) for xi in x]

        # Concatenate along the filter dimension
        x = torch.cat(x, 1)

        # Flatten
        x = x.view(x.size(0), -1)

        # Process ALBERT embeddings
        ALBERT_embeddings = self.ALBERT(input_ids=input_ids, attention_mask=attention_mask)
        x_ALBERT = ALBERT_embeddings.last_hidden_state

        # Apply attention in ALBERT embeddings
        x_ALBERT = self.ALBERT_attention(x_ALBERT)

        # Add a dimension to match the expected input shape
        x_ALBERT = x_ALBERT.unsqueeze(-1)

        # ALBERT Global average pooling
        x_ALBERT = self.ALBERT_global_avg_pool(x_ALBERT).squeeze(-1)

        # Concatenate Word2Vec and ALBERT embeddings
        x_combined = torch.cat((x, x_ALBERT), 1)

        # Apply dropout
        x_combined = self.dropout(x_combined)

        # Apply fully connected layer
        x = self.fc(x_combined)

        if prediction:
          return x, x_combined
        else:
          return x


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
  

"""
OPTUNA RESULTS
"""

def create_results_dataframe(study):
    # Create a list to hold all trial data
    trial_data = []

    # Iterate through all completed trials
    for trial in study.trials:
        # Retrieve the user attributes for the trial
        user_attrs = trial.user_attrs
        user_attrs["trial_number"] = trial.number
        user_attrs["value"] = trial.value  # The objective value (e.g., validation F1 score)

        # Append the trial data to the list
        trial_data.append(user_attrs)

    # Create a DataFrame from the list of trial data
    df = pd.DataFrame(trial_data)

    # Optionally, you might want to sort the DataFrame based on the objective value or another metric
    df = df.sort_values("value", ascending=False)

    return df


"""
PREDICTION RESULTS FOR TRAIN, VAL AND TEST DATASETS
"""

# Function to determine category
def determine_category(row):
    if row['label'] == 1 and row['pred'] == 1:
        return 'TP'  # True Positive
    elif row['label'] == 0 and row['pred'] == 0:
        return 'TN'  # True Negative
    elif row['label'] == 0 and row['pred'] == 1:
        return 'FP'  # False Positive
    elif row['label'] == 1 and row['pred'] == 0:
        return 'FN'  # False Negative


def make_prediction(model, df, date_set, name):
  # Create the dataloaders
  data_loader = DataLoader(date_set, batch_size, shuffle=False)

  # Assuming model is your trained model and training_loader is your DataLoader
  model.eval()  # Set the model to evaluation mode

  predictions = []
  true_labels = []
  device= 'cuda'

  with torch.no_grad():  # No need to track gradients for predictions
      for data in tqdm(data_loader, desc="Prediction"):
          word_indices = data['word_indices'].to(device)
          bert_inputs = {key: value.to(device) for key, value in data['bert_inputs'].items()}
          targets = data['targets'].to(device)

          # Forward pass, get predictions
          outputs = model(word_indices, bert_inputs)

          # Convert outputs to probabilities (if your model outputs logits)
          probs = torch.sigmoid(outputs).squeeze()

          # Optionally, convert probabilities to binary predictions
          # For example, using a threshold of 0.5
          batch_predictions = (probs >= 0.5).long()

          # Store predictions and true labels for later use
          predictions.extend(batch_predictions.cpu().numpy())
          true_labels.extend(targets.cpu().numpy())

  results = df.copy()
  results['pred'] = predictions

  # Apply the function to create a new column 'category'
  results['category'] = results.apply(determine_category, axis=1)

  calculate_metrics(predictions, true_labels, name)

  return results


def calculate_metrics(predictions, true_labels, name):

  print(f'============={name}==============')

  accuracy = accuracy_score(true_labels, predictions)
  precision = precision_score(true_labels, predictions)
  recall = recall_score(true_labels, predictions)
  f1 = f1_score(true_labels, predictions)
  print('accuracy:', accuracy, 'precision', precision, 'recall:', recall, 'f1:', f1)

  # ROC AUC Curve
  fpr, tpr, thresholds = roc_curve(true_labels, predictions)
  roc_auc = auc(fpr, tpr)

  plt.figure()
  plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic')
  plt.legend(loc="lower right")
  plt.show()

  # Calibration Curve
  prob_true, prob_pred = calibration_curve(true_labels, predictions, n_bins=10)
  plt.figure()
  plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='calibration plot')
  plt.plot([0, 1], [0, 1], linestyle='--', label='perfectly calibrated')
  plt.xlabel('Mean predicted probability')
  plt.ylabel('Fraction of positives')
  plt.title('Calibration plot')
  plt.legend()
  plt.show()

  # Precision-Recall Curve
  precision, recall, _ = precision_recall_curve(true_labels, predictions)
  plt.figure()
  plt.plot(recall, precision, marker='.', label='Precision-Recall curve')
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.title('Precision-Recall curve')
  plt.legend()
  plt.show()

  # Confusion Matrix
  cm = confusion_matrix(true_labels, predictions)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm)
  disp.plot()
  plt.title('Confusion Matrix')
  plt.show()

  # Classification Report
  print(classification_report(true_labels, predictions))





