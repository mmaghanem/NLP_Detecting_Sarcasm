"""
GLOBAL VARIABLES DICTIONARY
Contains a comprehensive list of file names
associated with keys to be used to import across
project files
"""

FPATH = '/content/drive/MyDrive/266_project/project/support/'

GVD = {

  # Embeddings for model
  'bert_embeddings' : 'bert_embeddings.pkl',

  # Embeddings
  'text_for_embeddings' : 'text_for_embeddings.csv',
  'word2vec_embeddings' : 'word2vec_embeddings.pt',  
  'word2vec_model' : 'word2vec_model.model',
  'word2vec_model_syn1neg' : 'word2vec_model.model.syn1neg.npy',
  'word2vec_model_wv_vectors' : 'word2vec_model.model.wv.vectors.npy',

  # train val and test csv
  'train_df' : 'train_df.csv',
  'val_df' : 'val_df.csv',
  'test_df' : 'test_df.csv',
    
  # Sampled train val and test csv
  'train_df_sample' : 'train_df_sample.csv',
  'val_df_sample' : 'val_df.csv',
  'test_df_sample' : 'test_df.csv',

  # Optuna
  'optuna_study_w2v_bert' : 'optuna_study_w2v_bert.pkl',

  # Model output for embedding analysis
  'results_for_embedding_analysis' : 'results_for_embedding_analysis.csv',

}