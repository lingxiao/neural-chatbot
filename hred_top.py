""" File to build and train the entire computation graph in tensorflow
"""
import os
import numpy as np
import tensorflow as tf
import subprocess
import cPickle
import math

from app import *
tf.logging.set_verbosity(tf.logging.DEBUG) # test

from models.ts_hred.src.hred.train import Trainer

root            = get_path('model/hred')
data_root       = get_path('movie/sess-concat')
chkpt_root      = get_path('checkpoint')

logs_dir        = os.path.join(root      , 'logs')
checkpoint_file = os.path.join(chkpt_root, 'hred/hred-movie.ckpt')
train_file      = os.path.join(data_root, 'train.pkl')
valid_file      = os.path.join(data_root, 'test.pkl')
idx2w_file      = get_path('movie/idx2w')

unk_symbol = 1
eoq_symbol = 3    # end of sentence
eos_symbol = 4    # end of conversation

n_buckets = 20
max_itter = 10000000

vocab_size    = 50005
embedding_dim = 64
query_dim     = 128
session_dim   = 256
batch_size    = 50
max_length    = 50

if __name__ == '__main__':

	with tf.Graph().as_default():

	    trainer = Trainer(

	              CHECKPOINT_FILE = checkpoint_file
	            , idx2w_file      = idx2w_file
	            , TRAIN_FILE      = train_file
	            , VALID_FILE      = valid_file
	            , LOGS_DIR        = logs_dir
	            
	            , EMBEDDING_DIM = embedding_dim
	            , SESSION_DIM   = session_dim
	            , QUERY_DIM     = query_dim
	            , BATCH_SIZE    = batch_size
	            , MAX_LENGTH    = max_length
	            
	            , VOCAB_SIZE = vocab_size
	            , EOQ_SYMBOL = eoq_symbol
	            , EOS_SYMBOL = eos_symbol
	            , UNK_SYMBOL = unk_symbol

	            , N_BUCKETS  = n_buckets
	            , MAX_ITTER  = max_itter

	            , RESTORE    = True)

	    trainer.train(batch_size=batch_size, max_length=max_length)




