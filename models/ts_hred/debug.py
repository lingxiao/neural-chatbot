""" File to build and train the entire computation graph in tensorflow
"""
import os
import numpy as np
import tensorflow as tf
import subprocess
import pickle
import cPickle
import math

from app import *
tf.logging.set_verbosity(tf.logging.DEBUG) # test

from models.ts_hred.src.hred.train import Trainer
import models.ts_hred.src.sordoni.data_iterator as sordoni_data_iterator



unk_symbol    = 1
eoq_symbol    = 3    # end of sentence
eos_symbol    = 4    # end of conversation

n_buckets     = 20
max_itter     = 10000000

vocab_size    = 50005
embedding_dim = 64
session_dim   = 4
batch_size    = 64
query_dim     = 128
max_length    = 220


############################################################

root            = get_path('model/hred')
data_root       = get_path('movie/sess-concat')
chkpt_root      = get_path('checkpoint')

logs_dir        = os.path.join(root      , 'logs')
checkpoint_file = os.path.join(chkpt_root, 'hred/hred-movie.ckpt')
train_file      = os.path.join(data_root, 'train.pkl')
valid_file      = os.path.join(data_root, 'valid.pkl')
idx2w_file      = get_path('movie/idx2w')


############################################################

idx2w = cPickle.load( open(idx2w_file,'rb') )

with open(valid_file,'rb') as h:
	ws = pickle.load(h)

SEED = 1234 

train, valid = sordoni_data_iterator.get_batch_iterator(np.random.RandomState(SEED), {
	    'eoq_sym'       : eoq_symbol ,
	    'eos_sym'       : eos_symbol ,
	    'sort_k_batches': n_buckets  ,
	    'bs'            : 100        ,
	    'train_session' : train_file ,
	    'seqlen'        : 200        ,
	    'valid_session' : valid_file
	})


train.start()

data = train.next()

data_x = data['x']
data_y = data['y']





