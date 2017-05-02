""" Module     : Make data that conform to what this wants
    Contributor: Xiao Ling
    Date       : April 13th, 2017
"""

import os
import numpy as np
import tensorflow as tf
import subprocess
import cPickle as pickle
import math
import unittest

from utils import *
import app.config as config


from models.ts_hred.src.sordoni.SS_dataset import *
import itertools


import models.ts_hred.src.sordoni.data_iterator as with_data

############################################################

# now we need to decode and encode it 
'''
    @Use: given vocab dict and list of indices, output string of words
'''
def decode(idx2w, bs):
    return ' '.join(idx2w[b] for b in bs)

def encode(w2idx, ws):
    words = ws.split(' ')

    bs    = []

    for w in words:
        if w in w2idx: 
            bs.append(w2idx[w])
        else:
            bs.append(w2idx['<unk>'])
    return bs

def unit_test_encode_decode():

    # round trip
    xs = 'hello world'
    assert decode(idx2w, encode(w2idx, xs)) == xs

    print "round trip!"


'''
    check our index file is consistent
    and we can load training data as expected
    where you left off:
        you need to convert this to a picke file before running
'''

w2idx = pickle.loads(open(config.get_path('w2idx'),'rb').read())
idx2w = pickle.loads(open(config.get_path('idx2w'),'rb').read())

# train_my_path = os.path.join(config.get_path('sess-concat'), 'train.pkl')
# test_my_path  = os.path.join(config.get_path('sess-concat'), 'test.pkl')
# train_my = pickle.load(open(train_my_path,'rb'))
# test_my  = pickle.load(open(test_my_path,'rb'))


# q = train_my[0]
# a = train_my[1]


############################################################
'''
    play with sordoni_data_iterator
'''
PROJ_ROOT          = os.getcwd()
ROOT               = os.path.join(PROJ_ROOT, 'models/ts_hred/')

train_file = os.path.join(ROOT,'data/sordoni/dev_large/train.ses.pkl')
valid_file = os.path.join(ROOT,'data/sordoni/dev_large/valid.ses.pkl')
vocab_file = os.path.join(ROOT,'data/sordoni/dev_large/train.dict.pkl')

train = pickle.loads(open(train_file).read())
vocab = pickle.loads(open(vocab_file).read())

idx2w = { v : k for k,v,_ in vocab }

vocab_size = 20003

unk_symbol = 0
eoq_symbol = 1
eos_symbol = 2

'''
    Network Parameters
'''
n_buckets     = 20
max_itter     = 10000000
seed          = 1234

embedding_dim = 64
query_dim     = 128
session_dim   = 256
batch_size    = 50
max_length    = 400


state = {
        'eoq_sym'       : eoq_symbol,
        'eos_sym'       : eos_symbol,
        'sort_k_batches': n_buckets,
        'bs'            : batch_size,
        'train_session' : train_file,
        'seqlen'        : max_length,
        'valid_session' : valid_file,
    }

train_data, valid_data = with_data.get_batch_iterator(
      np.random.RandomState(seed)
    , state)

'''
    learn how they finangle the data
'''
train_data.start()
data        = train_data.next()
seq_len     = data['max_length']
prepend     = np.ones((1, data['x'].shape[1]))
x_data_full = np.concatenate((prepend, data['x']))
x_batch     = x_data_full[:seq_len]
y_batch     = x_data_full[1:seq_len + 1]

data1   = train_data.next()

bs1 = np.ndarray.tolist(data['x'][0])
bs2 = np.ndarray.tolist(data1['x'][0])

'''
    understand get_batch_iterator
'''
# s = np.random.RandomState(seed)
# k_batches = state['sort_k_batches']
# batch_size = state['bs']
               
# SSIterator.__init__(self, s, *args, **kwargs)
 



















