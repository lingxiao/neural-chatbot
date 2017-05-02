""" Module     : Make data that conform to what this wants
    Contributor: Xiao Ling
    Date       : April 13th, 2017
"""

import os
import numpy as np
import tensorflow as tf
import subprocess
import cPickle
import math
import unittest

from utils import *
import app.config as config

import models.ts_hred.src.sordoni.data_iterator as sordoni_data_iterator
from models.ts_hred.src.hred.utils     import make_attention_mask
from models.ts_hred.src.hred.optimizer import Optimizer
from models.ts_hred.src.hred.hred      import HRED
from models.ts_hred.src.hred.trainer   import Trainer

############################################################
'''
    prep data
'''
PROJ_ROOT          = os.getcwd()
ROOT               = os.path.join(PROJ_ROOT, 'models/ts_hred/')

SORDONI_VOCAB_FILE = os.path.join(ROOT,'data/sordoni/dev_large/train.dict.pkl')
SORDONI_TRAIN_FILE = os.path.join(ROOT,'data/sordoni/dev_large/train.ses.pkl')
SORDONI_VALID_FILE = os.path.join(ROOT,'data/sordoni/dev_large/valid.ses.pkl')
VOCAB_SIZE         = 50003

# observe this is a set of sentences
train = cPickle.loads(open(SORDONI_TRAIN_FILE).read())
valid = cPickle.loads(open(SORDONI_VALID_FILE).read())

'''
    we need a dictionary where:

    vocab = {'<unk>': 0, '</q>': 1, '</s>': 2}

    train_dict = [('<unk>', 0,0), ('</q>', 1,0), ('</s>', 2,0)]
'''
vocab = cPickle.loads(open(SORDONI_VOCAB_FILE,'rb').read())

v_lookup = {k: v for v, k, count in vocab}

w2idx = { k : v for k,v,_ in vocab }

idx2w = { v : k for k,v in w2idx.iteritems() }


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

w2idx_my = cPickle.loads(open(config.get_path('w2idx'),'rb').read())
idx2w_my = cPickle.loads(open(config.get_path('idx2w'),'rb').read())

train_my = np.load(os.path.join(config.get_path('sess-concat'), 'train.npy'))
test_my  = np.load(os.path.join(config.get_path('sess-concat'), 'test.npy'))































