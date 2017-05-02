""" Module     : File to train on movie data
    Contributor: Xiao Ling
    Date       : April 13th, 2017
"""

import os
import numpy as np
import tensorflow as tf
import subprocess
import cPickle
import math

import src.sordoni.data_iterator as sordoni_data_iterator
from src.hred.utils     import make_attention_mask
from src.hred.optimizer import Optimizer
from src.hred.hred      import HRED
from src.hred.trainer   import Trainer

tf.logging.set_verbosity(tf.logging.DEBUG) # test

############################################################
'''
    Paths
    ROOT should be project directory for ts-hred:
    ie: $HOME/neural-chatbot/models/ts-hred
'''


# ROOT               = os.getcwd()

# VALIDATION_FILE    = os.path.join(ROOT, 'data/val_session.out' )
# TEST_FILE          = os.path.join(ROOT, 'data/test_session.out')
# LOGS_DIR           = os.path.join('logs')
# CHECKPOINT_FILE    = os.path.join('checkpoints/model-huge-attention-fixed.ckpt')

# SORDONI_VOCAB_FILE = os.path.join(ROOT,'data/sordoni/dev_large/train.dict.pkl')
# SORDONI_TRAIN_FILE = os.path.join(ROOT,'data/sordoni/dev_large/train.ses.pkl')
# SORDONI_VALID_FILE = os.path.join(ROOT,'data/sordoni/dev_large/valid.ses.pkl')
# VOCAB_SIZE         = 50003

# '''
#     Vocab Parameters
#     conform to plan with heejin
# '''
# SETTING = {
#         'tokens':{
#               'pad'        : '_'
#             , 'unk'        : '<unk>'
#             , 'go'         : '<go>'
#             , 'eos'        : '</s>'
#             , 'eoq'        : '</q>'
#             }

#         , 'vocab-size' : 50000
#         , 'max-len'    : 400}

# PAD_ID = 0
# UNK_ID = 1
# GO_ID  = 2
# EOS_ID = 3
# EOQ_ID = 4

'''
    Network Parameters
'''
N_BUCKETS     = 20
MAX_ITTER     = 10000000
SEED          = 1234


EMBEDDING_DIM = 64
QUERY_DIM     = 128
SESSION_DIM   = 256
BATCH_SIZE    = 50
MAX_LENGTH    = 50


RESTORE = True

############################################################
'''
    Train
'''
# if __name__ == '__main__':

#     with tf.Graph().as_default():

#         trainer = Trainer(

#                   vocab_file      = SORDONI_VOCAB_FILE
#                 , train_file      = SORDONI_TRAIN_FILE
#                 , VALID_FILE      = SORDONI_VALID_FILE

#                 , checkpoint_file = CHECKPOINT_FILE
#                 , logs_dir        = LOGS_DIR

#                 , vocab_size      = VOCAB_SIZE
#                 , unk_symbol      = UNK_ID
#                 , eoq_symbol      = EOQ_ID
#                 , eos_symbol      = EOS_ID

#                 , restore         = RESTORE
#                 , n_buckets       = N_BUCKETS
#                 , max_itter       = MAX_ITTER
#                 , embedding_dim   = EMBEDDING_DIM
#                 , query_dim       = QUERY_DIM
#                 , session_dim     = SESSION_DIM
#                 , batch_size      = BATCH_SIZE
#                 , max_length      = MAX_LENGTH
#                 , seed            = SEED
#                 )

#         trainer.train( batch_size = BATCH_SIZE
#                      , max_length = MAX_LENGTH)

     























