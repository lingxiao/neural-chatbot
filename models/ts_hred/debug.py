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


############################################################

root            = get_path('model/hred')
data_root       = get_path('movie/sess-concat')
chkpt_root      = get_path('checkpoint')

logs_dir        = os.path.join(root      , 'logs')
checkpoint_file = os.path.join(chkpt_root, 'hred/hred-movie.ckpt')
train_file      = os.path.join(data_root, 'train.pkl')
valid_file      = os.path.join(data_root, 'test.pkl')
idx2w_file      = get_path('movie/idx2w')

############################################################

vocab_lookup_dict = cPickle.load(open(idx2w_file,'rb'))
