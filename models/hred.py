############################################################
# Module  : Reimplement HRED
# Date    : March 31st, 2017
# Author  : xiao ling, Heejing Jeong
############################################################

from __future__ import print_function

import os
import random
import operator
import numpy as np
import tensorflow as tf

from utils import *


############################################################
'''
    Hred primitives
'''


'''
    TODO: understand exactly what this is???

    @Use: construct encoder
    @Given:  - size of the vocab `vocab_size` :: Int
             - list of `cell_sizes`           :: [Int]
    @Output: a list of multi-rnn cells        :: [MultiRNNCell]
'''
def to_encoder(vocab_size, cell_sizes):

    cells = []

    for cell_size in cell_sizes:
        if not cells:
            cell, _ = build_layer(vocab_size, cell_size)
            cells.append(cell)
        else:
            cell, _ = build_layer(cells[i-1].state_size, cell_size)
            cells.append(cell)      
    
    return cells            


'''
    @Use: given dimension of input `input_dim` and number 
          of cells in the layer `layer_size`,
          output a multiRnnCell

    @Input: 
        - input dimension :: Int : dimension of input vector space
        - cell size       :: Int  :  what exactly is this?
    @Output: rnn cell     :: MultiRNNCell
'''
def build_layer(input_dim, cell_size):

    cells = [tf.contrib.rnn.LSTMCell(input_dim) \
            for _ in xrange(cell_size)]

    return tf.contrib.rnn.MultiRNNCell(cells),cells 

############################################################
'''
    Batching functions

    @Use  :  get a random batch of from `data_set`
             pad source and target

    @Input: - data_set binned into buckets, each bucket is a list
              of tuples, with question and response pairs
              of type            :: [[([Int], [Int])]] 
            - batch size         :: Int
            - pad id             :: Int

    @Output: encoder input of dimension : encoder_length * batch_size        
             decoder input of dimension : decoder_length * batch_size        
             mask of dimension          : decoder_length * * batch_size

'''
def get_batch(data, seq_len, batch_size, PAD_ID):
    '''
        select arbitrary passages from data
    '''
    randbits = [random.randint(0,len(data)-1) for _ in range(batch_size)]
    samples  = [data[b] for b in randbits]

    source  = [to_source(PAD_ID, seq_len, q) for q,_ in samples]
    target  = [to_target(PAD_ID, seq_len, r) for _,r in samples]

    '''
        output source,target,mask as matrices of dimension:
            batch_size * seq_len
    '''
    inputs  = np.transpose (source)
    outputs = np.transpose (target)
    mask    = np.transpose([[np.sign(t).astype(float) for t in ts] for ts in target])

    return inputs, outputs, mask, randbits

'''
    @Use: given 
            - numerical id of `PAD_ID`                  :: Int
            - length of encoder sentence `encoder_len`, :: Int
            - list of indices `idx`                     :: [Int]
    @output: !not-reversed! and padded encoder sequence
'''
def to_source(PAD_ID, source_len, idx):
    return idx + [PAD_ID] * (source_len - len(idx))


'''
    @Use: given 
            - numerical id of `PAD_ID` 
            - length of encoder sentence `decoder_len`,
            - list of indices `idx`
    @output: padded decoder sequence
'''
def to_target(PAD_ID, target_len, idx):
    return idx + [PAD_ID] * (target_len - len(idx))

############################################################
'''
    unit tests
'''
def unit_test_get_batch():

    # for now take seq_size as some constant
    seq_len    = 10
    batch_size = 2
    PAD_ID     = 0

    data = dummy_data(seq_len)  

    '''
        get_batch
    '''
    os,ts, ms, _ = get_batch(data, seq_len, batch_size, PAD_ID)

    print('\n>> ran get_batch_unit_test')

''' 
    dummy data

    @Use: generate fake data where each question and response has 
          length `seq_length`
    @Output: results of list of tuples of form:
            (question, response)
            where each question and response is a list of
            of indicies
            the indicies are not zero padded and can be of length
            [1, seq_length]
'''
def dummy_data(seq_len):
    num_data = 10
    inputs  = [rnd.randint(0,100, random.randint(1,seq_len)).tolist() for _ in range(num_data)]
    outputs = [rnd.randint(0,100, random.randint(1,seq_len)).tolist() for _ in range(num_data)]
    data    = zip(inputs, outputs)
    return data



