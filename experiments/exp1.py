############################################################
# Module  : hw4 - main
# Date    : March 11th, 2017
# Author  : xiao ling, Heejing Jeong
############################################################

from __future__ import print_function

import os
import nltk
import pickle
import random 
import numpy as np
import numpy.random as rnd
import operator

import tensorflow as tf
# import tf.contrib.rnn as rnn
# from tensorflow.python.ops.rnn import *
# from tensorflow.python.ops.rnn_cell import *
# from tensorflow.python.ops import variable_scope, seq2seq
# from tensorflow.contrib.rnn.python.ops import core_rnn
# from tensorflow.contrib.rnn.python.ops import core_rnn_cell
# from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq

from app.config import PATH
from models import *
from utils import *


os.system('clear')

'''
	Network setting
'''
SETTING = {
		'vocab': {
			'tokens': {
				# special vocab tokens
				  'unk'   : '<unk>'
				, 'pad'   : '_' 
				, 'go'    : '<go>'}
			, 'vocab-size'  : 50
			# maximum question and answer lengths
			# note this is set very high because
			# we are trying to learn sequential
			# dependence in conversation snippets,
			# so no intermediate responses can be
			# removed
	        , 'maxq' : 10
	        , 'maxr' : 10
			},
        'network': {
	          'lstm-hidden-size': 5
	        , 'lstm-num-layers' : 4
	        , 'learning-rate'   : 0.1
        }}

############################################################
'''	
	dummy data

	@Use: generate fake data where each question and response has 
	      length `max_seq_length`
    @Output: results of list of tuples of form:
    		(question, response)
    		where each question and response is a list of
    		of indicies
    		the indicies are not zero padded and can be of length
    		[1, max_seq_length]

'''
def dummy_data(max_seq_len):
	num_data = 10
	inputs  = [rnd.randint(0,100, random.randint(1,max_seq_len)).tolist() for _ in range(num_data)]
	outputs = [rnd.randint(0,100, random.randint(1,max_seq_len)).tolist() for _ in range(num_data)]
	data    = zip(inputs, outputs)
	return data


# for now take seq_size as some constant
max_seq_len    = 10
batch_size = 2

PAD_ID     = 0
EOS_ID     = 1  # end of session char


data = dummy_data(max_seq_len)	

'''
	get_batch
'''
source,target,mask, _ = get_batch(data, max_seq_len, batch_size, PAD_ID)


############################################################
'''
	model
'''
# it seems like it's preduent to first understand
# what the parameters are doing

'''
	input parameters
'''
vocab_size = 100
batch_size = 2

'''
	TODO: figure out what this is??
'''
topology   = [10,5]   
cell_sizes = [8 ,2]

'''
	network parameters	
'''
num_layers     = len(topology)
max_seq_len    = reduce(operator.mul, topology, 1)
cell_type      = 'BasicLSTMCell'
embed          = False
forward_only   = False


def build_inputs(batch_size, seq_len, input_dim):
	return [tf.placeholder(tf.float32, \
		   [batch_size, input_dim]) \
	        for _ in range(seq_len)]


# enc_cells  = []
# dec_cells  = []
# dec_inputs = []

# for cell_size, seq_size in zip(cell_sizes, topology):

# 	input_dim = vocab_size if not encoder \
# 	      else encoder[-1].state_size

# 	cell, _   = build_layer(input_dim, cell_size)

# 	# dec_input = build_inputs(batch_size, seq_size, input_dim)

# 	enc_cells.append(cell)
# 	dec_cells.append(cell)


def _build_layer(input_size, layer_size):


    enc_cell = tf.contrib.rnn.LSTMCell(input_size)

    if layer_size > 1:

        enc_cell = [enc_cell]

        for _ in range(1, layer_size):
            enc_cell.append(tf.contrib.rnn.LSTMCell(input_size, enc_cell[-1].output_size))
        enc_cell = tf.contrib.rnn.MultiRNNCell(enc_cell)

    return enc_cell

def build_inputs(batch_size, seq_len, input_size):
    return [tf.placeholder(tf.float32, [batch_size, input_size]) for _ in range(seq_len)]

enc_cells = []    

for i in range(0, 2):

    size = enc_cells[i - 1].state_size \
           if i > 0 else vocab_size
    cell = _build_layer(size, cell_sizes[i])

    enc_cells.append(cell)
	
    # dec_input = build_inputs(batch_size, topology[i], size)







'''
	a matrix of dimension: max_seq_len * batch_size * vocab_size
	question: what exactly is a a seq_len ???	

enc_inputs = [tf.placeholder(tf.float32              \
	           , [batch_size, vocab_size]            \
	           , name='Encoder_Input_{}'.format(q))  \
			  for q in range(max_seq_len)]
'''
