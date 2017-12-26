############################################################
# Module  : Reimplement sequence to sequence
# Date    : March 31st, 2017
# Author  : https://github.com/ematvey/tensorflow-seq2seq-tutorials/blob/master/1-seq2seq.ipynb
############################################################

from __future__ import print_function

import os
import random
import numpy as np

import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import core_rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq

from utils import *

############################################################
'''
	Private methods

	@Use  :  get a random batch of from `data_set`
			 pad and reverse input to bucket input size
			 pad output to bucket input size and append GO symbol to front
			 construct binary mask for output

	@Input: - data_set binned into buckets, each bucket is a list
	          of tuples, with question and response pairs
	          of type            :: [[([Int], [Int])]] 
	        - buckets            :: [(Int,Int)]
	        - batch size         :: Int
	        - pad id             :: Int
	        - go id              :: Int

	@Output: encoder input of dimension : encoder_length * batch_size        
	         decoder input of dimension : decoder_length * batch_size        
	         mask of dimension          : decoder_length * * batch_size
'''
def get_batch(data_set, buckets, batch_size, PAD_ID, GO_ID):

	'''
		select a random bucket and get max input and output length
	'''
	bucket_id                  = random.choice(range(len(buckets)))
	encoder_size, decoder_size = buckets[bucket_id]

	'''
		encoder_input is question, decoder input is response
		encoder and decoder inputs are matrices of dim:
			batch_size * encoder_size
	'''
	encode_decode  = [random.choice(data_set[bucket_id]) for _ in range(batch_size)]
	encoder_inputs = [to_encoder_input(PAD_ID, encoder_size, es)        for es,_ in encode_decode]
	decoder_inputs = [to_decoder_input(PAD_ID, GO_ID, decoder_size, ds) for _,ds in encode_decode]

	'''
		Each matrix is encoder_size * batch_size
	'''
	batch_encoder = np.transpose(np.array(encoder_inputs, dtype = np.int32))
	batch_decoder = np.transpose(np.array(decoder_inputs, dtype = np.int32))

	'''
		constuct binary mask for decoder inputs so that the GO_ID is clipped
		all non-padding symbols has value 1, and all pad symbols has value 0
		and an extra pad is appended to the end
	'''
	mask = [d[1:] + [PAD_ID] for d in decoder_inputs]
	mask = [[int(w != PAD_ID) for w in ws] for ws in mask]
	mask = np.transpose(np.array(mask))

	return batch_encoder, batch_decoder, mask

'''
	@Use: given 
			- numerical id of `PAD_ID` 
			- length of encoder sentence `encoder_len`,
	        - list of indices `idx`
	@output: reversed and padded encoder sequence
'''
def to_encoder_input(PAD_ID, encoder_len, idx):
	es = idx + [PAD_ID]*(encoder_len - len(idx))
	return list(reversed(es))

'''
	@Use: given 
			- numerical id of `PAD_ID` 
			- numerical id of `GO_ID` 
			- length of encoder sentence `decoder_len`,
	        - list of indices `idx`
	@output: padded decoder sequence
'''
def to_decoder_input(PAD_ID, GO_ID, decoder_len, idx):
	return [GO_ID] + idx + [PAD_ID] * (decoder_len - len(idx) - 1)

############################################################
'''
	unit tests
'''
def get_batch_unit_test():

	buckets  = [(3,3), (6,6)]

	data_set = ([([1, 1], [4, 3])                    # (3,3) bucket
		        ,([3, 3], [4]   )
		        ,([5]   , [6]   )
		        ],
	            [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2])  # (6,6) bucket
	            ,([3, 3, 3]      , [5, 6]         )
	            ])

	batch_size = 2
	PAD_ID     = 0
	GO_ID      = 2

	encoder_inputs, decoder_inputs, mask = get_batch(data_set, buckets, batch_size, PAD_ID, GO_ID)




