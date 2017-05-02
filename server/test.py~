############################################################
# Module  : hw4 - main
# Date    : March 11th, 2017
# Author  : xiao ling, Heejing Jeong
############################################################

from __future__ import print_function

import os
import nltk
import pickle
import numpy as np

import tensorflow as tf
from tensorflow.contrib import rnn

from prelude   import *
from utils     import *
from hw4       import *

os.system('clear')
'''
	problem: right now the model is 
	kind of opaque to you how it works

	solution: really understand the seq2seq code?

	so you have a q,a pair.
	now you need to encode the q into a hidden
	vector until <EOS>, then generate 
	tokens for response function.

	you need to genenrate each token in response
	conditioned on the previous token, hidden state
	and latent conversation vector c

	so instead of :

	Pr[r_1, .. r_t | h_q]	

	we have:

	Pr[r_1, ... r_t, | h_q, c]

	where c is updated after each exchange
	according to some function

	so each batch should present a n-round
	consecutive conversation snippet sampled
	at random.

	this batch should be encode already
	maybe you can encode this live.
'''
############################################################
'''
	Project config and paths
'''

SETTING = {

			# vocab parameters
		  'unk'        : '<unk>'
		, 'pad'        : '_'
		, 'vocab-size' : 5000

			# maximum question and answer lengths
			# note this is set very high because
			# we are trying to learn sequential
			# dependence in conversation snippets,
			# so no intermediate responses can be
			# removed

        , 'maxq' : 400
        , 'maxr' : 400}

root      = os.getcwd()
input_dir = os.path.join(root, 'data/hw4/input')

PATH = {'raw-dir'    : os.path.join(root, 'data/phone-home')
       , 'w2idx'     : os.path.join(input_dir, 'w2idx.pkl' )
	   , 'idx2w'     : os.path.join(input_dir, 'idx2w.pkl' )
	   , 'normalized': os.path.join(input_dir, 'normalized.txt')}

############################################################
'''
	normalizing text and make batcher
'''
# w2idx, idx2w, normed = preprocessing_convos(SETTING, PATH)
phone = Phone(SETTING, PATH)

q,r = phone.next_train_batch(1)[0]
a,b = phone.next_test_batch(1)[0]

m = phone

ws = 'hello world'
ks = m.word_to_index('question', ws)
bs = m.index_to_hot(ks)

ks1 = m.hot_to_index(bs)
ws1 = m.index_to_word(ks1)

bs1 = m.word_to_hot('question', ws)
ws2 = m.hot_to_word(bs1)






















