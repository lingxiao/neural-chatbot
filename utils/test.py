############################################################
# Module  : for heejin to read and understand server api
# Date    : March 11th, 2017
# Author  : xiao ling, Heejing Jeong
############################################################

from __future__ import print_function

import os
import nltk
import pickle
import numpy as np
from utils     import *

os.system('clear')

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
# phone = Phone(SETTING, PATH)

# get training data
questions,responses = phone.get_train()

# decoding training data
# pick out one sentece
sentence_index = questions[0]
sentence_word  = phone.index_to_word(sentence_index)

# encode sentence to index
sentence_index_2 = phone.word_to_index('question', sentence_word)


















