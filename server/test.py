############################################################
# Module  : hw4 - main
# Date    : March 11th, 2017
# Author  : xiao ling, Heejing Jeong
############################################################

import os
from utils  import *
from server import *
from app.config import PATH

os.system('clear')

############################################################
'''
	Project config and paths
'''
SETTING = {'tokens': {
				# special vocab tokens
				  'unk'   : '<unk>'
				, 'pad'   : '_' 
				, 'go'    : '<go>'}
			, 'vocab-size'  : 50000
	        , 'maxq' : 10
	        , 'maxr' : 10
		   }

############################################################

'''
	preprocess all raw converastions
	for details, see utils/preprocess.py
'''
normed_path = os.path.join(PATH['directories']['normed'], 'normalized.txt')

if True:
	w2idx, idx2w, _ = preprocessing_convos(
			          SETTING
	                , PATH['directories']['raw']
	                , normed_path
	                , PATH['directories']['log']
	                , PATH['file-paths']['w2idx']
	                , PATH['file-paths']['idx2w'])

############################################################
'''
	normalizing text and make batcher
'''
server = Phone(SETTING
	        , PATH['file-paths']['w2idx']
	        , PATH['file-paths']['idx2w']
	        , normed_path)

q,r = server.next_train_batch(1,True)
a,b = server.next_test_batch(1)

ws = 'hello world'
ks = server.word_to_index('question', ws)
bs = server.index_to_hot(ks)

ks1 = server.hot_to_index(bs)
ws1 = server.index_to_word(ks1)

bs1 = server.word_to_hot('question', ws)
ws2 = server.hot_to_word(bs1)






















