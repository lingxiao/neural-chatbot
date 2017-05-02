############################################################
# Module  : main for deploying code
# Date    : March 11th, 2017
# Author  : xiao ling, Heejing Jeong
############################################################

from __future__ import print_function

import os
import re
import nltk
import pickle
import numpy as np
import zipfile
import gzip

from utils import *
from server import *
from app import *


############################################################
'''
	@Use: preprocess phone data
'''
def main_phone():

	'''
		corpus vocab size is net of reserved tokens
		see app/config.py for details
	'''
	vocab_size = config.RESERVED_TOKENS['vocab-size'] \
	             - len(config.RESERVED_TOKENS)

	preprocess_phone( input_dir = config.get_path('phone/raw')
		            , out_dir   = config.get_path('phone/sess-normed'))


	'''
		preprocess all raw phone converastions
		for details, see utils/preprocess.py
	'''
	normalize_and_index( RESERVED_TOKENS
                        , config.get_path('phone/sess-normed')
                        , config.get_path('phone/sess-idx')
                        , config.get_path('phone/sess-concat')
                        , config.get_path('phone/w2idx')
                        , config.get_path('phone/idx2w')
                        , max_len    = 400
                        , vocab_size = 8036)

'''
	@Use: preprocess movie data
'''
def main_movie(dev = False):

	if dev: 
		suffix = '-dev'
	else:
		suffix = ''

	print('\n>> Running main_movie in '+ suffix + ' mode ...')

	zip_root = config.get_path( 'movie/zip' )
	raw_root = config.get_path( 'movie/raw' )
	out_root = config.get_path( 'movie/sess-normed' + suffix )

	# extract_movie   (zip_root = zip_root, raw_root = raw_root)
	preprocess_movie(raw_root = raw_root, out_root = out_root)

	normalize_and_index( RESERVED_TOKENS
	                    , config.get_path('movie/sess-normed' + suffix)
	                    , config.get_path('movie/sess-idx'    + suffix)
	                    , config.get_path('movie/sess-concat' + suffix)
	                    , config.get_path('movie/w2idx'       + suffix)
	                    , config.get_path('movie/idx2w'       + suffix)
	                    , max_len    = 400
	                    , vocab_size = 50000)

main_movie()



















