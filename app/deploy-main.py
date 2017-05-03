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

suffix = ''
raw_root        = config.get_path( 'movie/raw' )
out_root        = config.get_path( 'movie/sess-normed' + suffix )
input_dir       = config.get_path('movie/sess-normed' + suffix)	
sess_idx_dir    = config.get_path('movie/sess-idx'    + suffix)
sess_concat_dir = config.get_path('movie/sess-concat' + suffix)
w2idx_path      = config.get_path('movie/w2idx'       + suffix)
idx2w_path      = config.get_path('movie/idx2w'       + suffix)
max_len         = 400
vocab_size      = 50000

def norm_and_index( RESERVED_TOKENS, input_dir, max_len = 100, vocab_size = 50000):

	print('\n\t>> running normalize_and_index')
	print('\n\t>> opening preprocessed sessions and normalizing ...')

	writer = Writer(config.get_path('log'), 1)

	sess_long = normalize_all( input_dir
	                         , RESERVED_TOKENS
	                         , writer)

	print('\n\t>> found ' + str(len(sess_long)) + ' total sessions')

	print('\n\t>> removing sessions that do not conform to maximum utterance length of ' + str(max_len))

	sessions = []
	rmv      = 0

	for sess in sess_long:

	    if any(len(xs.split(' ')) > max_len for _,xs in sess):
	        print('\n\t>> removing long session')
	        rmv += 1
	    else:
	        sessions.append(sess)     

	print('\n\t>> removed ' + str(rmv) + ' sessions')

	'''
	    construct tokens for word to index
	'''
	print('\n\t>> building idx2w w2idx dictionary ...')

	all_tokens = ' '.join(xs for _,xs in join(sessions))

	w2idx, idx2w  = index(all_tokens, RESERVED_TOKENS, vocab_size)

	return w2idx, idx2w, sessions

def encode_delimited( RESERVED_TOKENS, w2idx, idx2w ):

    print('\n\t>> constructing encoded sessions ...')
    '''
        construct sessions encoded according to w2idx

        make big concactenated version and split
        into 80 percent train and 20 percent validate

        construct index version of all sessions
        movie conversation:
    '''
    sessions_idx = []

    for session in sessions:

        sess_idx = []

        for _, utter in session:
            idx = encode( w2idx, RESERVED_TOKENS['unk'], utter)
            sessions_idx.append(idx)

    return sessions_idx

def construct_session(sess, w2idx, RESERVED_TOKENS):

	out = []

	for s in sess:
		out += s 
		out += [w2idx[RESERVED_TOKENS['eos']]]

	out += [w2idx[RESERVED_TOKENS['eoc']]]

	return out


if False:
	preprocess_movie(raw_root = raw_root, out_root = out_root)

w2idx, idx2w, sessions = norm_and_index(RESERVED_TOKENS, input_dir, max_len = 400, vocab_size = vocab_size )

sessions_idx = encode_delimited(RESERVED_TOKENS, w2idx, idx2w)    
sessions_idx = [construct_session(s, w2idx, RESERVED_TOKENS) for s in chunks( sessions_idx, 4 )]

print('\n\t>> constructing a version containing the concactenation of all session-indices')
cut        = int(float(len(sessions_idx))*0.8)
train_idxs = sessions_idx[ 0:cut ]
test_idxs  = sessions_idx[ cut:  ]

'''
    make big concactenated version and split
    into 80 percent train and 20 percent validate
'''
print('\n\t>> constructing a version containing the concactenation of all sessions')
train = sessions[0:cut]
test  = sessions[cut:]

'''
    remove existing files
'''
print('\n\t>> removing existing .npy files')
shutil.rmtree(sess_idx_dir)
os.mkdir     (sess_idx_dir)


'''
    save output
'''
print('\n\t>> saving all ' + str(len(sessions)) + ' normalized sessions in .txt and .npy form ...')

num = 1

for sess_norm, sess_idx in zip(sessions, sessions_idx):

    name       = 'sess-' + str(num)
    out_normed = os.path.join(input_dir    , name + '.txt')
    out_idx    = os.path.join(sess_idx_dir , name + '.npy')

    with open(out_normed, 'w') as h:
        for t,xs in sess_norm:
            h.write(t + ': ' + xs + '\n')

    with open(out_idx, 'wb') as h:
        np.save(h, np.asarray(sess_idx))

    num +=1 

'''
   save concactenated version of all sessions
   as pkl file
'''     
print('\n\t>> saving all concactenated files')       

train_path = os.path.join(sess_concat_dir, 'train')
valid_path = os.path.join(sess_concat_dir, 'valid')

# with open(train_path + '.txt','w') as h:
#     for xs in join(train):
#         h.write(xs + '\n')

# with open(valid_path + '.txt','w') as h:
#     for xs in join(test):
#         h.write(xs + '\n')

with open(train_path + '.pkl','wb') as h:
    pickle.dump(train_idxs,h)

with open(valid_path + '.pkl','wb') as h:
    pickle.dump(test_idxs,h)

'''
    save w2idx and idx2w
'''
print('\n\t>> saving w2idx and idx2w')

with open(w2idx_path, 'wb') as h:
    pickle.dump(w2idx, h)

with open(idx2w_path, 'wb') as h:
    pickle.dump(idx2w, h)





