############################################################
# Module  : Server outputting data
# Date    : Febuary 28th
# Author  : Xiao Ling  
############################################################

import os
import pickle
import numpy as np
from utils import *

############################################################
'''
	Server for project data
	@Use: given 
			* vocab setting of form (ie):
				SETTING = {
						'vocab': {
							'tokens': {
								# special vocab tokens
								  'unk'   : '<unk>'
								, 'pad'   : '_' 
								, 'go'    : '<go>'}
							, 'vocab-size'  : 50
						}}
			* PATH varible with fields:
				- w2idx: path to word to index dictionary
				- idx2w: path to index to word dictionary
				- normalized: path to normalized .txt file
				              containing all converstions

		   constructs a server that serves training
		   and testing batches for validation
'''
class Server:
	
	def __init__( self
				, SETTING
				, w2idx_path
				, idx2w_path
				, normed_dir
				, index_dir
				, log_dir  ):

		writer = Writer(log_dir,1)

		writer.tell('constructing Phone Server object')

		if  os.path.isfile(w2idx_path) \
		and os.path.isfile(idx2w_path) \
		and os.path.exists(normed_dir):

			writer.tell('opening assets from: '  + '\n'
					  + w2idx_path  + '\n'
					  + idx2w_path  + '\n'
					  + normed_dir + '\n')


			w2idx        = pickle.load(open(w2idx_path,'rb'))
			idx2w        = pickle.load(open(idx2w_path,'rb'))
			normed_paths = [os.path.join(normed_dir,p) for p in os.listdir(normed_dir) if '.txt' in p] 

			num       = 1
			sessions  = []
			idx_paths = []

			writer.tell('encoding all .txt sessions')

			for p in normed_paths:

				sess = open(p,'rb').read().split('\n')
				sess = [x.split(': ')[1] for x in sess if len(x.split(': ')) == 2]
				idxs = [[wrd_2_idx(SETTING, w2idx, w) for w in xs.split(' ')] for xs in sess]
				sessions.append(idxs)

				out_p = os.path.join(index_dir, 'session-' + str(num) + '.npy')

				with open(out_p, 'wb') as h:
					np.save(out_p, idxs)

				idx_paths.append(p)

				num += 1


			'''
				storing dicts
			'''
			self.paths = idx_paths
			self.w2idx = w2idx
			self.idx2w = idx2w
			self.sessions = sessions

			writer.tell('initalized!')

		else:
			raise NameError('Failed to initalize server, missing paths')


	'''
		@Use: output all sessions
	'''
	def sessions(self):
		return self.sessions

############################################################
'''
	@Use: given setting and w2idx mapping word to their
	      integer encoding, and a word, output 
	      corresponding index, or 0 if word is OOV
'''
def wrd_2_idx(SETTING, w2idx, word):
	if word in w2idx:
		return w2idx[word]
	else:
		return w2idx[SETTING['tokens']['unk']]



