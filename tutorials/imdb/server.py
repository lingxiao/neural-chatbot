############################################################
# Module  : IMDB class to output training and test data
# Date    : March 15th, 2017
# Author  : xiao ling
############################################################

from __future__ import print_function

import os
import nltk
import pickle
import random
import numpy as np

from prelude import *
from utils import *


############################################################
'''
	imdb class
'''
class Imdb:

	def __init__(self, SETTING, data_dir, out_dir):

		self.setting = SETTING
		self.dirs    = {'data': data_dir, 'output': out_dir}

		'''
			open preprocessd files and w2idx dictionary
		'''
		train_pos_path = os.path.join(out_dir, 'train-pos.txt' )
		train_neg_path = os.path.join(out_dir, 'train-neg.txt' )
		test_pos_path  = os.path.join(out_dir, 'test-pos.txt'  )
		test_neg_path  = os.path.join(out_dir, 'test-neg.txt'  )
		w2idx_path     = os.path.join(out_dir, 'imdb-w2idx.pkl')

		os.system('clear')
		
		if  os.path.isfile(train_pos_path) \
		and os.path.isfile(train_neg_path) \
		and os.path.isfile(test_pos_path ) \
		and os.path.isfile(test_neg_path ):

			print ('\n>> opening assets from ' + out_dir)

			eop = SETTING['End-of-Paragraph']

			train_pos  = open(train_pos_path, 'r').read().split(eop)
			train_neg  = open(train_neg_path, 'r').read().split(eop)
			test_pos   = open(test_pos_path , 'r').read().split(eop)
			test_neg   = open(test_neg_path , 'r').read().split(eop)
			w2idx_r    = open(w2idx_path, 'rb')

			w2idx      = pickle.load(w2idx_r)
			idx2w      = dict((idx,w) for w,idx in w2idx.iteritems())

			print ('\n>> pruning data for those that are longer than max-length' 
				  + ' and shorter than min-length')

			print('\n>> encoding training and test data')

			train_pos = [(e,[1]) for e in [encode(SETTING, w2idx, s) for s in train_pos] if e]
			train_neg = [(e,[0]) for e in [encode(SETTING, w2idx, s) for s in train_neg] if e]
			test_pos  = [(e,[1]) for e in [encode(SETTING, w2idx, s) for s in test_pos ] if e]
			test_neg  = [(e,[0]) for e in [encode(SETTING, w2idx, s) for s in test_neg ] if e]

			print('\n>> there are ' + str(len(train_pos)) + ' positive training reviews conforming to length')
			print('\n>> there are ' + str(len(train_neg)) + ' negative training reviews conforming to length')
			print('\n>> there are ' + str(len(test_pos))  + ' positive test reviews conforming to length'    )
			print('\n>> there are ' + str(len(test_neg))  + ' negative test reviews conforming to length'    )

			print('\n>> preparing the batches')

			trains = train_pos + train_neg
			tests  = test_pos  + test_neg

			for _ in range(10):
				random.shuffle(trains)
				random.shuffle(tests )

			self.train = trains
			self.test  = tests

			self.w2idx     = w2idx
			self.idx2w     = idx2w

			'''
				set up internal batch counter
			'''
			self.train_batch = 0
			self.test_batch  = 0

		else:
			print('\n>> preparing files from ' + data_dir)
			preprocess_imdb(SETTING, data_dir, out_dir)
			return Imdb(SETTING, data_dir, out_dir)

	'''
		@Use: Given a list of words, encode into indices
	'''
	# from_words :: [String] -> [Int]
	def from_words(self, words):
		return [encode(self.setting, self.w2idx, w) for w in words]

		'''	
		@Use: given a list of indices, decode into words
	'''
	# to_words :: [Int] -> [String]
	def to_words(self,idxs):
		return [self.idx2w[i] for i in idxs]

	'''
		@Use: Given batch_size, get next batch 
	     	  of training data
	     	  if we ran out of data, reshuffle
	     	  and start again

	     	  if one_hot flag is one, then output one hot encoding
	     	  else output integers
	'''
	# train_next_batch :: Int -> Bool -> ([[Int]],[[Int]])
	def train_next_batch(self, batch_size, one_hot=True):

		train = self.train
		b     = self.train_batch

		if b >= len(train):

			self.train_batch = 0
			print ('\n>> used all data, reshuffling data')
			for _ in range(10):
				random.shuffle(self.train)
			return self.train_next_batch(batch_size)

		else:
			print ('\n>> getting next training batch of ' + str(batch_size))

			xs = train[b:b + batch_size]
			self.train_batch += batch_size

			if one_hot:
				hots = [(to_one_hot(self.setting['VOCAB_SIZE'], x), to_one_hot(self.setting['num_classes'], y)) for x,y in xs]
				return np.asarray([x for x,_ in hots]), np.asarray([y for _,y in hots])
			else:
				return xs


	'''
		@Use: Given batch_size, get next batch 
	     	  of test data
	     	  if we ran out of data, reshuffle
	     	  and start again
	'''
	# test_next_batch :: Int -> ([[Int]],[[Int]])
	def test_next_batch(self, batch_size):

		test = self.test
		b    = self.test_batch

		if b >= len(test):

			print ('\n>> used all data, reshuffling data')
			self.test_batch = 0
			for _ in range(10):
				random.shuffle(self.test)
			return self.test_next_batch(batch_size)

		else:
			print ('\n>> getting next test batch of ' + str(batch_size))

			xs = test[b:b + batch_size]
			self.test_batch += batch_size

			if one_hot:
				hots = [(to_one_hot(self.setting['VOCAB_SIZE'], x), to_one_hot(self.setting['num_classes'], y)) for x,y in xs]
				return np.asarray([x for x,_ in hots]), np.asarray([y for _,y in hots])
			else:
				return xs

	'''
		@Use: output all test data as np.array of examples
		      and list of labels
	'''
	# test :: (np.array (np.array Int), np.array (np.array Int))
	def get_test(self):
		ts   = self.test
		hots = [(to_one_hot(self.setting['VOCAB_SIZE'], x), to_one_hot(self.setting['num_classes'], y)) for x,y in ts]
		return np.asarray([x for x,_ in hots]), np.asarray([y for _,y in hots])

############################################################
'''
	@Use: given setting and w2idx mapping word to their
	      integer encoding, and a word, output 
	      corresponding index, or 0 if word is OOV
'''
def word_to_index(SETTING, w2idx, word):
	if word in w2idx:
		return w2idx[word]
	else:
		return w2idx[SETTING['UNK']]


'''
	@Use: Given a normalized review of type String
	      output one hot and padded encoding of review
'''
# encode :: Dict String String
#        -> Dict String Int -> String
#        -> Either Bool [[Int]]
def encode(SETTING, w2idx, review):

	tokens  = review.split(' ')

	if  len(tokens) > SETTING['max-length'] \
	or  len(tokens) < SETTING['min-length']:
		return False
	else:
		pads    = SETTING['max-length'] - len(tokens)
		tokens  = tokens + [SETTING['PAD']]*pads
		idxs    = [word_to_index(SETTING, w2idx, t) for t in tokens]
		# one_hot = to_one_hot(SETTING, idxs)
		return idxs
'''
	@Use: given dimension of one hot vector `depth`
	      and a list of `idxs`, each index corresponding
	      to the value that should be hot in the one-hot vector
	      output depth x len(idxs) matrix 
'''
# to_one_hot :: Int -> [Int] -> np.ndarray (np.ndarray Int)
def to_one_hot(depth,idxs):

	hots = []

	for i in idxs:
		col    = [0] * depth
		col[i] = 1
		hots.append(col)

	if len(hots) == 1:
		hots = hots[0]
	# return np.asarray(hots)
	return np.ndarray.transpose(np.asarray(hots))

############################################################
'''	
	Top level preprocssing imdb function:

	@Use: Given:
			- path to imdb data directory
			- path to output directory 
			- Setting with key:
				'UNK' denoting symbol for OOV word
				'VOCAB_SIZE' denoting number of allowed words

			open positive and negative files from both train
			and test, normalize the text and construct 
			word to index dictionary
'''
# preprocess_imdb :: String -> String 
#            -> Dict String _ 
#            -> IO (Dict String Int)
def preprocess_imdb(SETTING, data_dir, out_dir):

	train   = get_data(os.path.join(data_dir, 'train'))
	test    = get_data(os.path.join(data_dir, 'test' ))

	train_pos = train['positive']
	train_neg = train['negative'] 
	test_pos  = test ['positive']
	test_neg  = test ['negative'] 

	'''
		normalizing text
	'''
	print ('\n>> normalizing training text ...')
	train_pos = [normalize(xs) for xs in train_pos]
	train_neg = [normalize(xs) for xs in train_neg]

	print ('\n>> normalizing test text ...')
	test_pos  = [normalize(xs) for xs in test_pos]
	test_neg  = [normalize(xs) for xs in test_neg]

	'''
		construct tokens for word to index
	'''
	tokens = ' '.join([
		  ' '.join(train_pos)
		, ' '.join(train_neg)
		, ' '.join(test_pos )
		, ' '.join(test_neg )
		])


	idx2w, w2idx, dist = index(tokens, SETTING)

	'''
		save results of normalization delimited 
		by end of paragraph `<EOP>` token
	'''
	print('\n>> saving all results ...')

	train_pos_path = os.path.join(out_dir, 'train-pos.txt')
	train_neg_path = os.path.join(out_dir, 'train-neg.txt')
	test_pos_path  = os.path.join(out_dir, 'test-pos.txt')
	test_neg_path  = os.path.join(out_dir, 'test-neg.txt')
	w2idx_path     = os.path.join(out_dir, 'imdb-w2idx.pkl')

	eop = SETTING['End-of-Paragraph']

	with open(train_pos_path, 'w') as h:
		h.write(eop.join(train_pos))

	with open(train_neg_path, 'w') as h:
		h.write(eop.join(train_neg))

	with open(test_pos_path, 'w') as h:
		h.write(eop.join(test_pos))

	with open(test_neg_path, 'w') as h:
		h.write(eop.join(test_neg))

	with open(w2idx_path, 'wb') as h:
		pickle.dump(w2idx, h)

	return w2idx

############################################################
'''
	@Use: given `data_dir`, open positive and negative
	      examples, concactente them into their respective
	      giant strings and output as a dictionary
'''
# get_data :: String -> Dict String String
def get_data(data_dir):

	print ('\n>> getting data from ' + data_dir)

	# path to positive and negative examples
	train_pos = os.path.join(data_dir, 'pos')
	train_neg = os.path.join(data_dir, 'neg')

	poss = [os.path.join(train_pos, p) for p in os.listdir(train_pos)]
	negs = [os.path.join(train_neg, p) for p in os.listdir(train_neg)]

	'''
		todo: you need to save the preprocessed stuff
		or present it as some mini batch suitable format
	'''	
	# open all files as one long string
	pos_toks = [open(p,'r').read() for p in poss]
	neg_toks = [open(p,'r').read() for p in negs]
	# pos_toks = ' '.join([open(p,'r').read() for p in poss])
	# neg_toks = ' '.join([open(p,'r').read() for p in negs])

	return {'positive': pos_toks, 'negative': neg_toks}

'''
	@Use: Given a string, tokenize by:
			- casefolding
			- whitespace stripping
			- elminate puncutation
'''
# normalize :: String -> String
def normalize(xs):
	tok = Tokenizer(casefold=True, elim_punct=True)
	ys  = xs.decode('utf-8')
	ys  = tok.tokenize(ys)
	ys  = ' '.join(ys)
	ys  = ys.encode('utf-8')
	return ys

'''
	@Use: Given a string, build dictionary mapping unique
		  tokens in string to their frequency
'''
# build_dict :: String -> Dict String Int
def build_dict(xs):

	print ('\n>> building dictionary ...')
	ws  = xs.split(' ')
	dic = dict.fromkeys(set(ws))

	for w in ws:
		if not dic[w]: dic[w]  = 1
		else         : dic[w] += 1

	return dic

'''
 read list of words, create index to word,
  word to index dictionaries
    return tuple( vocab->(word, count), idx2w, w2idx )

'''
# index :: String 
#       -> Dict String Int 
#       -> ([String], Dict String Int, nltk.Probability.FreqDist)
def index(tokenized_sentences, SETTING):
	print ('\n>> building idx2w w2idx dictionary ...')

	tokenized_sentences = [[w] for w in tokenized_sentences.split(' ')]

	# get frequency distribution
	freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
	# get vocabulary of 'vocab_size' most used words
	vocab = freq_dist.most_common(SETTING['VOCAB_SIZE'])
	# index2word
	index2word = [SETTING['PAD']]        \
	           + [SETTING['UNK']]        \
	           + [ x[0] for x in vocab ] \
	# word2index
	word2index = dict([(w,i) for i,w in enumerate(index2word)] )
	return index2word, word2index, freq_dist
















