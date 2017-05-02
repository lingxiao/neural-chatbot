"""
Author: Heejin Chloe Jeong
Affiliation: University of Pennsylvania

Util for DRL - Seq2Seq model for version 1.0 
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import tensorflow as tf

_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

"""
>>> idx2w[0]
'_'
>>> idx2w[1]
'<unk>'
>>> idx2w[2]
'<go>'
>>> idx2w[3]
'</s>'
>>> idx2w[4]
'</c>'

"""
PAD_ID = 0
UNK_ID = 1
GO_ID = 2
EOS_ID = 3
EOC_ID = 4
BOC = [[GO_ID]] # Beginning of Conversation Vector!

removing_ids = [EOS_ID,EOC_ID]#,94,95,1440,143,6772,45094] 
"""
>>> w2idx['<']
94
>>> w2idx['>']
95
>>> w2idx['i']
6
>>> w2idx['@']
1440
>>> w2idx['/']
143
>>> w2idx['\xc2\xa4']
6772
>>> w2idx['\xc2\xa4i']
45094
"""

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

def basic_tokenizer(sentence):
	""" Very basic tokenizer:
	"""
	words = []
	for space_separated_fragment in sentence.strip().split():
		words.extend(_WORD_SPLIT.split(space_separated_fragment))
	return [w for w in words if w]


def sentence_to_token_ids(sentence, vocab, tokenizer = None, normalize_digits = True):

	"""Convert a string to a list of integers representing token-ids.

	"""

	if tokenizer:
		words = tokenizer(sentence)
	else:
		words = basic_tokenizer(sentence)

	if not normalize_digits:
		return [vocab.get(w,UNK_ID) for w in words]

	# Normalize digits by 0 before looking words up in the vocab
	return [vocab.get(_DIGIT_RE.sub(b"0",w), UNK_ID) for w in words]

def refine_words(sentence_ids, w2idx, idx2w):
	negatives = ['haven','hasn','hadn','wouldn','shouldn','mustn','couldn','didn','don','doesn','isn','aren','wasn','weren']
	# there is no am and have
	for i in xrange(len(sentence_ids)-1):
		if sentence_ids[i] == w2idx['i'] and sentence_ids[i+1] == w2idx['m']:
			sentence_ids[i+1] = w2idx['am']
		elif sentence_ids[i] in [w2idx['you'],w2idx['we'],w2idx['they']] and sentence_ids[i+1] == w2idx['re']:
			sentence_ids[i+1] = w2idx['are']
		elif sentence_ids[i] in [w2idx['i'],w2idx['you'],w2idx['we'],w2idx['they']] and sentence_ids[i+1] == w2idx['ve']:
			sentence_ids[i+1] = w2idx['have']
		elif sentence_ids[i] in [w2idx['i'],w2idx['you'], w2idx['he'], w2idx['she'], w2idx['it'], w2idx['we'],w2idx['they']] and sentence_ids[i+1] == w2idx['d']:
			sentence_ids[i+1] = w2idx['would']
		elif sentence_ids[i] ==w2idx['won'] and sentence_ids[i+1] == w2idx['t']:
			sentence_ids[i] == w2idx['will']
			sentence_ids[i+1] = w2idx['not']
		elif sentence_ids[i] ==w2idx['can'] and sentence_ids[i+1] == w2idx['t']:
			sentence_ids[i+1] = w2idx['not']
		elif sentence_ids[i] ==w2idx['what'] and sentence_ids[i+1] == w2idx['s']:
			sentence_ids[i+1] = w2idx['is']
		elif sentence_ids[i] ==w2idx['il'] or sentence_ids[i] ==w2idx['ll'] :
			sentence_ids[i] = w2idx['will']
		elif sentence_ids[i] == w2idx['gonna']:
			sentence_ids = sentence_ids[:i]+[w2idx['going'],w2idx['to']] + sentence_ids[i+1:]
		elif sentence_ids[i] == w2idx['wanna']:
			sentence_ids = sentence_ids[:i]+[w2idx['want'],w2idx['to']] + sentence_ids[i+1:]
		elif idx2w[sentence_ids[i]] in negatives and sentence_ids[i+1] == w2idx['t']:
			sentence_ids[i] = w2idx[idx2w[sentence_ids[i]].split('n')[0]]
			sentence_ids[i+1] = w2idx['not']
	
	return sentence_ids
	"""
	i m = i am
	you/we/they re = you/we/they are
	I/you/we/they ve = I/you/we/they have
	can t = cannot
	il = will
	ll = will
	gonna = going to
	wanna = want to
	haven t = have not
	hadn t = had not
	hasn t = has not
	wouldn t = would not
	shouldn t = should not
	couldn t = could not 
	mustn t = must not
	won t = will not
	didn t = did not
	don t = do not
	i/you/he/she/we/they/it d = i/you/he/she/we/they would (can be 'had' though)
	isn t = is not
	aren t = are not
	weren t = were not
	wasn t = was not
	what s = what is 
	doesn t = does not
	"""








