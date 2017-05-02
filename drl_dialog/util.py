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

