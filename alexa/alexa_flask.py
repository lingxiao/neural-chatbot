"""
ALEXA FLASK FOR TWITTER BOT

Author: Heejin Chloe Jeong
Affiliation: University of Pennsylvania

Description:
In order to use, you need to provide paths to data_dir and train_dir, and in those directories, you should have "w2idx", "idx2w" for data and checkpoint files.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from random import randint
from flask import Flask, render_template
from flask_ask import Ask, statement, question, session
import tensorflow as tf

import math
import os
import random
import sys
import time
import pickle
sys.path.insert(0, '/Users/heejinjeong/Desktop/Develop/Alexa/neural-chatbot/drl_dialog')
import util
import seq2seq_model_rl as s2s_mdl
import mutual_info_model_new as mi_mdl
#import pdb

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

app = Flask(__name__)
ask = Ask(app, "/")
logging.getLogger("flask_ask").setLevel(logging.DEBUG)

PARAM = {"learning_rate": 0.5, "learning_rate_decay_factor": 0.99, "max_gradient_norm": 5.0, "batch_size": 64, "layer_size": 512, "num_layers": 3,
"vocab_size": 50005, "max_train_data_size": 0, "steps_per_checkpoint": 200, "decode": True, "max_utterance_len":80, "backward": False}

data_dir = "/Users/heejinjeong/Dropbox/dialog-class/neural-chatbot/neural-chatbot-inputs/movie/sess-idx"
train_dir = "/Users/heejinjeong/Desktop/Develop/Alexa/ckpt_movie_seq2seq/042217"
#train_dir = "/Users/heejinjeong/Desktop/Develop/Alexa/ckpt_movie_seq2seq/back_043017"
if PARAM["backward"]:
	_buckets = [(40,40),(80,80)]
	
else:
	_buckets = [(80,40),(160,80)]
	

BOC = [[util.GO_ID]]

def create_model(session, train_dir, forward_only):
	"""Create translation model and initialize or load parameters in session.
	If there is any problem on checkpoint, see "tensorflow/contrib/framework/python/framework/checkpoint_util.py"

	"""
	if PARAM["backward"]:
		model = s2s_mdl.Seq2SeqModel(
		PARAM["vocab_size"], _buckets,
		PARAM["layer_size"], PARAM["num_layers"], PARAM["max_gradient_norm"], PARAM["batch_size"],
		PARAM["learning_rate"], PARAM["learning_rate_decay_factor"], forward_only=forward_only)
	else:
		with tf.variable_scope("forward"):
			model = s2s_mdl.Seq2SeqModel(
		PARAM["vocab_size"], _buckets,
		PARAM["layer_size"], PARAM["num_layers"], PARAM["max_gradient_norm"], PARAM["batch_size"],
		PARAM["learning_rate"], PARAM["learning_rate_decay_factor"], forward_only=forward_only)

	ckpt = tf.train.get_checkpoint_state(train_dir)
	print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
	model.saver.restore(session, ckpt.model_checkpoint_path)

	return model

def decode():
	global sess, model, w2idx, idx2w, prev_ids
	#msg = render_template('start')
	sess = tf.Session() 
	# Create model and load parameters.
	model = create_model(sess, train_dir, True)
	model.batch_size = 1  # We decode one sentence at a time.

	w2idx =  pickle.load(open(os.path.join(data_dir,"w2idx.pkl"),"rb"))
	idx2w =  pickle.load(open(os.path.join(data_dir,"idx2w.pkl"),"rb"))
	idx2w[util.EOS_ID] = '<eos>'
	w2idx['<eos>'] = util.EOS_ID
	if PARAM["backward"]:
		prev_ids = []
	else:
		prev_ids = BOC[0]

	print("Finished Starting")


@ask.launch
def get_welcome_response():
	decode()
	welcome_msg = render_template('welcome')
	return statement(welcome_msg)

#@ask.intent("YesIntent")
#def start():	
#	return statement(msg)

@ask.intent("Context", convert={"echo": str})
def seq2seq(echo):
	print(echo)
	sentence = echo
	if not(sess) or not(model):
		return statement(render_template('notyet'))
	# Get token-ids for the input sentence.
	token_ids = util.sentence_to_token_ids(tf.compat.as_bytes(sentence), w2idx, idx2w)
	# Which bucket does it belong to?
	bucket_id = min([b for b in xrange(len(_buckets)) if _buckets[b][0] > len(token_ids)])
	# Get a 1-element batch to feed the sentence to the model.
	encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [([prev_ids]+token_ids, [])]}, bucket_id)
	# Get output logits for the sentence.
	_, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,target_weights, bucket_id, True)
	# This is a greedy decoder - outputs are just argmaxes of output_logits.
	outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
	# If there is an EOS symbol in outputs, cut them at that point.
	if util.EOS_ID in outputs:
		outputs = outputs[:outputs.index(util.EOS_ID)]
	response = " ".join([tf.compat.as_str(idx2w[output]) for output in outputs])
	print(response)
	if PARAM["backward"]:
		prev_ids = []
	else:
		prev_ids = outputs
	return statement(response)


if __name__ == '__main__':	
	app.run(debug=True)






