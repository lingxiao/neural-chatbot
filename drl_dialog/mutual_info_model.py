"""
Author: Heejin Chloe Jeong
Affiliation: University of Pennsylvania

DRL - Mutual Information model for version 1.0 
For model descriptions, see "Deep Reinforcement Learning for Dialogue Generation", Jiwei Li et al.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
from six.moves import xrange
import tensorflow as tf
import pdb
import util

#assert tf.__version__ == '1.0.1'

from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
from tensorflow.contrib.rnn.python.ops import core_rnn_cell

class Mutual_Info(object):
	def __init__(self,pre_trained_seq2seq, pre_trained_backward, vocab_size, buckets, layer_size, num_layers, 
		max_gradient_norm, candidate_size, learning_rate, learning_rate_decay_factor, 
		use_lstm=False, num_samples=512, forward_only = False, dtype= tf.float32):

	"""Create a Model:
		Similar to the seq2seq_model_rl.py code but it has differences in:
		- loss function
		-
		INPUTS:
			vocab_size: size of vocabulary
			buckets: a list of pairs (I,O), where I specifies maximum input length that 
				will be processed in that bucket, and O specifies maximum output length. Traning 
				instances that have inputs longer than I or outputs longer than O will be pushed 
				to the next bucket and padded accordingly. We assume that the list is sorted.
				** We may not use bucketing for Dialogue.
			layer_size: the number of units in each layer
			num_layers: the number of the layers in the model
			max_gradient_norm : gradients will be clipped to maximally this norm?
			candidate_size : the number of candidates (actions)
			learning_rate : learning rate to start with.
			learning_rate_decay_factor : decay learning rate by this much when needed.
			use_lstm: True -> LSTM cells, False -> GRU cells
			num_samples: the number of samples for sampled softmax
			forward_only : if set, we do not construct the backward pass in the model
			dtype: the data type to use to store internal variables.
	"""
		self.vocab_size = vocab_size
		self.buckets = buckets
		self.buckets_back = [(x[1],x[1]) for x in buckets]
		self.batch_size = """? necessary?"""
		self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype = dtype)
		self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate*learning_rate_decay_factor)
		self.global_step = tf.Variable(0, trainable=False)
		self.pre_trained_seq2seq = pre_trained_seq2seq
		self.pre_trained_backward = pre_trained_backward

		if num_samples > 0 and num_samples < self.vocab_size:
			w_t = tf.get_variable("proj_w_mi",[self.vocab_size, layer_size], dtype = dtype)
			w = tf.transpose(w_t)
			b = tf.get_variable("proj_b_mi", [self.vocab_size], dtype=dtype)
			output_projection = (w,b)

			def mi_score(state, action):
				"""
				Args:
					state: [first utterance, second utterance]
					action: action utterance
				"""
				bucket_id = min([b for b in xrange(len(self.buckets)) if self.buckets[b][0] > len(state[0]+state[1])])
				encoder_input, decoder_input, target_weight = self.prepare_step(state[0]+state[1],action,bucket_id)				
				_, _, output_logits = self.pre_trained_seq2seq.step(sess, encoder_input, decoder_input, target_weight, bucket_id, True)
				log_prob = sum([output_logits[action[i]] - np.log(sum(np.exp(output_logits[i][0]))) for i in xrange(self.buckets[bucket_id][1])])

				bucket_id_back = min([b for b in xrange(len(self.buckets_back)) if self.buckets_back[b][0] > len(state[1])])
				encoder_input_back, decoder_input_back, target_weight_back = self.prepare_step(action, state[1],bucket_id_back)
				_, _, output_logits_back = self.pre_trained_backward.step(sess, encoder_input_back, decoder_input_back, target_weight_back, bucket_id_back, True)
				log_prob_back = sum([output_logits_back[state[1][i]] - np.log(sum(np.exp(output_logits_back[i][0]))) for i in xrange(self.buckets_back[bucket_id_back][1])])

				return -log_prob/float(len(action)) - log_prob_back/float(len(state[1]))

			loss_function = mi_score	

		if use_lstm:
			single_cell = core_rnn_cell.BasicLSTMCell(layer_size)
		else:
			single_cell = core_rnn_cell.GRUCell(layer_size)

		if num_layers > 1:
			cell = core_rnn_cell.MultiRNNCell([single_cell]*num_layers)
		else:
			cell = single_cell

		def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
			return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(encoder_inputs, decoder_inputs, cell, num_encoder_symbols = vocab_size, num_decoder_symbols=vocab_size, embedding_size = layer_size, output_projection = output_projection, feed_previous = do_decode, dtype = dtype)

		self.state = tf.placeholder(tf.int32, shape=[1,None], name="state") # "encoder{0}".format(N) -> 'encoderN'
		self.bucket_id = tf.placeholder(tf.int32, name ="bucket_id")
		state_encoder, action_decoder_dummy, target_weight_dummy = self.prepare_step(self.state, self.bucket_id)
		self.action = generate_action(state_encoder,action_decoder_dummy, self.buckets, bucket_id, lambda x,y:seq2seq_f(x,y,False))
		self.losses = loss_function(self.state,self.action)

		params = tf.trainable_variables()
		if not forward_only:
			self.updates = []
			opt = tf.train.GradientDescentOptimizier(self.learning_rate)
			for b in xrange(len(buckets)):
				self.updates.append(opt.minimize(self.losses[b],params))

		self.saver = tf.train.Saver(tf.global_variables())

	def step(self,session, encoder_input, bucket_id, forward_only ):

		input_feed = {}
		input_feed[self.state.name] = encoder_input
		input_feed[self.bucket_id.name] = bucket_id

		if not forward_only:
			output_feed = [self.updates[bukcet_id], self.losses[bucket_id]]
		else:
			output_feed = [self.losses[bucket_id]]

		outputs = session.run(output_feed,input_feed)
		if not foward_only:
			return outputs[1]
		else:
			return outputs[0]



	def prepare_step(self, encoder_input, bucket_id):
		"""Proper format for Seq2Seq model without batch method.
		"""

		encoder_size, decoder_size = self.buckets[bucket_id]
		proper_encoder  = list(reversed(encoder_input+[util.PAD_ID]*(encoder_size-len(encoder_input))))
		proper_decoder = [util.GO_ID] + [util.PAD_ID]*(decoder_size-len(decoder_input))
		
		final_encoder = [np.array([x], dtype = np.int32) for x in proper_encoder]
		final_decoder = [np.array([x], dtype = np.int32) for x in proper_decoder]
		target_weights_dummy = []
		for len_idx in xrange(decoder_size):
			target_weights_dummy_tmp = np.ones(1,dtype=np.float32)
			if len_idx < decoder_size -1:
				target = proper_decoder[len_idx+1]
			if len_idx == decoder_size-1 or target == util.PAD_ID:
				target_weights_dummy_tmp[0] = 0.0
			target_weights_dummy.append(target_weights_dummy_tmp)

		return final_encoder, final_decoder, target_weights_dummy 


	def generate_action(self, state_encoder, action_decoder_dummy, buckets, bucket_id, seq2seq):
		all_inputs = state + action_decoder_dummy

		with ops.name_scope(name, "model_with_buckets", all_inputs):
			with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse = True if bucket_id>0 else None):
				outputs, _ = seq2seq(state[:buckets[bucket_id][0]], action_decoder_dummy[:buckets[bucket_id][1]])
					
		action_ids = [int(np.argmax(logit, axis = 1)) for logit in outputs]
		if util.EOS_ID in action_ids:
			action_ids = action_ids[:action_ids.index(util.EOS_ID)]

		return action_ids













	
















	
















	
















	
















	
















	
















