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
from tensorflow.python.ops import variable_scope

from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
from tensorflow.contrib.rnn.python.ops import core_rnn_cell

class Mutual_Info(object):
	def __init__(self, sess, pre_trained_seq2seq, pre_trained_backward, vocab_size, buckets, layer_size, num_layers, 
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
		self.sess = sess
		self.vocab_size = vocab_size
		self.buckets = buckets
		self.buckets_back = [(x[1],x[1]) for x in buckets]
		self.batch_size = """? necessary?"""
		self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype = dtype)
		self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate*learning_rate_decay_factor)
		self.global_step = tf.Variable(0, trainable=False)
		self.pre_trained_seq2seq = pre_trained_seq2seq
		self.pre_trained_backward = pre_trained_backward
		self.bucket_id = len(buckets)-1
		if num_samples > 0 and num_samples < self.vocab_size:
			w_t = tf.get_variable("proj_w_mi",[self.vocab_size, layer_size], dtype = dtype)
			w = tf.transpose(w_t)
			b = tf.get_variable("proj_b_mi", [self.vocab_size], dtype=dtype)
			output_projection = (w,b)

			"""
			def mi_score(states, actions, weights, states_back, actions_back, weights_back, bucket_id):
				#Args:
				#	states:[first utterance, second utterance]
				#	actions: action utterance

				pdb.set_trace()
				#bucket_id = min([b for b in xrange(len(self.buckets)) if self.buckets[b][0] > len(states)])
				states_input = self.sess.run()
				_, _, output_logits = self.pre_trained_seq2seq.step(self.sess, states, actions, weights, bucket_id, True)
				# output_logits: 
				log_prob = []
				for word_idx in xrange(len(actions)):
					tmp = [output_logits[word_idx][batch_idx][actions[word_idx][batch_idx]] - np.log(sum(np.exp(output_logits[word_idx][batch_idx]))) for batch_idx in xrange(batch_size)]
					log_prob.append(np.inner(tmp, weights[word_idx]))

				#bucket_id_back = min([b for b in xrange(len(self.buckets_back)) if self.buckets_back[b][0] > len(states_back)])
				_, _, output_logits_back = self.pre_trained_backward.step(self.sess, actions_back, states_back, weights_back, bucket_id, True)

				log_prob_back = []
				for word_idx in xrange(len(states_back)):
					tmp = [output_logits_back[word_idx][batch_idx][states_back[word_idx][batch_idx]] - np.log(sum(np.exp(output_logits_back[word_idx][batch_idx]))) for batch_idx in xrange(batch_size)]
					log_prob_back.append(np.inner(tmp, weights_back[word_idx]))

				# -log_prob/float(len(action)) - log_prob_back/float(len(state[1]))
				return -sum(log_prob)/float(len(actions)) - log_prob_back/float(len(states_back))

			loss_function = mi_score	
			"""

		if use_lstm:
			single_cell = core_rnn_cell.BasicLSTMCell(layer_size)
		else:
			single_cell = core_rnn_cell.GRUCell(layer_size)

		if num_layers > 1:
			cell = core_rnn_cell.MultiRNNCell([single_cell]*num_layers)
		else:
			cell = single_cell

		def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
			return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(encoder_inputs, decoder_inputs, cell,
			 num_encoder_symbols = vocab_size, num_decoder_symbols=vocab_size, embedding_size = layer_size, 
			 output_projection = output_projection, feed_previous = do_decode, dtype = dtype)
		self.seq2seq_f = seq2seq_f

		self.states, self.states_back = [], []
		self.actions , self.actions_back = [], []
		self.weights, self.weights_back = [], []

		for i in xrange(self.buckets[-1][0]):
			self.states.append(tf.placeholder(tf.int32, shape=[None], name ="state{0}".format(i)))

		for i in xrange(self.buckets_back[-1][1]):
			self.states_back.append(tf.placeholder(tf.int32, shape=[None], name ="state_back{0}".format(i)))

		for i in xrange(self.buckets[-1][1]):
			self.actions.append(tf.placeholder(tf.int32, shape=[None], name ="action{0}".format(i)))
			self.actions_back.append(tf.placeholder(tf.int32, shape=[None], name ="action_back{0}".format(i)))
			self.weights.append(tf.placeholder(dtype, shape=[None], name="weight_rl{0}".format(i)))
			self.weights_back.append(tf.placeholder(dtype, shape=[None], name="weight_rl_back{0}".format(i)))

		#self.losses = loss_function(self.states, self.actions, self.weights, self.states_back, self.actions_back, self.weights_back, self.bucket_id)
		self.losses = []
		for i in xrange(len(buckets)):
			self.losses.append(tf.placeholder(tf.float32, shape = [None], name = "losses{0}".format(i)))

		params = tf.trainable_variables()
		pdb.set_trace()
		if not forward_only:
			self.gradient_norms = []
			self.updates = []
			opt = tf.train.GradientDescentOptimizer(self.learning_rate)
			for b in xrange(len(buckets)):
				gradients = tf.gradients(self.losses[b],params)
				clipped_gradients, global_norm = tf.clip_by_global_norm(gradients, max_gradient_norm) # Clips values of multiple tensors by the ratio of the sum of their norms.
				self.gradient_norms.append(global_norm)
				self.updates.append(opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)) #An Operation that applies the specified gradients. If global_step was not None, that operation also increments global_step.
				#self.updates.append(opt.minimize(self.losses[b],params))

		self.saver = tf.train.Saver(tf.global_variables())

	def mi_score(self,states, actions, weights, states_back, actions_back, weights_back, bucket_id):
				
				"""
				Args:
					states:[first utterance, second utterance]
					actions: action utterance
				"""
				pdb.set_trace()
				#bucket_id = min([b for b in xrange(len(self.buckets)) if self.buckets[b][0] > len(states)])
				states_input = self.sess.run()
				_, _, output_logits = self.pre_trained_seq2seq.step(self.sess, states, actions, weights, bucket_id, True)
				# output_logits: 
				log_prob = []
				for word_idx in xrange(len(actions)):
					tmp = [output_logits[word_idx][batch_idx][actions[word_idx][batch_idx]] - np.log(sum(np.exp(output_logits[word_idx][batch_idx]))) for batch_idx in xrange(batch_size)]
					log_prob.append(np.inner(tmp, weights[word_idx]))

				#bucket_id_back = min([b for b in xrange(len(self.buckets_back)) if self.buckets_back[b][0] > len(states_back)])
				_, _, output_logits_back = self.pre_trained_backward.step(self.sess, actions_back, states_back, weights_back, bucket_id, True)

				log_prob_back = []
				for word_idx in xrange(len(states_back)):
					tmp = [output_logits_back[word_idx][batch_idx][states_back[word_idx][batch_idx]] - np.log(sum(np.exp(output_logits_back[word_idx][batch_idx]))) for batch_idx in xrange(batch_size)]
					log_prob_back.append(np.inner(tmp, weights_back[word_idx]))

				# -log_prob/float(len(action)) - log_prob_back/float(len(state[1]))
				return -sum(log_prob)/float(len(actions)) - log_prob_back/float(len(states_back))

	def step(self, session, states, states_back, actions, actions_back, weights, weights_back, bucket_id, forward_only ):
		# Check if the sizes match.
		state_size, action_size = self.buckets[bucket_id]
		action_back_size, state_back_size = self.buckets_back[bucket_id]

		if len(states) != state_size:
		  raise ValueError("Encoder length must be equal to the one in bucket,"
		                   " %d != %d." % (len(states), state_size))
		if (len(actions) != action_size) or (len(actions) != len(actions_back)):
		  raise ValueError("Decoder length must be equal to the one in bucket,"
		                   " %d != %d." % (len(actions), action_size))
		if len(weights) != action_size:
		  raise ValueError("Weights length must be equal to the one in bucket,"
		                   " %d != %d." % (len(weights), action_size))
		if len(states_back) != state_back_size:
		  raise ValueError("Encoder length must be equal to the one in bucket,"
		                   " %d != %d." % (len(states_back), state_back_size))
		if len(weights_back) != state_back_size:
		  raise ValueError("Weights length must be equal to the one in bucket,"
		                   " %d != %d." % (len(weights_back), state_back_size))

		losses = self.mi_score(states, actions, weights, states_backactions_back, weights_back, bucket_id)

		input_feed = {}
		for l in xrange(len(self.buckets)):
			input_feed[self.losses[l].name] = losses[l]

		for l in xrange(state_size):
			input_feed[self.states[l].name] = states[l]

		for l in xrange(action_size):
			input_feed[self.actions[l].name] = actions[l]
			input_feed[self.weights[l].name] = weights[l]
			input_feed[self.actions_back[l].name] = actions_back[l]

		for l in xrange(state_back_size):
			input_feed[self.states_back[l].name] = states_back[l]
			input_feed[self.weights_back[l].name] = weights_back[l]

		if not forward_only:
			output_feed = [self.updates[bucket_id], self.losses[bucket_id]]
		else:
			output_feed = [self.losses[bucket_id]]

		outputs = session.run(output_feed,input_feed)
		if not foward_only:
			return outputs[1]
		else:
			return outputs[0]

	def get_batch(self, data, bucket_id):
		"""Proper format for Seq2Seq model.
		"""

		state_size, action_size = self.buckets[bucket_id]
		states, states_back, action_dums = [],[]
		batch_states, batch_states_back, batch_actions_dums, batch_weights_back = [], [], [], []

		for _ in xrange(self.batch_size):
			state_i01, state_i02 = random.choice(data[bucket_id])
			state_i = state_i01 + state_i02 + [util.PAD_ID]*(state_size - len(state_i01) - len(state_i02))
			state_j = [util.GO_ID] + statei02 + [util.PAD_ID]*(state_size/2 - len(statei02)-1)
			states.append(list(reversed(state_i)))
			states_back.append(state_j) # no reversed! 

			action_i_dum = [util.GO_ID] + [util.PAD_ID]*(action_size-1)
			action_dums.append(action_i_dum)
			

		for length_idx in xrange(state_size):
			batch_states.append(np.array([states[batch_idx][length_idx] for batch_idx in xrange(self.batch_size)], dtype =np.int32))

		for length_idx in xrange(state_size/2):
			batch_states_back.append(np.array([states_back[batch_idx][length_idx] for batch_idx in xrange(self.batch_size)], dtype = np.int32))
			w = np.ones(self.batch_size, dtype = np.float32)
			for batch_idx in xrange(self.batch_size):
				# The corresponding target is decoder_input shifted by 1 forward.
				if length_idx < (state_size/2) - 1:
					target = states_back[batch_idx][length_idx+1]
				if length_idx == (state_size/2) - 1 or target == util.PAD_ID:
					w[batch_idx] = 0.0
			batch_weights_back.append(w)

		for length_idx in xrange(action_size):
			batch_actions_dums.append(np.array([action_dums[batch_idx][length_idx] for batch_idx in xrange(self.batch_size)], dtype = np.int32))

		batch_actions, batch_actions_back, batch_weights = self.generate_batch_action(batch_states, batch_actions_dums, bucket_id, self.seq2seq_f)	

		return batch_states, batch_states_back, batch_actions, batch_actions_back, batch_weights, batch_weights_back

	def generate_batch_action(self, state_encoders, action_decoder_dummies, bucket_id, seq2seq, name=None):
		all_inputs = state_encoders + action_decoder_dummies
		state_size  = buckets[bucket_id][0]
		action_size = buckets[bucket_id][1]

		with ops.name_scope(name, "model_with_buckets", all_inputs):
			with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse = True if bucket_id>0 else None):
				outputs, _ = seq2seq(state_encoders[:state_size], action_decoder_dummies[:action_size], True)
				# outputs - A list of the same length as decoder_inputs of 2D Tensors with shape [batch_size x num_decoder_symbols] containing the generated outputs.	
			batch_output = [int(np.argmax(logit,axis =1)) for logit in outputs]
			batch_output_T = np.ndarray.tolist(np.trasnpose(np.array(batch_output)))
			batch_output_T_padded = []
			batch_output_T_reversed = []
			for sentence_ids in batch_output_T:
				if util.EOS_ID in sentence_ids:
					tmp  = sentence_ids[:sentence_ids.index(util.EOS_ID)]
					batch_output_T_padded.append([util.GO_ID] + tmp + [util.PAD_ID]*(action_size - len(tmp)-1))
				else:
					batch_output_T_padded.append([util.GO_ID]+sentence_ids[:-1])
				batch_output_T_reversed.append(list(reversed(batch_output_T_padded[-1])))

			batch_output_padded = []
			batch_output_reversed = []
			batch_weights = []
			for length_idx in xrange(action_size):
				batch_output_padded.append(np.array([batch_output_T_padded[batch_idx][length_idx] for batch_idx in xrange(self.batch_size)], dtype=np.int32))
				batch_output_reversed.append(np.array([batch_output_T_reversed[batch_idx][length_idx] for batch_idx in xrange(self.batch_size)],dtype=np.int32))
				# Weights
				batch_w = np.ones(self.batch_size, dtype = np.float32)
				for batch_idx in xrange(self.batch_size):
					if length_idx < action_size -1:
						target = batch_output_T_padded[batch_idx][length_idx+1]
					if length_idx == action_size -1 or target == util.PAD_ID:
						batch_weights[batch_idx] = 0.0

		return batch_output_padded, batch_output_reversed, batch_weights

	def prepare_step(self, encoder_input, bucket_id):
		"""Proper format for Seq2Seq model.
		"""

		encoder_size, decoder_size = self.buckets[bucket_id]
		proper_encoder  = list(reversed(encoder_input+[util.PAD_ID]*(encoder_size-len(encoder_input))))
		proper_decoder = [util.GO_ID] + [util.PAD_ID]*decoder_size
		
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


	def generate_action(self, state_encoder, action_decoder_dummy, buckets, bucket_id, seq2seq, name=None):
		all_inputs = state + action_decoder_dummy

		with ops.name_scope(name, "model_with_buckets", all_inputs):
			with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse = True if bucket_id>0 else None):
				outputs, _ = seq2seq(state[:buckets[bucket_id][0]], action_decoder_dummy[:buckets[bucket_id][1]], True)
					
		action_ids = [int(np.argmax(logit, axis = 1)) for logit in outputs]
		if util.EOS_ID in action_ids:
			action_ids = action_ids[:action_ids.index(util.EOS_ID)]

		return action_ids













	
















	
















	
















	
















	
















	
















