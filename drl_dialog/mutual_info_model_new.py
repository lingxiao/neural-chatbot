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
import pickle
#xxx = pickle.load(open("/home/ubuntu/data/idx2w.pkl","rb"))

#assert tf.__version__ == '1.0.1'
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import ops
#from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
from tensorflow.contrib.rnn.python.ops import core_rnn_cell

class Mutual_Info(object):
	def __init__(self, pre_trained_seq2seq, pre_trained_backward, vocab_size, buckets, layer_size, num_layers, 
		max_gradient_norm, batch_size, learning_rate, learning_rate_decay_factor, 
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
		self.batch_size = batch_size
		self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype = dtype)
		self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate*learning_rate_decay_factor)
		self.global_step = tf.Variable(0, trainable=False)
		self.pre_trained_seq2seq = pre_trained_seq2seq
		self.pre_trained_backward = pre_trained_backward
		#self.bucket_id = tf.placeholder(tf.int32, shape=(2,), name="bucket_id") # [bucket_id, 0]
		self.bucket_id = 0
		# Variables
		
		w_t = tf.get_variable("proj_w",[self.vocab_size, layer_size], dtype = dtype)
		w = tf.transpose(w_t)
		b = tf.get_variable("proj_b", [self.vocab_size], dtype=dtype)
		output_projection = (w,b)

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
		
		self.states, self.states_back, self.action_dums = [], [], [] # states_back : the 2nd half of the states (each)
		self.actions , self.actions_back = [], []
		self.weights, self.weights_back = [],[]

		for i in xrange(self.buckets[-1][0]):
			self.states.append(tf.placeholder(tf.int32, shape=[None], name ="state{0}".format(i)))

		for i in xrange(self.buckets[-1][1]):
			self.action_dums.append(tf.placeholder(tf.int32, shape=[None], name ="action_dum{0}".format(i)))
			self.actions.append(tf.placeholder(tf.int32, shape=[None], name ="action{0}".format(i)))
			self.weights.append(tf.placeholder(dtype, shape=[None], name ="weight_rl{0}".format(i)))

		for i in xrange(self.buckets_back[-1][0]):
			self.actions_back.append(tf.placeholder(tf.int32, shape=[None], name ="action_back{0}".format(i)))

		for i in xrange(self.buckets_back[-1][1]):
			self.states_back.append(tf.placeholder(tf.int32, shape=[None], name ="state_back{0}".format(i)))

		for i in xrange(self.buckets[-1][1]):
			self.weights_back.append(tf.placeholder(dtype, shape=[None], name="weight_rl_back{0}".format(i)))

		# 1. Get batch actions 
		#>>self.actions, self.actions_back, self.weights, self.joint_logits = self.generate_batch_action(self.states, self.action_dums, self.bucket_id, lambda x,y:seq2seq_f(x,y,True), output_projection= output_projection)
		self.actions_sam, self.logprob = self.generate_batch_action(self.states, self.action_dums, self.bucket_id, lambda x,y:seq2seq_f(x,y,True), output_projection= output_projection)
		# 2. Get the loss
		def mi_score(states, actions, weights, states_back, actions_back, weights_back):
			
			"""
			Args
			#	states, states_back, weights_back : placeholder
			#	actions, actions_back, weights : from generate_batch_action 
			"""
			#self.feeding_data(self.pre_trained_seq2seq, self.buckets, states, actions, weights)
			#self.feeding_data(self.pre_trained_backward, self.buckets_back, actions_back, states_back, weights_back)

			#output_logits = tf.slice(tf.constant(output_logits, dtype=tf.float32), self.bucket_id, [1,-1])
			
			# if self.bucket_id < (len(self.buckets)-1):
			# 	for i in xrange(self.buckets[-1][1]-self.buckets[self.bucket_id][1]):
			# 		actions.append(tf.placeholder(tf.int32, shape=[None], name="action{0}".format(i+self.buckets[self.bucket_id][1])))
			# 		weights.append(tf.placeholder(tf.int32, shape=[None], name="weight_rl{0}".format(i+self.buckets[self.bucket_id][1])))
			# with tf.variable_scope("forward", reuse=True) as scope:
			# 	scope.reuse_variables()
			# 	output_logits,_ = tf.contrib.legacy_seq2seq.model_with_buckets(states, actions, actions[0:],weights, self.buckets, lambda x,y: self.pre_trained_seq2seq.seq2seq_f(x,y,True), softmax_loss_function=self.pre_trained_seq2seq.softmax_loss_function)
			
			output_logits = self.pre_trained_seq2seq.outputs[self.bucket_id]
			#output_logprob = [-tf.log(tf.ones(shape = (self.batch_size, self.vocab_size), dtype=tf.float32) + tf.exp(-logit)) for logit in output_logits]
			log_prob = []
			logprob_s2s = tf.nn.log_softmax(output_logits,dim=0)

			for word_idx in xrange(self.buckets[self.bucket_id][1]):
				one_hot_mat = tf.one_hot(actions[word_idx],depth=self.vocab_size, on_value = 1.0, off_value=0.0, axis =1, dtype=tf.float32 )	
				tmp1 = tf.reshape(tf.slice(logprob_s2s, [word_idx,0,0],[1,-1,-1]), shape = (self.batch_size, self.vocab_size))
				log_prob_word = tf.subtract(tf.reduce_sum(tf.multiply(tmp1 , one_hot_mat),1), tf.log(tf.reduce_sum(tf.exp(tmp1),1)))
				log_prob.append(tf.multiply(log_prob_word, weights[word_idx]))
			
			output_logits_back = self.pre_trained_backward.outputs[self.bucket_id]
			#output_logprob_back = [-tf.log(tf.ones(shape = (self.batch_size, self.vocab_size), dtype=tf.float32) + tf.exp(-logit)) for logit in output_logits_back]
			log_prob_back = []
			logprob_back = tf.nn.log_softmax(output_logits_back,dim=0)
			w_back_new = [np.ones(self.batch_size, dtype = np.float32)] + weights_back[:-1]
			
			for word_idx in xrange(self.buckets_back[self.bucket_id][1]):
				one_hot_mat = tf.one_hot(states_back[word_idx],depth=self.vocab_size, on_value = 1.0, off_value=0.0, axis =1, dtype=tf.float32 )	
				tmp2 = tf.reshape(tf.slice(logprob_back, [word_idx,0,0],[1,-1,-1]), shape = (self.batch_size, self.vocab_size))
				log_prob_word = tf.subtract(tf.reduce_sum(tf.multiply(tmp2 , one_hot_mat),1), tf.log(tf.reduce_sum(tf.exp(tmp2),1)))
				log_prob_back.append(tf.multiply(log_prob_word, w_back_new[word_idx]))
			
			return tf.divide(tf.add_n(log_prob), tf.add_n(weights[:self.buckets[self.bucket_id][1]])) + tf.divide(tf.add_n(log_prob_back), tf.add_n(w_back_new[:self.buckets_back[self.bucket_id][1]])) #+ tf.constant(20.0, shape=(self.batch_size,), dtype = tf.float32)
		
		if not forward_only:
			self.neg_penalty = tf.placeholder(tf.float32, shape=[None], name="neg_penalty") #repeat_penalty(self.actions)
			self.reward =  mi_score(self.states, self.actions, self.weights, self.states_back, self.actions_back, self.weights_back) + tf.scalar_mul(tf.constant(0.05,shape=()), tf.add_n(self.weights[:self.buckets[self.bucket_id][1]]))
			joint_logprob = tf.reduce_sum(self.logprob,axis=0)
			# 3. Gradient Descent Optimization
			params = [x for x in tf.trainable_variables() if "mi" in str(x.name).split("/")]
			cost = tf.scalar_mul(tf.constant(-1.0,shape=()), tf.add(self.neg_penalty, self.reward)) #tf.add(self.neg_penalty, self.reward)
			opt = tf.train.GradientDescentOptimizer(self.learning_rate)
			gradients = tf.gradients(tf.matmul(tf.reshape(cost, shape=(self.batch_size,1)), tf.reshape(joint_logprob,shape=(self.batch_size,1)), transpose_a=True), params)
			clipped_gradients, global_norm = tf.clip_by_global_norm(gradients, max_gradient_norm) # Clips values of multiple tensors by the ratio of the sum of their norms.
			self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step) #An Operation that applies the specified gradients. If global_step was not None, that operation also increments global_step.

		self.names = {str(x.name).split(":0")[0] : x for x in tf.global_variables() if 'mi' in str(x.name).split("/")}	
		self.saver = tf.train.Saver(self.names)
		

	def step(self, session, states, states_back, weights_back, bucket_id, forward_only):
		state_size, action_size = self.buckets[bucket_id]
		action_back_size, state_back_size = self.buckets_back[bucket_id]

		if len(states) != state_size:
			raise ValueError("Encoder length must be equal to the one in bucket,"
		                   " %d != %d." % (len(states), state_size))
		if len(states_back) != state_back_size:
			raise ValueError("Encoder length must be equal to the one in bucket,"
		                   " %d != %d." % (len(states_back), state_back_size))
		if len(weights_back) != state_back_size:
			raise ValueError("Weights length must be equal to the one in bucket,"
		                   " %d != %d." % (len(weights_back), state_back_size))

		input_feed = {}
		#input_feed[self.bucket_id.name] = [bucket_id, 0]
		for l in xrange(state_size):
			input_feed[self.states[l].name] = states[l]

		action_dums = []
		for _ in xrange(self.batch_size):
			action_i_dum = [util.GO_ID] + [util.PAD_ID]*(action_size-1)
			action_dums.append(action_i_dum)

		for l in xrange(action_size):
			input_feed[self.action_dums[l].name] =  np.array([action_dums[batch_idx][l] for batch_idx in xrange(self.batch_size)], dtype = np.int32)

		for l in xrange(state_back_size):
			input_feed[self.states_back[l].name] = states_back[l]
			input_feed[self.weights_back[l].name] = weights_back[l]

		#>>output_feed = [self.actions, self.actions_back, self.weights]
		#>>outputs = session.run(output_feed, input_feed)
		output_feed = self.actions_sam
		outputs = session.run(output_feed, input_feed)
		outputs_T = np.ndarray.tolist(np.transpose(np.array(outputs)))
		for batch_idx in xrange(self.batch_size):
			if util.EOS_ID in outputs_T[batch_idx]:
				outputs_T[batch_idx] = outputs_T[batch_idx][:outputs_T[batch_idx].index(util.EOS_ID)]
			outputs_T[batch_idx] = [util.GO_ID] + outputs_T[batch_idx] + (action_size-len(outputs_T[batch_idx])-1)*[util.PAD_ID]
		actions = []
		for length_idx in xrange(action_size):
			actions.append([outputs_T[batch_idx][length_idx] for batch_idx in xrange(self.batch_size)])
		if not forward_only:

			#>>if len(outputs[0]) != action_size:
			if len(actions) != action_size:
				pdb.set_trace()
				raise ValueError("Encoder length must be equal to the one in bucket,"
			                   " %d != %d." % (len(states), state_size))
			""">>
			if len(outputs[1]) != action_back_size:
				pdb.set_trace()
				raise ValueError("Encoder length must be equal to the one in bucket,"
			                   " %d != %d." % (len(states_back), state_back_size))
			if len(outputs[2]) != action_size:
				pdb.set_trace()
				raise ValueError("Weights length must be equal to the one in bucket,"
			                   " %d != %d." % (len(weights_back), state_back_size))
			"""
			actions_rev = list(reversed(actions))
			weights =[]
			for length_idx in xrange(action_size):
				batch_weight = np.ones(self.batch_size, dtype = np.float32)
				for batch_idx in xrange(self.batch_size):
					# The corresponding target is decoder_input shifted by 1 forward.
					if length_idx < action_size - 1:
						target = actions[length_idx+1][batch_idx]

					if length_idx == action_size - 1 or target == util.PAD_ID:
						batch_weight[batch_idx] = 0.0
				weights.append(batch_weight)
			cum_weights = np.sum(weights,0)

			neg_penalty = []
			for batch_idx in xrange(self.batch_size):
				neg_penalty.append(sum([-10.0 for word_idx in xrange(action_size-1) if ((actions[word_idx][batch_idx] == actions[word_idx+1][batch_idx]) and not(actions[word_idx][batch_idx] == util.PAD_ID))] )) 
				#if (actions[1][batch_idx] == 32) and (cum_weights[batch_idx] == 1.0): # preventing "no"
				#	penalty[batch_idx] -= 5.0
				#if (actions[1][batch_idx] == 6) and ( actions[2][batch_idx]==123) and (actions[3][batch_idx] == 126):
				#	penalty[batch_idx] -= 5.0

			input_feed[self.neg_penalty.name] = neg_penalty


			for l in xrange(state_size):
				input_feed[self.pre_trained_seq2seq.encoder_inputs[l].name] = states[l]
			for l in xrange(action_size):
				input_feed[self.actions[l].name] = actions[l]
				input_feed[self.pre_trained_seq2seq.decoder_inputs[l].name] = actions[l]#>>outputs[0][l]
				input_feed[self.pre_trained_seq2seq.target_weights[l].name] = weights[l]#>>outputs[2][l]

			input_feed[self.weights[0].name] = np.ones(self.batch_size, dtype = np.float32)
			for l in xrange(action_size-1):
				input_feed[self.weights[l+1].name] = weights[l]

			for l in xrange(action_back_size):
				input_feed[self.actions_back[l].name] = actions_rev[l]
				input_feed[self.pre_trained_backward.encoder_inputs[l].name] = actions_rev[l]#>>outputs[1][l]
				input_feed[self.pre_trained_backward.encoder_inputs[l+action_back_size].name] = actions_rev[l]##>>outputs[1][l]
			
			for l in xrange(state_back_size):
				input_feed[self.pre_trained_backward.decoder_inputs[l].name] = states_back[l]
				input_feed[self.pre_trained_backward.target_weights[l].name] = weights_back[l]

			output_feed = [self.updates, self.reward]
			outputs_final = session.run(output_feed, input_feed)
			#pdb.set_trace()
			return sum(outputs_final[1])/self.batch_size
		return actions

	def generate_batch_action(self, state_encoders, action_decoder_dummies, bucket_id, seq2seq, output_projection, name=None):
		# bucket_id is a placeholder
		#state_size  = tf.slice(tf.constant(self.bucekts, dtype=tf.int32), bucket_id, [1,1]) #
		#action_size = tf.case(tf.divice(state_size,2), tf.int32) #
		state_size = self.buckets[bucket_id][0]
		action_size = self.buckets[bucket_id][1]

		all_inputs = state_encoders + action_decoder_dummies
		with ops.name_scope(name, "model_with_buckets", all_inputs):
			with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse = True if bucket_id>0 else None):
				output_logits, _ = seq2seq(state_encoders[:state_size], action_decoder_dummies[:action_size])
				# output_logits - A list of the same length as decoder_inputs of 2D Tensors with shape [batch_size x num_decoder_symbols](64, 512) containing the generated outputs.	

		output_logits = [tf.matmul(output, output_projection[0]) + output_projection[1] for output in output_logits]
		batch_logprob = tf.reduce_max(tf.nn.log_softmax(output_logits,dim=0),axis=2)
		#batch_logits = [tf.reduce_max(logit, axis=1) for logit in output_logits]
		batch_output = [tf.cast(tf.argmax(logit, axis = 1), dtype=tf.int32) for logit in output_logits]
		""">>
		batch_output_2D_T = np.transpose(np.array([tf.split(v,self.batch_size*[1],axis=0) for v in batch_output])) # np.ndarray.tolist(? 64x40. Each row -> each sentence
		go  = tf.constant(util.GO_ID, shape=(1,), dtype=tf.int32)
		eos = tf.constant(util.EOS_ID, shape=(), dtype=tf.int32)
		pad = tf.constant(util.PAD_ID, shape=(1,), dtype=tf.int32)
		w_1 = tf.constant(1, shape =(1,), dtype = tf.float32)
		w_0 = tf.constant(0, shape =(1,), dtype = tf.float32)

		batch_output_T_padded = []
		weights_T = []
		for sen_idx in xrange(self.batch_size):
			tmp = [go]
			tmp_w = [w_1]
			for word_idx in xrange(action_size-1):
				condition1 = tf.equal(eos, tf.reshape(batch_output_2D_T[sen_idx][word_idx], shape=()))
				condition2 = tf.equal(tf.reshape(pad, shape=()),tf.reshape(batch_output_2D_T[sen_idx][word_idx-1], shape=()) )
				tmp.append(tf.cond(tf.logical_or( condition1, condition2 ), lambda:pad, lambda:batch_output_2D_T[sen_idx][word_idx]))
				tmp_w.append(tf.cond(tf.logical_or( condition1, condition2), lambda:w_0, lambda: w_1))
			batch_output_T_padded.append(tmp)
			weights_T.append(tmp_w)

		tmp_matrix = np.ndarray.tolist(np.transpose(batch_output_T_padded))
		batch_output_padded = [tf.cast(tf.concat(w,0), tf.int32) for w in tmp_matrix] 
		batch_output_padded_rev = list(reversed(batch_output_padded))
		tmp_matrix_w = np.ndarray.tolist(np.transpose(np.array(weights_T)))
		batch_weights = [tf.cast(tf.concat(w,0), tf.float32) for w in tmp_matrix_w]

		# prob computation
		batch_joint_logits = tf.add_n([tf.multiply(batch_logits[i], batch_weights[i]) for i in xrange(action_size)])
		return batch_output_padded, batch_output_padded_rev, batch_weights, batch_joint_logits
		"""
		return batch_output, batch_logprob

	def get_batch(self, data, bucket_id):
		"""Proper format for Seq2Seq model.
		"""

		state_size, action_size = self.buckets[bucket_id]
		states, states_back = [],[]
		batch_states, batch_states_back, batch_weights_back = [], [], []
		for _ in xrange(self.batch_size):
			state_i01, state_i02 = random.choice(data[bucket_id])
			state_i = state_i01 + state_i02 + [util.PAD_ID]*(state_size - len(state_i01) - len(state_i02))
			state_j = [util.GO_ID] + state_i02 + [util.PAD_ID]*int(state_size/2 - len(state_i02)-1)
			states.append(list(reversed(state_i)))
			states_back.append(state_j) # no reversed! 
		for length_idx in xrange(state_size):
			batch_states.append(np.array([states[batch_idx][length_idx] for batch_idx in xrange(self.batch_size)], dtype =np.int32))
		for length_idx in xrange(int(state_size/2)):
			batch_states_back.append(np.array([states_back[batch_idx][length_idx] for batch_idx in xrange(self.batch_size)], dtype = np.int32))
			w = np.ones(self.batch_size, dtype = np.float32)
			for batch_idx in xrange(self.batch_size):
				# The corresponding target is decoder_input shifted by 1 forward.
				if length_idx < int(state_size/2) - 1:
					target = states_back[batch_idx][length_idx+1]
				if length_idx == int(state_size/2) - 1 or target == util.PAD_ID:
					w[batch_idx] = 0.0
			batch_weights_back.append(w)
		return batch_states, batch_states_back, batch_weights_back

	def feeding_data(self, model, buckets, states, actions, weights):
		"""
		Arg:
			states: placeholder
			actions: 
			weights:

		"""
		for l in xrange(buckets[self.bucket_id][0]):
			l_name = str(model.encoder_inputs[l].name).split(":0")[0]
			model.encoder_inputs[l] = tf.identity(states[l], name = l_name)


		for l in xrange(buckets[self.bucket_id][1]):
			l_name = str(model.decoder_inputs[l].name).split(":0")[0]
			l_name_w = str(model.target_weights[l].name).split(":0")[0]
			model.decoder_inputs[l] = tf.identity(actions[l], name = l_name)
			model.target_weights[l] = tf.identity(weights[l], name = l_name)
		return None

















