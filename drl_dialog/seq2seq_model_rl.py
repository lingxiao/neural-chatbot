"""
Author: Heejin Chloe Jeong
Affiliation: University of Pennsylvania

DRL - Seq2Seq model for version 1.0 
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

assert tf.__version__ == '1.0.1'

from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
from tensorflow.contrib.rnn.python.ops import core_rnn_cell

class Seq2SeqModel(object):
	def __init__(self, vocab_size, buckets, layer_size, num_layers, 
		max_gradient_norm, batch_size, learning_rate, learning_rate_decay_factor, 
		use_lstm=False, num_samples=512, MI_use = False, forward_only = False, dtype= tf.float32):

		"""Create a Model:
		Similar to the seq2seq_model.py code in the tensorflow version 0.12.1
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
			batch_size : the size of the batches used during training; the model construction
				is independent of batch_size, so it can be changed after initialization if this is convenient, e.g., for decoding.
			learning_rate : learning rate to start with.
			learning_rate_decay_factor : decay learning rate by this much when needed.
			use_lstm: True -> LSTM cells, False -> GRU cells
			num_samples: the number of samples for sampled softmax
			forward_only : if set, we do not construct the backward pass in the model
			dtype: the data type to use to store internal variables.
		
		"""
		self.vocab_size = vocab_size
		self.buckets = buckets
		self.batch_size = batch_size
		self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype = dtype)
		self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate*learning_rate_decay_factor)
		self.global_step = tf.Variable(0, trainable=False)

		output_projection = None
		softmax_loss_function = None

		# Sampled softmax only makes sense if we sample less than vocabulary size.
		if num_samples > 0 and num_samples < self.vocab_size:
			w_t = tf.get_variable("proj_w",[self.vocab_size, layer_size], dtype = dtype)
			w = tf.transpose(w_t)
			b = tf.get_variable("proj_b", [self.vocab_size], dtype=dtype)
			output_projection = (w,b)

			def sampled_loss(labels, inputs): # The order is opposite to the order in 0.12.x version!!! What the hell?
				
				labels = tf.reshape(labels,[-1,1]) # -1 makes it 1-D. 
				# We need to compute the sampled_softmax_loss using 32bit flotas to avoid numerical instabilities.
				local_w_t = tf.cast(w_t, tf.float32)
				local_b = tf.cast(b,tf.float32)
				local_inputs = tf.cast(inputs, tf.float32)
				# tf.nn -> <module 'tensorflow.python.ops.nn' from 'PATH/tensorflow/python/ops/nn.pyc'>
				return tf.cast(tf.nn.sampled_softmax_loss(weights=local_w_t,
                biases=local_b,
                labels=labels,
                inputs=local_inputs,
                num_sampled=num_samples,
                num_classes=self.vocab_size),dtype)

	  	softmax_loss_function = sampled_loss
	  	self.softmax_loss_function = softmax_loss_function

			# Create the internal multi-layer cell for our RNN.
		if use_lstm:
			single_cell = core_rnn_cell.BasicLSTMCell(layer_size)
		else:
			single_cell = core_rnn_cell.GRUCell(layer_size)

		if num_layers > 1:
			cell = core_rnn_cell.MultiRNNCell([single_cell]*num_layers)
		else:
			cell = single_cell

		# The seq2seq function: we use embedding for the input and attention.
		def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
			return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(encoder_inputs, decoder_inputs,
			 cell, num_encoder_symbols = vocab_size, num_decoder_symbols=vocab_size, embedding_size = layer_size, 
			 output_projection = output_projection, feed_previous = do_decode, dtype = dtype)
		self.seq2seq_f = seq2seq_f
		# Feeds for inputs.
		self.encoder_inputs = []
		self.decoder_inputs = []
		self.target_weights = []
		for i in xrange(buckets[-1][0]):
			self.encoder_inputs.append(tf.placeholder(tf.int32,shape=[None], name="encoder{0}".format(i))) # "encoder{0}".format(N) -> 'encoderN'

		for i in xrange(buckets[-1][1]+1): # For EOS
			self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name= "decoder{0}".format(i)))
			self.target_weights.append(tf.placeholder(dtype, shape=[None], name="weight{0}".format(i)))
		targets = [self.decoder_inputs[i+1] for i in xrange(len(self.decoder_inputs) -1)] # (i+1) because of GO symbol at the beginning

		# Training outputs and losses (a list(len(buckets) of 1-D batched size tensors)
		if forward_only:
			self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(self.encoder_inputs, self.decoder_inputs, targets, 
	  		self.target_weights, buckets, lambda x,y:seq2seq_f(x,y,True),softmax_loss_function=softmax_loss_function)

			if output_projection is not None:
				for b in xrange(len(buckets)):
					self.outputs[b] = [tf.matmul(output, output_projection[0]) + output_projection[1] for output in self.outputs[b]]
		else:
			self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(self.encoder_inputs, self.decoder_inputs, targets, 
	  		self.target_weights, buckets, lambda x,y:seq2seq_f(x,y,False),softmax_loss_function=softmax_loss_function)

		params = tf.trainable_variables() # Returns all variables created with trainable=True
		if not forward_only:
			self.gradient_norms = []
			self.updates = []
			opt = tf.train.GradientDescentOptimizer(self.learning_rate)
			for b in xrange(len(buckets)):
				gradients = tf.gradients(self.losses[b], params)
				clipped_gradients, global_norm = tf.clip_by_global_norm(gradients, max_gradient_norm) # Clips values of multiple tensors by the ratio of the sum of their norms.
				self.gradient_norms.append(global_norm)
				self.updates.append(opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)) #An Operation that applies the specified gradients. If global_step was not None, that operation also increments global_step.

		if MI_use:
			self.names = {str(x.name).split(":0")[0] : x for x in tf.global_variables() if 'forward' in str(x.name).split("/")}
			self.saver = tf.train.Saver(self.names)
		else:
			self.saver = tf.train.Saver(tf.global_variables())

	def step(self, session, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only):
		
		"""Run a step of the model feeding the given inputs.
			
		INPUT: 
			session: tensorflow session to use.
      encoder_inputs: list of numpy int vectors to feed as encoder inputs.
      decoder_inputs: list of numpy int vectors to feed as decoder inputs. (decoder target)
      target_weights:list of numpy float vectors to feed as target weights.
      bucket_id: which bucket of the model to use.
      forward_only: whether to do the backward step or only forward.

    OUTPUT:
    	A triple consisting of gradient norm (or None if we did not do backward), average perplexity, and the outputs.

    """
    # Check if the sizes match.
		encoder_size, decoder_size = self.buckets[bucket_id]
		if len(encoder_inputs) != encoder_size:
		  raise ValueError("Encoder length must be equal to the one in bucket,"
		                   " %d != %d." % (len(encoder_inputs), encoder_size))
		if len(decoder_inputs) != decoder_size:
		  raise ValueError("Decoder length must be equal to the one in bucket,"
		                   " %d != %d." % (len(decoder_inputs), decoder_size))
		if len(target_weights) != decoder_size:
		  raise ValueError("Weights length must be equal to the one in bucket,"
		                   " %d != %d." % (len(target_weights), decoder_size))

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.

		input_feed = {}
		for l in xrange(encoder_size):
			input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]

		for l in xrange(decoder_size):
			input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
			input_feed[self.target_weights[l].name] = target_weights[l]

		# Since our targets are decoder inputs shifted by one, we need one more.
		last_target = self.decoder_inputs[decoder_size].name
		input_feed[last_target] = np.zeros([self.batch_size],dtype=np.int32)

		#Output feed: depends on whether we do a backward step or not.
		if not forward_only:
			output_feed = [self.updates[bucket_id], #An Operation that applies the specified gradients
											self.gradient_norms[bucket_id], # clipped, global gradient norm
											self.losses[bucket_id]] 
		else:
			output_feed = [self.losses[bucket_id]]
			for l in xrange(decoder_size):
				output_feed.append(self.outputs[bucket_id][l])

		outputs = session.run(output_feed, input_feed)
		if not forward_only:
			return outputs[1], outputs[2], None # Graident norm, loss, no outputs?
		else:
			return None, outputs[0], outputs[1:]

	def get_batch(self, data, bucket_id):
		"""Get a random batch of data from the specified bucket, prepare for step.

		To feed data in step(..) it must be a list of batch-major vectors, while
		data here contains single length-major cases. So the main logic of this
		function is to re-index data cases to be in the proper format for feeding.

		INPUT:
		data: a tuple of size len(self.buckets) in which each element contains lists of pairs 
		of input and output data that we use to create a batch. 
		bucket_id : integer, which bucket to get the batch for.

		OUTPUT:
		The triple (encoder_inputs, decoder_inputs, target_weights) for the constructed batch that hs the proper format to call the step function.

		"""

		encoder_size, decoder_size = self.buckets[bucket_id]
		encoder_inputs, decoder_inputs = [],[]
		"""
    1) Get a random batch of encoder and decoder inputs from data, 
		2) Pad them if needed, reverse encoder inputs and 
		3) Add Go to decoder

		"""
		for _ in xrange(self.batch_size):
			encoder_i, decoder_i = random.choice(data[bucket_id])
			encoder_i = encoder_i + [util.PAD_ID]*(encoder_size - len(encoder_i))
			encoder_inputs.append(list(reversed(encoder_i)))

			decoder_i = [util.GO_ID] + decoder_i + [util.PAD_ID]*(decoder_size - len(decoder_i)-1)
			decoder_inputs.append(decoder_i)

		batch_encoder_inputs, batch_decoder_inputs, batch_weights = [],[],[]

		# Batch encoder/decoder inputs are just re-indexed encoder_inputs/decoder_inputs. (Transpose?)

		for length_idx in xrange(encoder_size):
			batch_encoder_inputs.append(np.array([encoder_inputs[batch_idx][length_idx] for batch_idx in xrange(self.batch_size)], dtype = np.int32))

		for length_idx in xrange(decoder_size):
			batch_decoder_inputs.append(np.array([decoder_inputs[batch_idx][length_idx] for batch_idx in xrange(self.batch_size)], dtype = np.int32))

			# Create target_weights to be 0 for targets that are padding. (for each ith element)
			batch_weight = np.ones(self.batch_size, dtype = np.float32)
			for batch_idx in xrange(self.batch_size):
				# The corresponding target is decoder_input shifted by 1 forward.
				if length_idx < decoder_size - 1:
					target = decoder_inputs[batch_idx][length_idx+1]
				if length_idx == decoder_size - 1 or target == util.PAD_ID:
					batch_weight[batch_idx] = 0.0

			batch_weights.append(batch_weight)
		pdb.set_trace()
		return batch_encoder_inputs, batch_decoder_inputs, batch_weights















	





















