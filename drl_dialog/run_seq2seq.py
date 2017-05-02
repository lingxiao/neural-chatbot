"""
Author: Heejin Chloe Jeong
Affiliation: University of Pennsylvania

DRL - Seq2Seq model for version 1.0 
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import pickle
import pdb
import util 

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import seq2seq_model_rl as s2s_mdl

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("layer_size", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("vocab_size", 50005, "vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 1000,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("backward", False,
                            "Seq2Seq Backward")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_integer("max_utterance_len",100,"the length of the maximum utterance length")

FLAGS = tf.app.flags.FLAGS

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

BOC = [[util.GO_ID]] # Beginning of Conversation Vector!
if FLAGS.backward:
	print("Backward Seq2Seq")
	_buckets = [(40,40),(80,80)]
	ckpt_name = "seq2seq_backward.ckpt"
else:
	_buckets = [(80,40),(160,80)]
	ckpt_name = "seq2seq.ckpt"

def read_data(data_path, max_size = None):
	data_set = [[] for _ in _buckets]
	w2idx =  pickle.load(open(os.path.join(FLAGS.data_dir,"w2idx.pkl"),"rb"))
	idx2w =  pickle.load(open(os.path.join(FLAGS.data_dir,"idx2w.pkl"),"rb"))

	for f_name in os.listdir(data_path):
		dialogue = np.ndarray.tolist(np.load(os.path.join(data_path, f_name)))
		for k in xrange(len(dialogue)-2):
			"""
			HERE FOR REMOVING BAD TOKENS
			source_ids = []
			target_ids = []
			for i in xrange(len(dialogue[k])):
				if i == 0 or i == len(dialogue[k]): # Assume that it wouldn't start with 'i>' or finish with '<i'
					if not(dialogue[k][i] in util.removing_ids):
						source_ids.append(dialogue[k][i])
				else:
					if not(dialogue[k][i] in util.removing_ids) and not(dialogue[k][i]==6 and (dialogue[k][i+1] ==95 or dialogue[k][i-1]==94)):
						source_ids.append(dialogue[k][i])

			for i in xrange(len(dialogue[k+1])):
				if i == 0 or i == len(dialogue[k+1]): # Assume that it wouldn't start with 'i>' or finish with '<i'
					if not(dialogue[k+1][i] in util.removing_ids):
						source_ids.append(dialogue[k+1][i])
				else:
					if not(dialogue[k+1][i] in util.removing_ids) and not(dialogue[k+1][i]==6 and (dialogue[k+1][i+1] ==95 or dialogue[k+1][i-1]==94)):
						source_ids.append(dialogue[k+1][i])

			for i in xrange(len(dialogue[k+2])):
				if i == 0 or i == len(dialogue[k+2]): # Assume that it wouldn't start with 'i>' or finish with '<i'
					if not(dialogue[k+2][i] in util.removing_ids):
						target_ids.append(dialogue[k+2][i])
				else:
					if not(dialogue[k+2][i] in util.removing_ids) and not(dialogue[k+2][i]==6 and (dialogue[k+2][i+1] ==95 or dialogue[k+2][i-1]==94)):
						target_ids.append(dialogue[k+2][i])
			"""
			source_ids = [x for x in dialogue[k] if not(x in util.removing_ids)] + [x for x in dialogue[k+1] if not(x in util.removing_ids)]
			source_ids = util.refine_words(source_ids, w2idx, idx2w)
			target_ids = [x for x in dialogue[k+2] if not(x in util.removing_ids)]
			target_ids = util.refine_words(target_ids, w2idx, idx2w)
			target_ids.append(util.EOS_ID)

			for bucket_id, (source_size, target_size) in enumerate(_buckets):
				if len(source_ids) < source_size and len(target_ids) < target_size:
					data_set[bucket_id].append([source_ids,target_ids])
					break		
	return data_set

def read_data_backward(data_path, max_size = None):
	data_set = [[] for _ in _buckets]

	for f_name in os.listdir(data_path):
		dialogue = np.ndarray.tolist(np.load(os.path.join(data_path, f_name)))
		print("======== reading data ======== file: "+f_name)
		for k in xrange(len(dialogue)-1):
			source_ids = [x for x in dialogue[k+1] if not(x in util.removing_ids)]
			target_ids = [x for x in dialogue[k] if not(x in util.removing_ids)]
			target_ids.append(util.EOS_ID)

			for bucket_id, (source_size, target_size) in enumerate(_buckets):
				if len(source_ids) < source_size and len(target_ids) < target_size:
					data_set[bucket_id].append([source_ids, target_ids])
					break
	return data_set

def create_model(session, forward_only):
	"""Create translation model and initialize or load parameters in session.
	If there is any problem on checkpoint, see "tensorflow/contrib/framework/python/framework/checkpoint_util.py"

	"""
	if FLAGS.backward:
		model = s2s_mdl.Seq2SeqModel(
	      FLAGS.vocab_size, _buckets,
	      FLAGS.layer_size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
	      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, forward_only=forward_only)
	else:
		with tf.variable_scope("forward"):
			model = s2s_mdl.Seq2SeqModel(
	      FLAGS.vocab_size, _buckets,
	      FLAGS.layer_size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
	      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, forward_only=forward_only)

	  
	if FLAGS.decode:
		ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
		print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
		model.saver.restore(session, ckpt.model_checkpoint_path)
	else:
		print("Created model with fresh parameters.")
		session.run(tf.global_variables_initializer())
	return model


def train():
	with tf.Session() as sess:
		print("Creating %d layers of %d units." % (FLAGS.num_layers,FLAGS.layer_size))
		model = create_model(sess,False)

		print ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)

		if FLAGS.backward:
			train_set = read_data_backward(os.path.join(FLAGS.data_dir,"train"),FLAGS.max_train_data_size)
			dev_set = read_data_backward(os.path.join(FLAGS.data_dir,"dev"))
		else:
			train_set = read_data(os.path.join(FLAGS.data_dir,"train"),FLAGS.max_train_data_size)
			dev_set = read_data(os.path.join(FLAGS.data_dir,"dev"))
		train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
		train_total_size = float(sum(train_bucket_sizes))

		# A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
		# to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
		# the size if i-th training bucket, as used later.
		train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in xrange(len(train_bucket_sizes))]

		# This is the training loop.
		step_time, loss = 0.0, 0.0
		current_step = 0
		previous_losses = []

		while True:
			# There is only one bucket...
			#bucket_id = 0
			random_number_01 = np.random.random_sample()
			bucket_id = min([i for i in xrange(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])
			start_time = time.time()
			encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)
			_, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)
			step_time += (time.time()-start_time)/FLAGS.steps_per_checkpoint
			loss += step_loss / FLAGS.steps_per_checkpoint
			current_step += 1

			# For saving checkpoint, print statistics and run evals.
			if current_step % FLAGS.steps_per_checkpoint == 0:
				# Print statistics for the previous epoch.
				perplexity = math.exp(loss) if loss < 300 else float('inf')
				print("global step %d learning rate %.4f step-time %.2f perplexity" "%.2f" % 
    			(model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity))
    		
				# Decrease the learning rate if no improvement was seen over last 3 times.
				if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
					sess.run(model.learning_rate_decay_op)
				previous_losses.append(loss)

				# Save checkpoint and zero timer and loss.
				checkpoint_path = os.path.join(FLAGS.train_dir, ckpt_name)
				model.saver.save(sess, checkpoint_path, global_step = model.global_step)
				step_time, loss = 0.0, 0.0

				# Run evals on development set and print their perplexity.

				encoder_inputs, decoder_inputs, target_weights = model.get_batch(dev_set, bucket_id )
				_, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
				eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
				print(" eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
				sys.stdout.flush()

def decode():
	with tf.Session() as sess:

		model = create_model(sess,True)
		model.batch_size = 1

		w2idx =  pickle.load(open(os.path.join(FLAGS.data_dir,"w2idx.pkl"),"rb"))
		idx2w =  pickle.load(open(os.path.join(FLAGS.data_dir,"idx2w.pkl"),"rb"))

		sys.stdout.write("> ")
		sys.stdout.flush()
		if FLAGS.backward:
			prev_ids = []
		else:
			prev_ids = BOC[0]
		sentence = sys.stdin.readline()

		while sentence:
			token_ids = util.sentence_to_token_ids(tf.compat.as_bytes(sentence),w2idx)
			#bucket_id = 0
			bucket_id = min([b for b in xrange(len(_buckets)) if _buckets[b][0] > len(token_ids)])

			encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(prev_ids+token_ids, [])]}, bucket_id)
			_, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
			# This is a greedy decoder - outputs are just argmaxes of output_logits.
			outputs = [int(np.argmax(logit, axis = 1)) for logit in output_logits]
			if util.EOS_ID in outputs:
				outputs = outputs[:outputs.index(util.EOS_ID)]

			print(" ".join([tf.compat.as_str(idx2w[output]) for output in outputs]))
			print("> ", end="")
			sys.stdout.flush()
			if FLAGS.backward:
				prev_ids = []
			else:
				prev_ids = outputs
			sentence = sys.stdin.readline()


def main(_):
  if FLAGS.self_test:
    self_test()
  elif FLAGS.decode:
    decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
