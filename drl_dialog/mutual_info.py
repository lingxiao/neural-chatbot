"""
Author: Heejin Chloe Jeong
Affiliation: University of Pennsylvania

DRL - Mutual Information Model
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

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import seq2seq_model_rl as s2s_mdl
import mutual_info_model as mi_mdl
import util

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("layer_size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("vocab_size", 6004, "Question vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_string("pretrain_dir", "/tmp", "Pretrained checkpoint files directory.")
tf.app.flags.DEFINE_string("pretrain_dir_back", "/tmp", "Pretrained checkpoint files directory for backward model.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")

FLAGS = tf.app.flags.FLAGS

BOC = [[util.GO_ID]] # Beginning of Conversation Vector!
_buckets = [(80,40),(160,80)]

def init_model(session, train_dir, forward_only):
  """Create translation model and initialize or load parameters in session."""
  model_seq2seq = s2s_mdl.Seq2SeqModel(
      FLAGS.vocab_size, _buckets,
      FLAGS.layer_size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, forward_only=forward_only)
  ckpt_seq2seq = tf.train.get_checkpoint_state(FLAGS.pretrain_dir)
  print("Pretrained - Reading model parameters from %s" % ckpt_seq2seq.model_checkpoint_path)
  model_seq2seq.saver.restore(session, ckpt_seq2seq.model_checkpoint_path)

  model_seq2seq_back = s2s_mdl.Seq2SeqModel(
      FLAGS.vocab_size, _buckets,
      FLAGS.layer_size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, forward_only=forward_only)
  ckpt_seq2seq_back = tf.train.get_checkpoint_state(FLAGS.pretrain_dir_back)
  print("Pretrained - Reading model parameters from %s" % ckpt_seq2seq_back.model_checkpoint_path)
  model_seq2seq_back.saver.restore(session, ckpt_seq2seq_back.model_checkpoint_path)

  model = mi_mdl.Mutual_Info(model_seq2seq, model_seq2seq_back, FLAGS.vocab_size, _buckets,
      FLAGS.layer_size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, forward_only=forward_only)
  model.saver.restore(session, ckpt_seq2seq.model_checkpoint_path)
  
  return model

def learning():
	with tf.Session() as sess:
		print("Creating %d layers of %d units." % (FLAGS.num_layers,FLAGS.layer_size))
    model = create_model(sess,False)

    """Reading data? For initial conversation?
    """
    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []

		while True:

      bucket_id
      start_time = time.time()
      step_loss = model.step(sess, state, dummy_action )














def gen_candidates(prev_ut, curr_ut, policy):
  pass


