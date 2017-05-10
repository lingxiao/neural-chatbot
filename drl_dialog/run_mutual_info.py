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
import mutual_info_model_new as mi_mdl
import util

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("layer_size", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("vocab_size", 50005, "Question vocabulary size.")
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

def init_model(session, forward_only):
  """Create translation model and initialize or load parameters in session."""
  if not forward_only:
    print("Creating %d layers of %d units for Pretrained backward model." % (FLAGS.num_layers,FLAGS.layer_size))
    model_seq2seq_back = s2s_mdl.Seq2SeqModel(
          FLAGS.vocab_size, _buckets,
          FLAGS.layer_size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
          FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, forward_only=True)
    ckpt_seq2seq_back = tf.train.get_checkpoint_state(FLAGS.pretrain_dir_back)
    print("Pretrained - Reading model parameters from %s" % ckpt_seq2seq_back.model_checkpoint_path)
    model_seq2seq_back.saver.restore(session, ckpt_seq2seq_back.model_checkpoint_path)

    print("Creating %d layers of %d units for Pretrained model" % (FLAGS.num_layers,FLAGS.layer_size)) 
    with tf.variable_scope("forward"):
       model_seq2seq = s2s_mdl.Seq2SeqModel(
          FLAGS.vocab_size, _buckets,
          FLAGS.layer_size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
          FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, MI_use = True, forward_only=True)
    ckpt_seq2seq = tf.train.get_checkpoint_state(FLAGS.pretrain_dir)
    print("Pretrained - Reading model parameters from %s" % ckpt_seq2seq.model_checkpoint_path)
    model_seq2seq.saver.restore(session, ckpt_seq2seq.model_checkpoint_path)

    print("Creating %d layers of %d units for the Main model." % (FLAGS.num_layers,FLAGS.layer_size))
    with tf.variable_scope("mi") as scope:
      model = mi_mdl.Mutual_Info( model_seq2seq, model_seq2seq_back, FLAGS.vocab_size, _buckets,
          FLAGS.layer_size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
          FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, forward_only=forward_only)
    
    session.run(tf.variables_initializer(model.names.values()))
    assign_var = [tf.assign(v, model_seq2seq.names["forward/"+k.split("mi/")[1]]) for (k,v) in model.names.items()]    
    session.run(assign_var)
  else:
    print("Creating %d layers of %d units for the Main model." % (FLAGS.num_layers,FLAGS.layer_size))
    with tf.variable_scope("mi") as scope:
      model = mi_mdl.Mutual_Info( None, None, FLAGS.vocab_size, _buckets,
          FLAGS.layer_size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
          FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, forward_only=forward_only)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
    
  return model

def read_data(data_path, max_size=None):
  data_set = [[] for _ in _buckets]
  w2idx =  pickle.load(open(os.path.join(FLAGS.data_dir,"w2idx.pkl"),"rb"))
  idx2w =  pickle.load(open(os.path.join(FLAGS.data_dir,"idx2w.pkl"),"rb"))
  for f_name in os.listdir(data_path):
    dialogue = np.ndarray.tolist(np.load(os.path.join(data_path, f_name)))
    #print("======== reading data ======== file: "+f_name)
    for k in xrange(len(dialogue)-1):
      source_ids01 = [x for x in dialogue[k] if not(x in util.removing_ids)]
      source_ids01 = util.refine_words(source_ids01, w2idx, idx2w)
      source_ids02 = [x for x in dialogue[k+1] if not(x in util.removing_ids)]
      source_ids02 = util.refine_words(source_ids02, w2idx, idx2w)

      for bucket_id, (source_size, target_size) in enumerate(_buckets):
        if len(source_ids01) < source_size/2 and len(source_ids02) < source_size/2:
          data_set[bucket_id].append([source_ids01, source_ids02])
          break
  return data_set

def train():
  with tf.Session() as sess:
    model = init_model(sess,False)

    print ("Reading development and training data (limit: %d)."% FLAGS.max_train_data_size)
    train_set = read_data(os.path.join(FLAGS.data_dir,"train"),FLAGS.max_train_data_size)
    dev_set = read_data(os.path.join(FLAGS.data_dir,"dev"))
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))
     # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in xrange(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, reward = 0.0, 0.0
    current_step = 0
    previous_rewards = []

    while True:

      random_number_01 = np.random.random_sample()
      bucket_id = 0 #min([i for i in xrange(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])
      start_time = time.time()
      states, states_back, weights_back = model.get_batch(train_set,bucket_id)
      step_reward = model.step(sess, states, states_back, weights_back, bucket_id, False)
      
      step_time += (time.time()-start_time)/FLAGS.steps_per_checkpoint
      reward += step_reward / FLAGS.steps_per_checkpoint
      current_step += 1

      if current_step % FLAGS.steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        print("global step %d learning rate %.4f step-time %.2f reward " "%.2f" % 
        (model.global_step.eval(), model.learning_rate.eval(), step_time, reward))

        # Decrease the learning rate if no improvement was seen over last 3 times.
        if len(previous_rewards) > 2 and reward > max(previous_rewards[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_rewards.append(reward)

        # Save checkpoint and zero timer and reward.
        checkpoint_path = os.path.join(FLAGS.train_dir, "mi.ckpt")
        model.saver.save(sess, checkpoint_path, global_step = model.global_step)
        step_time, reward = 0.0, 0.0

        # Run evals on development set and print their perplexity.
        #states, states_back, weights_back = model.get_batch(dev_set, bucket_id )
        #eval_reward = model.step(sess, states, states_back, weights_back, bucket_id, True)
        #print(" eval: bucket %d eval_reward %.2f" % (bucket_id, eval_reward))
        sys.stdout.flush()
def decode():
  with tf.Session() as sess:

    model = init_model(sess,True)
    #model.batch_size = 1

    w2idx =  pickle.load(open(os.path.join(FLAGS.data_dir,"w2idx.pkl"),"rb"))
    idx2w =  pickle.load(open(os.path.join(FLAGS.data_dir,"idx2w.pkl"),"rb"))

    sys.stdout.write("> ")
    sys.stdout.flush()
    
    prev_ids = BOC[0]
    sentence = sys.stdin.readline()

    while sentence:
      token_ids = util.sentence_to_token_ids(tf.compat.as_bytes(sentence),w2idx,idx2w)
      bucket_id = 0
      #bucket_id = min([b for b in xrange(len(_buckets)) if _buckets[b][0] > len(token_ids)])
      states, states_back, weights_back = model.get_batch({bucket_id: [(prev_ids, token_ids)]},bucket_id)
      outputs = model.step(sess, states, states_back, weights_back, bucket_id, True)
      #[outputs[i][0] for i in xrange(_buckets[bucket_id][1])]
      #if util.PAD_ID in outputs:
      #  outputs = outputs[:outputs.index(util.PAD_ID)]
      output_sen = [outputs[i][0] for i in xrange(_buckets[bucket_id][1])]
      if util.PAD_ID in output_sen:
        output_sen = output_sen[:output_sen.index(util.PAD_ID)]

      print(" ".join([tf.compat.as_str(idx2w[output]) for output in output_sen]))
      print("> ", end="")
      sys.stdout.flush()
      prev_ids = output_sen
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





