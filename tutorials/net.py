############################################################
# Module  : homework4 - 
# Date    : Febuary 26th
# Author  : Xiao Ling  
############################################################

import os
import time
import numpy as np
import tensorflow as tf

import app
from prelude import *
from utils import *

############################################################
'''
	The Data: dummy data with long term sequential dependency:

		Pr[Xt = 1] = 1/2 for every t
		Pr[Y0 = 1] = 1/2

		Pr[Yt = 1 | X_{t-3} = 1] = 1
		Pr[Yt = 1 | X_{t-8} = 1] = 0.25
		Pr[Yt = 1 | X_{t-3} = 1 and X_{t-8} = 1] = 0.75

		questions for joao:

			- truncated back prop?
			- simpler flat implementation of what's online?

'''
def to_data(size):

	X = [np.random.choice([0,1], p = [0.5,0.5]) for _ in range(size)]
	Y = []

	for t in range(size):
		if t <= 2:
			q = 0.5
		elif t >= 8:
			if X[t-3] and X[t-8]: 
				q = 0.75
			elif X[t-3]:
				q = 1.0
			elif X[t-8]:
				q = 0.25
			else:
				q = 0.5

		yt = np.random.choice([0,1], p = [1-q, q])
		Y.append(yt)

	return X,Y

def to_batches(data,CONFIG):

	'''	
		break data into batches
		where each batch is of length

	'''
	X,Y        = data
	batch_size = CONFIG['batch-size']

	batch_len = len(X) // batch_size
	x_batchs  = list(chunks(X,batch_len))
	y_batchs  = list(chunks(Y,batch_len))

	'''
		divide again into minibatches for
		truncated backprop
	'''
	num_steps  = CONFIG['num-steps']
	num_epochs = batch_len // num_steps
	ranges     = [(e * num_steps, (e+1)*num_steps) for e in range(num_epochs)]

	xss        = [[x[s:t] for x in x_batchs] for s,t in ranges]
	yss        = [[y[s:t] for y in y_batchs] for s,t in ranges]
	batches    = zip(xss,yss)
	batches    = [zip(xs,ys) for xs,ys in batches]

	return batches

def to_epochs(n, num_data, CONFIG):
	for k in range(n):
		yield to_batches(to_data(num_data), CONFIG)

'''
	The Model with:
		one hot binary encoding x_t 
		hidden vector h_t 
		distribution over y

	h_t = tanh(W (x_t @ h_{t-1}) )
	P_t = softmax (Uh_t)
'''

############################################################
'''
	Run code
'''
# Global config variables
num_steps = 5 # number of truncated backprop steps ('n' in the discussion above)
batch_size = 200
num_classes = 2
state_size = 4
learning_rate = 0.1

"""
	Placeholders
"""
x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')
init_state = tf.zeros([batch_size, state_size])

"""
	RNN Inputs
"""
# Turn our x placeholder into a list of one-hot tensors:
# rnn_inputs is a list of num_steps tensors with shape [batch_size, num_classes]tfr566ttg f  ggbb                                                                   
x_one_hot  = tf.one_hot(x, num_classes)
rnn_inputs = tf.unstack(x_one_hot, axis=1)

"""
Definition of rnn_cell

This is very similar to the __call__ method on Tensorflow's BasicRNNCell. See:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/rnn/python/ops/core_rnn_cell_impl.py#L95
"""
with tf.variable_scope('rnn_cell'):
    W = tf.get_variable('W', [num_classes + state_size, state_size])
    b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))

def rnn_cell(rnn_input, state):
    with tf.variable_scope('rnn_cell', reuse=True):
        W = tf.get_variable('W', [num_classes + state_size, state_size])
        b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))
    return tf.tanh(tf.matmul(tf.concat([rnn_input, state], 1), W) + b)

state = init_state

rnn_outputs = []

for rnn_input in rnn_inputs:
    state = rnn_cell(rnn_input, state)
    rnn_outputs.append(state)
final_state = rnn_outputs[-1]

############################################################

# CONFIG = {'backprop-steps': 5     # truncated backprop 
#          ,'batch-size'    : 200
#          ,'num-classes'   : 2
#          ,'num-state'     : 4
#          ,'num-steps'     : 10
#          ,'learning-rate' : 0.1}

# backprop_steps = CONFIG['backprop-steps']
# batch_size     = CONFIG['batch-size']
# num_class      = CONFIG['num-classes']
# num_state      = CONFIG['num-state']
# num_step       = CONFIG['num-steps']
# learn_rate     = CONFIG['learning-rate']

# X,Y = to_data(5000)

# x  = tf.placeholder(tf.int32, [batch_size, num_step], name = 'input' )
# y  = tf.placeholder(tf.int32, [batch_size, num_step], name = 'output')
# h0 = tf.zeros([batch_size, num_step])
 	
# '''
# 	inputs
# '''
# x_one_hot = tf.one_hot(x, num_class)
# inputs    = tf.unstack(x_one_hot, axis = 1)

# '''
# 	network parameters

# 	y = x'W + b
# '''
# # def cell (x,h):
# # 	W = tf.Variable('W', [num_state, num_class + num_state])
# 	# question: how is this used exactly?

# h  = h0
# hs = []

# for x in inputs:
# 	ht = rnn_cell(x,h)
# 	# hs.append(ht)
































