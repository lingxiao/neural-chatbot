############################################################
# Module  : Seq to Seq
# Date    : Febuary 24th, 2017
# source  : http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
############################################################

import sys
sys.modules[__name__].__dict__.clear()

import numpy as np
from random import shuffle
import tensorflow as tf


############################################################
'''
	helper functions
'''
def one_hot(max_len, index):
	if index < max_len:
		hot = [0]*max_len
		hot[index] = 1
		return hot
	else:
		raise NameError('index out of range')

############################################################
'''
	generate data
'''
n = 20
k = n - 10

nbits = [('{0:0' + str(n) + 'b}').format(i) for i in range(2**k)]
nbits = [map(int,bs) for bs in nbits]
shuffle(nbits)

X = [np.array([[b] for b in bs]) for bs in nbits]
Y = [one_hot(n + 1, sum(bs)) for bs in nbits    ]    

'''
	Split into Train and Test
'''
partition = int(0.9 * len(nbits))

train_X   = X[0:partition]
train_Y   = Y[0:partition]

test_X    = X[partition:]
test_Y    = Y[partition:]

############################################################
'''
	Model
'''

'''
	Create hole for X and Y with
	[batch size, sequence length, input dim] fields
'''
data   = tf.placeholder(tf.float32, [None, 20,1])
label  = tf.placeholder(tf.float32, [None, 21]  )

hidden = 24
cell   = tf.nn.rnn_cell.LSTMCell(hidden, state_is_tuple=True)

'''
	unroll the network
'''
val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
val    = tf.transpose(val, [1, 0, 2])
last   = tf.gather(val, int(val.get_shape()[0]) - 1)

weight  = tf.Variable(tf.truncated_normal([hidden, int(label.get_shape()[1])]))
bias    = tf.Variable(tf.constant(0.1, shape=[label.get_shape()[1]]))
predict = tf.nn.softmax(tf.matmul(last, weight) + bias)

'''
	loss
'''
cross_entropy = - tf.reduce_sum(label * tf.log(tf.clip_by_value(predict,1e-10,1.0)))

'''
	optimizer
'''
step = tf.train.AdamOptimizer(0.2).minimize(cross_entropy)


'''
	init
'''
graph = tf.initialize_all_variables()
sess  = tf.Session()
sess.run(graph)

'''
	train
'''
batch_size = 1000
num_batch  = int(len(X)/batch_size)
epoch      = 5000























