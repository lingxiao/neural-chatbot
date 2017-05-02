############################################################
# Module  : Rnn with a hidden layer tracking state of the
#           conversatino
# Date    : March 11th, 2017
# Author  : xiao ling, Heejing Jeong
############################################################

from __future__ import print_function

import os
import nltk
import pickle
import numpy as np

import tensorflow as tf
from tensorflow.contrib import rnn

from prelude   import *
from utils     import *
from tutorials import *

'''
	re implemting rnn so taht it runs on tesla
'''

os.system('clear')

############################################################
'''
	Load data
'''
root      = os.getcwd()
data_dir  = os.path.join(root, 'data/hw4/')

'''
	Settings 
'''
SETTING = {'UNK'             : '<unk>'
          ,'PAD'             : '_'
          ,'End-of-Paragraph': '<EOP>'

          ,'VOCAB_SIZE'      : 6002
          ,'num_classes'     : 2
          ,'min-length'      : 5
          ,'max-length'      : 50}


############################################################
'''
	RNN
	training parameters
'''
learn_rate   = 0.001
train_iters  = 100000
batch_size   = 128
display_step = 10

'''
	network parameters
'''
n_vocab   = SETTING['VOCAB_SIZE'] # one hot vector for each word
n_steps   = SETTING['max-length'] # maximum of 25 words per review
n_hidden  = 128

'''
	graph input
'''
# X, Y :: Tensor
X = tf.placeholder(tf.float32, [None, n_vocab, n_steps])
Y = tf.placeholder(tf.float32, [None, n_classes]       )

'''
	network parameters
'''
# theta :: Dict String Variable
last_layer = {
	 'W': tf.Variable(tf.random_normal([n_hidden, n_classes]))
	,'b': tf.Variable(tf.random_normal([n_classes]))
}


mean_pool = {
	  'W': tf.Variable(tf.random_normal([n_hidden, n_classes]))
	 ,'b': tf.Variable(tf.random_normal([n_classes]))
}	 

'''
	@Use: given input X and parameters theta, 
		  output last hidden hT transformed by:
		  	yhat = W hT + b
'''
# RNN :: Tensor -> ([Tensor], [Tensor])
def RNN(X):
	'''
		conform data shape to rnn function requirements
		X shape       : batch-size * col * row
		required shape: col * batch_size * row
	'''
	X = tf.reshape  (X  , [-1, n_vocab])
	X = tf.split    (X , n_steps, 0   )

	# define instance of lstm cell
	lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)
	# iterate  though the the n_steps
	outputs, states = rnn.static_rnn(lstm_cell, X, dtype=tf.float32)

	return outputs, states

def from_last_layer(X):

	outputs, states = RNN(X)

	yhat = tf.matmul(outputs[-1], last_layer['W']) + last_layer['b']

	return yhat

'''
	@Use: given X and input parameter theta, 
	      output mean pooling of all hidden layers
'''
def mean_pooling(X):

	outputs, states = RNN(X)

	mean = tf.reduce_mean(states,0)

	yhat = tf.matmul(mean, mean_pool['W']) + mean_pool['b']

	return yhat


'''
	cost function and optimizer
'''
# Yhat, cost :: Tensor
# Yhat = from_last_layer(X)
Yhat = mean_pooling(X)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Yhat, labels=Y))

# opt :: Operation
opt   = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cost)

'''
	compute accuracy 
'''
# corrects, accuracy :: Tensor
corrects = tf.equal(tf.argmax(Y,1), tf.argmax(Yhat,1))     
accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))   


'''
	saving model
'''
saver = tf.train.Saver()

with tf.Session() as sess:

	var = tf.global_variables_initializer()
	sess.run(var)

	step = 1

	for _ in range(1):

	# while step * batch_size < train_iters:

		xs,ys = imdb.train_next_batch(batch_size)
		_,c   = sess.run([opt, cost], feed_dict={X: xs, Y: ys})

		print ('\n>> iteration ' + str(step))
		print ('\n>> cost: ' + str(c))


		step += 1

	if True:
		'''
			printing final accuracy
		'''
		print("\n>> Optimization Finished!")
		print("\n>> Computing accuracy on test data")
		corrects = tf.equal(tf.argmax(Yhat,1), tf.argmax(Y,1))
		accuracy = tf.reduce_mean(tf.cast(corrects,'float'))
		txs, tys = imdb.get_test()
		vaccu    = accuracy.eval({X: txs, Y: tys})    
		print ('accuracy : ' + str(vaccu))



















