############################################################
# Module  : multilayer perceptron
# Date    : March 14th, 2017
# Author  : https://github.com/aymericdamien/TensorFlow-Examples
############################################################

from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import Variable, Session,  \
                placeholder, reduce_mean,  \
                reduce_sum, matmul, random_normal

from tensorflow.examples.tutorials.mnist import input_data

import app
from prelude import *
from utils import *

os.system('clear')

############################################################
'''
	Data
'''
mnist = input_data.read_data_sets('/tmp/data', one_hot = True)

############################################################
'''
	Training Parameters
'''
learning_rate = 0.001
epochs        = 25
batch_size    = 100
display_step  = 1

'''
	network parameters
'''
num_px      = 28*28
layer_1     = 256
layer_2     = 256
num_classes = 10

############################################################
'''
	Graph input
'''
# x, y :: Tensor Float
x = placeholder('float', [None, num_px])
y = placeholder('float', [None, num_classes])

'''
	network parameters
'''
# theta :: Dict String Variable 
theta = {
	# 
	# weights
	  'h1': Variable(random_normal([num_px , layer_1]))
	, 'h2': Variable(random_normal([layer_1, layer_2]))
	, 'h3': Variable(random_normal([layer_2, num_classes]))
	# 
	# biases
	, 'b1': Variable(random_normal([layer_1]))
	, 'b2': Variable(random_normal([layer_2]))
	, 'b3': Variable(random_normal([num_classes]))
}

############################################################
'''
	create model

	@Use: Given input tensor x and initial 
	       network parameters, output prediction

	perceptron :: Tensor Float 
	           -> Dict String Variable 
	           -> Tensor Float
'''
def perceptron(x, theta):

	# h1, h2, yhat :: Tensor Float
	h1     = tf.nn.relu(matmul(x , theta['h1']) + theta['b1'])
	h2     = tf.nn.relu(matmul(h1, theta['h2']) + theta['b1'])
	yhat   = matmul(h2, theta['h3']) + theta['b3']

	'''
		note if we use out of the box softmax, the we get lots
		of NaNs 
	'''
	# yhat   = tf.nn.softmax(yhat)
	return yhat

y_pred = perceptron(x, theta)

'''
	loss and optimizer
	see this for softmax cross entropy with logits:
		http://stackoverflow.com/questions/34240703/difference-between-tensorflow-tf-nn-softmax-and-tf-nn-softmax-cross-entropy-with

	Note we cannot do -reduce_sum $ y * log yhat since 
	computing yhat = softmax(h3 * o2 + b3)  gives us nans
'''
# cost, opt :: Operation
# cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_pred), reduction_indices = 1))
cost = reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))
opt  = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

############################################################
'''
	Train model 
'''

cycles = int(mnist.train.num_examples/batch_size)

with Session() as sess:

	var = tf.global_variables_initializer() 
	sess.run(var)

	# for e in range(epochs):
	for e in range(2):

		for k in range(cycles):
		# for k in range(cycles):

			xs, ys = mnist.train.next_batch(batch_size)
			_, c   = sess.run([opt, cost], feed_dict={x: xs, y: ys})
	
			print ('\n>> cycle ' + str(k) + ' of epoch ' + str(e))
			print ('\n>> cost: ' + str(c))

	'''
		Test model
	'''		
	# correct_prediction :: Tensor
	correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
	
	# accuracy :: Tensor
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


 





