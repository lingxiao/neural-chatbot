############################################################
# Module  : logistic regression
# Date    : March 11th, 2017
# Author  : https://github.com/aymericdamien/TensorFlow-Examples
############################################################

from __future__ import print_function

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import app
from utils import *

os.system('clear')

############################################################
'''
	Data
'''
mnist = input_data.read_data_sets('/tmp/data', one_hot = True)

############################################################
'''
	Parameters
'''
learning_rate = 0.01
epochs        = 25
batch_size    = 100

num_px        = 28*28
num_classes   = 10

display_step  = 1

############################################################
'''
	Graph input

	X has dimension _ * 28^2
	Y has dimension _ * 10

	Note when we train X has dimension 100 * 28^2
	but when we test X has dimension 1000 * 28^2

'''
# X, Y :: Tensor Float32
X = tf.placeholder(tf.float32, [None, num_px])
Y = tf.placeholder(tf.float32, [None, num_classes])

'''
	model
'''
# W, b :: Variable
W = tf.Variable(tf.zeros([num_px, num_classes]))
b = tf.Variable(tf.zeros([10]))

# Yhat :: Tensor Float32
Yhat = tf.nn.softmax(tf.matmul(X,W) + b)

'''
	loss function
'''
# cost :: Operation
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(Yhat), reduction_indices = 1))

'''
	gradient descent optimizer
'''
# optimizer :: operation
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

############################################################
'''
	Full Training session
'''
def train():
	with tf.Session() as sess:

		'''
			initialize variables
			'''
		var = tf.global_variables_initializer() # :: operation

		sess.run(var)

		'''
			training
		'''
		for k in range(epochs):
			xs,ys = mnist.train.next_batch(batch_size)
			_, c  = sess.run([optimizer, cost], feed_dict={X: xs, Y: ys})
			print ('\n>> iter : ' + str(k) + ' with cost ' + str(c))

			vb = np.ndarray.tolist(b.eval())

			print('\n>> eval b\n: ', vb)

		print('\n>> finished optimization')

		'''
			validating
		'''
		corrects = tf.equal(tf.argmax(Yhat,1), tf.argmax(Y,1))
		accu     = tf.reduce_mean(tf.cast(corrects, tf.float32))

		print ('\n>> model accuracy: ', accu.eval({X: mnist.test.images, Y: mnist.test.labels}))


'''
	Interactive Training Session
'''
# isess :: InteractiveSession 
isess = tf.InteractiveSession()

# var :: operation
var = tf.global_variables_initializer()
isess.run(var)

# xs, ys :: np.ndarray
xs, ys = mnist.train.next_batch(batch_size)

# c :: np.Float32
_,c    = isess.run([optimizer,cost], feed_dict={X: xs, Y: ys})

# w :: List Float
vw = np.ndarray.tolist(W.eval())
vb = np.ndarray.tolist(b.eval())

'''
	Note `Tensor`s cannot be evaled, they can only be run
	since they are a computation defined at some input

	Variables and constants point to certain values, 
	so they can be evaluated

'''
c  = isess.run(cost, feed_dict={X: xs, Y: ys})    # :: np.float32
vx = isess.run(X   , feed_dict={X: xs})    # :: np.ndarray
vy = isess.run(Y   , feed_dict={Y: ys})

print('\n>> cost: ' + str(c))
print('\n>> run X: ', vx[0])
print('\n>> run Y: ', vy[0])

isess.close()
		










































