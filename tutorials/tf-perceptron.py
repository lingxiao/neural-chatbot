############################################################
# Module  : multilayer perceptron with viz
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
print('\n>> tf-percptron.py')

############################################################
'''
	Data
'''
mnist = input_data.read_data_sets('/tmp/data', one_hot = True)

logs_path = os.path.join( os.getcwd(), 'tutorials/logs')


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


	tf.summary.histogram('relu1',h1)
	tf.summary.histogram('relu2',h2)

	return yhat


'''
	loss and optimizer
	see this for softmax cross entropy with logits:
		http://stackoverflow.com/questions/34240703/difference-between-tensorflow-tf-nn-softmax-and-tf-nn-softmax-cross-entropy-with

	Note we cannot do -reduce_sum $ y * log yhat since 
	computing yhat = softmax(h3 * o2 + b3)  gives us nans
'''

with tf.name_scope('Model'):
	yhat = perceptron(x, theta)

with tf.name_scope('Cost'):
	cost = reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yhat, labels=y))

with tf.name_scope('SGD'):
	opt  = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    # Op to calculate every variable gradient
	grads = tf.gradients(cost, tf.trainable_variables())
	grads = list(zip(grads, tf.trainable_variables()))
	# Op to update all variables according to their gradient
	# apply_grads = opt.apply_gradients(grads_and_vars=grads)


with tf.name_scope('Accuracy'):
# Accuracy
	acc = tf.equal(tf.argmax(yhat, 1), tf.argmax(y, 1))
	acc = tf.reduce_mean(tf.cast(acc, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Create a summary to monitor cost tensor
tf.summary.scalar("cost", cost)

# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", acc)

# Create summaries to visualize weights
for var in tf.trainable_variables():
	tf.summary.histogram(var.name, var)



# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()


############################################################
'''
	Train model 
'''

cycles = int(mnist.train.num_examples/batch_size)

with Session() as sess:

	var = tf.global_variables_initializer() 
	sess.run(var)

    # op to write logs to Tensorboard
	summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

	for e in range(epochs):
	# for e in range(2):
		# for k in range(3):
		for k in range(cycles):

			xs, ys = mnist.train.next_batch(batch_size)
			_, c   = sess.run([opt, cost], feed_dict={x: xs, y: ys})
	
			print ('\n>> cycle ' + str(k) + ' of epoch ' + str(e))
			print ('\n>> cost: ' + str(c))

	'''
		Test model
	'''		
	# correct_prediction :: Tensor
	correct_prediction = tf.equal(tf.argmax(yhat, 1), tf.argmax(y, 1))
	
	# accuracy :: Tensor
	print("Accuracy:", acc.eval({x: mnist.test.images, y: mnist.test.labels}))


	print("Run the command line:\n" \
	+ "--> tensorboard --logdir=/tmp/tensorflow_logs " \
	+ "\nThen open http://0.0.0.0:6006/ into your web browser")






