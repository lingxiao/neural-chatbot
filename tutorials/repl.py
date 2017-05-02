############################################################
# Module  : interactive session to learn TF API
# Date    : March 11th, 2017
# Author  : https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
############################################################

from __future__ import print_function

import os
import time
import collections
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

import app
from prelude import *
from utils import *

# clear bash screen
os.system('clear')

############################################################
'''
	declare variables here
'''
# m :: numpy.ndarray
m = np.array([[1, 2, 3], [4, 5, 6]])

'''
	tf.placeholders can be initalized with
	fixed dim, or arbitrary dim
'''
# X :: Tensor
X = tf.placeholder('float32', [2,3])
Y = tf.placeholder('float32', [None, 3])
Z = tf.placeholder('float32')

############################################################
'''
	Notes on API
'''

'''
	>> transpose :: Tensor -> Tensor
		perm: dim 0 -> dim 1, dim 1 -> dim 0
'''
# m1 :: Tensor
m1 = tf.transpose(m,[1,0])  

'''
	>> reshape :: Tensor -> Tensor
'''
m2 = tf.reshape(m,[-1])
m3 = tf.reshape(m,[3,2])
m4 = tf.reshape(m,[1,6])
m5 = tf.reshape(m,[6,1])
m6 = tf.reshape(m,[-1,3])


'''
	>> split :: Tensor -> [Tensor]
		split(x, [dim1, dim2, ..], k)
			split tensor x into dimensions
			[dim1, dim2, ..] along dimension k
'''
Xs = tf.split(X, [1,2],1)

'''
	print all values in this list
'''
ms = zip([m,m1,m2,m3,m4,m5,m6], range(1000))

############################################################
'''
	REPL session to understand tf.Session.run

	`session` creates what is like a monad transformer stack
	
	run` runs the session and feeds the output to next computation:

		ie: run mx >>= \.x -> ...

	run :: fetches x feed_dict x options x meta_data -> fetches

	* fetches can be:

		- single graph element, which can be:
			* Operation
			* Tensor
			* SparseTensor
			* SparseTensorValue
			* String denoting name of tensor or operation on graph

		- nested list 
		- tuple
		- named tuple
		- dict
		- OrdeeredDict with graph elements at leaves

	* feed_dict overides value in the tensor graph, they can be:

		- if the key is a `Tensor`, the value can be:
			scalar
			string
			list
			ndarray

		- if key is 'Placeholder`, the value can be:
			whatever the type of the placeholder is

		- if the key is a nested tuple of `Tensors` 
		  the value should be a nested tuple with sae structre that
		  maps to their corresponding value as above

	* in reduced syntax, we have for logistic regression:

	-- * graph inputs

	x,y :: Tensor Float32
	x = tf.placeholder [None, num_px]	   
	y = tf.placeholder [None, num_classes]

	-- * model

	w, b :: Variable
	w = tf.variable $ tf.zeros [num_px, num_classes] 
	b = tf.variable $ tf.zeros [10]                  

	yhat :: Tensor Float32
	yhat = tf.nn.softmax $ x * w + b

	-- * loss function

	cost :: Operation
	cost = tf.reduce_mean $ - tf.reduce_sum (Y * log yhat) reduce_idx 

	-- * optimizer
	opt :: Operation
	opt = minimize (tf.train.Optimizer rate) cost

	-- * Note run takes in type `Operation` or `[Operation]`
	sess :: tf.Session
	sess = do
		var <- tf.global_variables_initializer() :: Operation
		run var

		for k in range epochs:
			xs, ys <- return $ mnist.train.next_batch batch_size
			_, _   <- run [opt, cost] ({x: xs, y: ys})


	-- * an even simpler example:
	sess :: tf.Session
	sess = do:
		a1 <- run a
		b1 <- run b
		v  <- run [a,b]
		w  <- run [a,b] ({a : a1, b: b1})

'''
# note tf.constant x is like `return x` or `pure x`
a = tf.constant([10,20]) # :: Tensor
b = tf.constant([20,30]) # :: Tensor
z = tf.Variable([100,200]) # :: Variable

with tf.Session() as s:

	var = tf.global_variables_initializer()
	s.run(var)

	'''
		run tensor
	'''
	a1 = s.run(a)  # :: np.ndarray
	b1 = s.run(b)  # :: np.ndarray
	z1 = s.run(z)  # :: np.ndarray


	print ('\n>> a1 = ' + str(a1) + ' :: ' + str(type(a1)))
	print ('\n>> b1 = ' + str(b1) + ' :: ' + str(type(b1)))
	print ('\n>> z1 = ' + str(z1) + ' :: ' + str(type(z1)))


	'''
		run [tensor]
	'''
	v = s.run([a,b])
	print ('\n>> v = ' +  str(v) + ' :: ' + str(type(v)))

	'''
		run named tuple
	'''
	d = collections.namedtuple('d', ['a','b'])
	w = s.run({'k1' : d(a,b), 'k2': [b,a]})
	print ('\n>> w = ' + str(v))

	'''	
		feed dict
	'''

############################################################
'''
	REPL session to understand basic matrix operations
'''
with tf.Session() as repl:
	var = tf.global_variables_initializer()
	repl.run(var)

	'''
		argmax, argmin
	'''
	v1 = tf.argmax(m5,0).eval()
	print ('>> argmax m5: ' + str(v1[0]))
	print('\n note argmax output index of maximum value in tensor')

	v2 = tf.argmin(m5,0).eval()
	print ('\n>> argmin m5: ' + str(v2[0]))
	print('\n note argmin output index of maximum value in tensor')

	'''
		reduce_mean x = (sum x)/(len x)
	''' 
	v3 = tf.reduce_mean(m2).eval()
	print('\n>> reduce_mean m2: ' + str(v3))


	'''
		equal
	'''
	b1 = tf.equal(tf.argmax(m5,0), tf.argmax(m5,0)).eval()
	print('\n>> equal (argmax m5 0) (argmax m5 0): ' + str(b1[0]) + '\n')

	b2 = tf.equal(tf.argmax(m5,0), tf.argmin(m5,0)).eval()
	print('\n>> equal (argmax m5 0) (argmin m5 0): ' + str(b2[0]) + '\n')

	if True:

		'''
			print out put of transforms of m
		'''
		print ('>> m: \n' + str(m) + '\n')
		for m,idx in ms[1:]:
			print ('>> m' + str(idx) + ': \n' + str(m.eval()) + '\n')

	'''
		print output of transforms of X
	'''
	x = np.array([[1,2,3],[11,12,13]])     # :: numpy.ndarray


	print ('>> X: \n')
	'''
		Note run is a lot like:

		runReader X [[1,2,3],[4,5,6]]
	'''
	print(repl.run(X, feed_dict = {X: x}))

	xs = repl.run(Xs, feed_dict = {X : x})

	print ('>> split X:\n')
	print(xs)

	print ('\n>> (split X)[0]')
	print (xs[0])

	if False:

		print ('>> Y at: \n')
		print(repl.run(Y, \
			feed_dict = {Y: [[1,2,3],[4,5,6]]}))
		print('\n')

		print ('>> Y at: \n')
		print(repl.run(Y, \
			feed_dict = {Y: [[1,2,3],[4,5,6],[4,5,6]]}))
		print('\n')

		print ('>> Z at: \n')
		print(repl.run(Z, \
			feed_dict = {Z: [1,1,1,1,1]}))
		print('\n')

		print ('>> Z at: \n')
		print(repl.run(Z, \
			feed_dict = {Z: [[1,1,1],[1,1,1]]}))
		print('\n')


############################################################
'''
	tensorflow interactive session playing in inpython
'''

'''
	let's start a isession with simple predifined values
'''
session = tf.InteractiveSession()

x = tf.Variable([1.,2.])    # :: Variable Float32
a = tf.constant([3.,3.])    # :: Tensor Float32

 
X = tf.placeholder(tf.float32,[1,2]) # Tensor Float32
W = tf.Variable(tf.zeros([2,2])) # :: Variable
U = tf.Variable([[1.,0.],[0.,1.]]) # :: Variable

'''
	initalize all variables
'''
var = tf.global_variables_initializer()
session.run(var)

'''
	note evaluating a and x is like:

	main = do:
		va  <- eval a
		vx  <- eval x
		vxa <- eval $ x - a

		x1  <- eval $ reshape x 1 2
		y1  <- x1 * W
		y2  <- x1 * U


'''
print('\n>> eval a: \n')
va = a.eval()         # :: np.ndarray
print(va)

print('\n>> eval x: \n')
print(x.eval())

xa = tf.subtract(x,a)
print ('\n>> eval $ x - a:\n')
print(xa.eval())

print('\n>> eval w: \n')
print(W.eval())

x1 = tf.reshape(x,[1,2])
print('\n>> x tranposed :' + str(x1.eval()))
# y1 = tf.matmul(X,W)

y1 = tf.matmul(x1,W)
print("\n x' * W: " + str(y1.eval()))

y2 = tf.matmul(x1,U)
print("\n x' * U: " + str(y2.eval()))


'''
	close session when done
'''
session.close()















