# Libraries
import os
import sys
import logging
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO,
											format="%s(asctime)s:%(levelname)s:%s(message)s")

class VanillaNeuralNetwork(object):
	"""
	Simple neural network model. 
	See __repr__ to instantiate the class.
	"""
	def __repr__(self):
		return "VanillaNeuralNetwork({}, {}, {}, {})".format("3", 
						"[3,6,1]", "0.001", "1000")

	def __init__(self,
								x = None,
								y = None,
								layers = None,
								neurons = None,
								learningRate = None,
								epochs = None):
		super(VanillaNeuralNetwork, self).__init__()
		# Init null objects
		if layers == None:
			raise Exception("layers cannot be empty.")
		if neurons == None:
			raise Exception("neurons canot be empty.")
		if learningRate == None:
			learningRate = 0.001
		if epochs == None:
			epochs = 1000
		if type(x) != np.ndarray:
			raise Exception("x is the input data, provide it.")
		if type(y) != np.ndarray:
			raise Exception("y is the input data, provide it.")
		# Assertions
		if len(neurons) != layers:
			raise ValueError("layers and len(neurons) must be equal.")
		# Class variables
		self.layers = layers
		self.neurons = neurons
		self.learningRate = learningRate
		self.epochs = epochs
		self.x = x
		self.y = y
		self.xInput = tf.placeholder(tf.float32, [1, neurons[0]])
		self.yInput = tf.placeholder(tf.float32, [1, neurons[-1]])
		# Init weights
		self.initWeights()

	def initWeights(self):
		"""
		Initialize weights of the neural network.
		Args:
			None
		Returns:
			None
		"""
		# Create a hash map to store the weights and bias
		self.weights = {}
		self.bias = {}
		# Create weight and bias matrices
		for i in range(self.layers-1):
			# Get the amount of neurons
			currentNeurons = self.neurons[i]
			nextNeurons = self.neurons[i+1]
			print(currentNeurons, nextNeurons)
			# Weights (axb)
			self.weights["w_{}_{}".format(i, i+1)] = tf.Variable(
											tf.random_normal([currentNeurons, nextNeurons]))
			# Bias (1xb)
			self.bias["b_{}_{}".format(i, i+1)] = tf.Variable(
											tf.random_normal([nextNeurons]))

	def logits(self,
						inputValue):
		"""
		Compute logits.
		Args:
			inputValue: 
		Returns:
			None
		"""
		# Compute first layer
		prevLayer = tf.add(tf.matmul(inputValue, self.weights["w_0_1"]),
												self.bias["b_0_1"])
		# Basis function (relu)
		prevLayer = tf.maximum(prevLayer, 0)
		# Compute the rest of the forward prop
		for i in range(1, self.layers-2):
			nextLayer = tf.add(tf.matmul(prevLayer,
												self.weights["w_{}_{}".format(i, i+1)]),
												self.bias["b_{}_{}".format(i, i+1)])
			# Basis function (relu)
			nextLayer = tf.maximum(nextLayer, 0)
			prevLayer = nextLayer
		# Compute last layer
		idxLastLayer = self.layers-1
		# In case of regression
		nextLayer = tf.add(tf.matmul(prevLayer,
												self.weights["w_{}_{}".format(idxLastLayer-1, idxLastLayer)]),
												self.bias["b_{}_{}".format(idxLastLayer-1, idxLastLayer)])
		# In case of classification
		# nextLayer = tf.add(tf.matmul(nextLayer,
		# 										self.weights["w_{}_{}".format()]),
		# 										self.bias["b_{}_{}".format()])
		# nextLayer = tf.maximum(nextLayer, 0)
		# Return value
		return nextLayer

	def predict(self,
							inputValue):
		init = tf.global_variables_initializer()
		# TF Session
		with tf.Session() as sess:
			sess.run(init)
			predictedValues = sess.run(self.logits(inputValue))
		return predictedValues

	def train(self, show = None):
		# Local variables
		if show == None:
			show = False
		# Hyperparameters
		batch_size = 10
		# Logits
		logits = self.logits(self.xInput)
		# Cost function (MSE)
		lossFunction = tf.reduce_mean(tf.pow(logits - self.yInput, 2))
		# Optimizer
		optimizer = tf.train.RMSPropOptimizer(learning_rate = self.learningRate).\
																											minimize(lossFunction)
		# Initializer
		init = tf.global_variables_initializer()
		# TF Session
		with tf.Session() as sess:
			sess.run(init)
			for epoch in range(self.epochs):
				# Run the training operation
				avg_cost = 0
				for j in range(batch_size):
					# Pick a random row
					i = int(np.random.rand()*self.x.shape[0])
					# Extract data
					x_val_for_placeholder = np.array(self.x[i, :]).reshape(1, self.neurons[0])
					y_val_for_placeholder = np.array(self.y[i, :]).reshape(1, self.neurons[-1])
					# Forward/backward prop
					_, cost = sess.run([optimizer, lossFunction],
															feed_dict={self.xInput: x_val_for_placeholder,
																			 self.yInput: y_val_for_placeholder})
					avg_cost += cost
				# Feedback
				if epoch % 10 == 0:
					print("Iteration {} :: cost: {}".format(epoch, avg_cost / self.x.shape[1]))
			# Show results if requested
			if show:
				self.showResults(sess)

	def showResults(self, sess = None):
		predictions = []
		for i in range(self.x.shape[1]):
			output = sess.run(self.logits(self.xInput), 
												feed_dict={self.xInput: np.array(self.x[0, i]).reshape(1, 1)})
			predictions.append(np.squeeze(output))
		print(predictions)
		plt.scatter(self.x[0,:], predictions)
		plt.title("Predicted curve")
		plt.show()

if __name__ == "__main__":
	# Optional ... load dataframe
	x_data = np.random.rand(100, 10)
	y_data = np.random.rand(100, 20)

	# Create neural net instance
	nn = VanillaNeuralNetwork(x = x_data,
	                            y = y_data,
	                            layers = 3,
	                            neurons = [10, 20, 20],
	                            learningRate = 0.001,
	                            epochs = 1000)
	# Train
	nn.train()
	# Predict
	r = nn.predict(np.array(np.random.rand(1, 10), np.float32))
	print(r)
