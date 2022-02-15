from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
	""" a kNN classifier with L2 distance """
	
	def __init__(self):
		pass
	
	def train(self, X, y):
		"""
		Train the classifier. For k-nearest neighbors this is just
		memorizing the training data.
		
		Inputs:
		X: A numpy array of shape (num_train, D) containing the training data
		consisting of num_train samples each of dimension D.
		y: A numpy array of shape (N,) containing the training labels, where
		y[i] is the label for X[i].
		"""
		self.X_train = X
		self.y_train = y
	
	
	
	def predict(self, X, k=1, num_loops=0):
		"""
		Predict labels for test data using this classifier.
		
		Inputs:
		- X: A numpy array of shape (num_test, D) containing test data consisting
			of num_test samples each of dimension D.
		- k: The number of nearest neighbors that vote for the predicted labels.
		- num_loops: Determines which implementation to use to compute distances
		between training points and testing points.
		
		Returns:
		- y: A numpy array of shape (num_test,) containing predicted labels for the
			test data, where y[i] is the predicted label for the test point X[i].
		"""
		if num_loops == 0:
			dists = self.compute_distances_no_loops(X)
		elif num_loops == 1:
			dists = self.compute_distances_one_loop(X)
		elif num_loops == 2:
			dists = self.compute_distances_two_loops(X)
		else:
			raise ValueError("Invalid value %d for num_loops" % num_loops)
		
		return self.predict_labels(dists, k=k)
	
	
	
	
	def compute_distances_two_loops(self, X):
		"""
		Compute the distance between each test point in X and each training point
		in self.X_train using a nested loop over both the training data and the
		test data.
		
		Inputs:
		- X: A numpy array of shape (num_test, D) containing test data.
		
		Returns:
		- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
		is the Euclidean distance between the ith test point and the jth training
		point.
		"""
		num_test = X.shape[0]
		num_train = self.X_train.shape[0]
		dists = np.zeros((num_test, num_train))
		for i in range(num_test):
			for j in range(num_train):
				#####################################################################
				# TODO:                                                             #
				# Compute the l2 distance between the ith test point and the jth    #
				# training point, and store the result in dists[i, j]. You should   #
				# not use a loop over dimension, nor use np.linalg.norm().          #
				#####################################################################
				# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
				
				# Can compute the l2 distance squared 
				# (x2-x1)^2 + (y2-y1)^2
				dist = np.subtract(X[i], self.X_train[j]) # [y0-x0, y1-x1, y2 - x2, ... , yn - xn]
				dist = np.square(dist) # [ (y0-x0)^2, (y1-x1)^2, (y2 - x2)^2, ... , (yn - xn)^2]
				# this is sum across all elements of vector
				dist = np.sum(dist)
				dists[i][j] = np.sqrt(dist)
				
				# this is the l2 distance or the prev version squared
				#dists[i, j] = np.linalg.norm(self.X_train[j] - X[i])
				
				
				
				
			
		
		
		# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
		return dists
	
	
	
	
	
	
	def compute_distances_one_loop(self, X):
		"""
		Compute the distance between each test point in X and each training point
		in self.X_train using a single loop over the test data.
		
		X = (num_test, D)
		X_train = (num_train, D)
		
		Input / Output: Same as compute_distances_two_loops
		"""
		num_test = X.shape[0]
		num_train = self.X_train.shape[0]
		dists = np.zeros((num_test, num_train))
		for i in range(num_test):
			#######################################################################
			# TODO:                                                               #
			# Compute the l2 distance between the ith test point and all training #
			# points, and store the result in dists[i, :].                        #
			# Do not use np.linalg.norm().                                        #
			#######################################################################
			# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
			
			# the ith row is the dist between ith test image and each training image 
			test_vector = X[i]
			
			# np.subtract(test_vector, self.X_train) - will subtract the row vector test_vector from each row of the matrix X_train 
			
			dists[i] = np.sqrt(np.sum( np.square(np.subtract(test_vector, self.X_train) ) , axis =1 ) )
			
			
			
		
		# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
		return dists
	
	
	
	
	
	
	def compute_distances_no_loops(self, X):
		"""
		Compute the distance between each test point in X and each training point
		in self.X_train using no explicit loops.
		X = (num_test, D)
		X_train = (num_train, D)
		Input / Output: Same as compute_distances_two_loops
		"""
		num_test = X.shape[0]
		num_train = self.X_train.shape[0]
		dists = np.zeros((num_test, num_train))
		#########################################################################
		# TODO:                                                                 #
		# Compute the l2 distance between all test points and all training      #
		# points without using any explicit loops, and store the result in      #
		# dists.                                                                #
		#                                                                       #
		# You should implement this function using only basic array operations; #
		# in particular you should not use functions from scipy,                #
		# nor use np.linalg.norm().                                             #
		#                                                                       #
		# HINT: Try to formulate the l2 distance using matrix multiplication    #
		#       and two broadcast sums.                                         #
		#########################################################################
		# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
		# https://people.duke.edu/~ccc14/sta-663-2016/03A_Numbers.html
		#http://scipy.github.io/old-wiki/pages/EricsBroadcastingDoc
		
		# doesn't work uses too much memory
		#dists = np.sum( np.square( X[None,:] - self.X_train[:, None] ), axis=-1)
		
		# expand equation (x-y)^2 = x^2 + y^2 - 2xy
		
		dists = np.sqrt(np.sum(np.square(X), axis=1).reshape(num_test, 1) + np.sum(np.square(self.X_train), axis=1) - 2 * X.dot(self.X_train.T))
		
		
		# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
		return dists
	
	
	
	
	def predict_labels(self, dists, k=1):
		"""
		Given a matrix of distances between test points and training points,
		predict a label for each test point.
		
		Inputs:
		- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
		gives the distance betwen the ith test point and the jth training point.
		
		Returns:
		- y: A numpy array of shape (num_test,) containing predicted labels for the
		test data, where y[i] is the predicted label for the test point X[i].
		"""
		num_test = dists.shape[0]
		y_pred = np.zeros(num_test)
		
		# argpartition will give back an array of indecies where the first k elements will be sorted
		k_smallest_idx = np.argpartition(dists, k, axis = 1)
		
		for i in range(num_test):
			# A list of length k storing the labels of the k nearest neighbors to
			# the ith test point.
			
			
			
			# get only the first k indecies
			#k_smallest_idx = k_smallest_idx[:, 0:k] 
			
			# get the indecies of the k smallest distances for this test image
			smallest_idx = k_smallest_idx[i, 0:k]
			
			# get the most frequent labels
			closest_y = self.y_train[smallest_idx]
			
			counts = np.bincount(closest_y)
			
			# the label for the ith test case
			y_pred[i] = np.argmax(counts)
			
			#########################################################################
			# TODO:                                                                 #
			# Use the distance matrix to find the k nearest neighbors of the ith    #
			# testing point, and use self.y_train to find the labels of these       #
			# neighbors. Store these labels in closest_y.                           #
			# Hint: Look up the function numpy.argsort.                             #
			#########################################################################
			# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
			
			pass
			
			# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
			#########################################################################
			# TODO:                                                                 #
			# Now that you have found the labels of the k nearest neighbors, you    #
			# need to find the most common label in the list closest_y of labels.   #
			# Store this label in y_pred[i]. Break ties by choosing the smaller     #
			# label.                                                                #
			#########################################################################
			# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
			
			pass
			
			# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
		
		return y_pred
	
	
	
	



























































