


import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt





class NearestNeighbor:
	
	
	def __init__(self):
		
		self.train_data = None
		self.train_labels = None
		
		
	
	
	
	def train(self, Xtr, Ytr):
		
		# Xtr = N x D matrix, N=number of training examples, D is the length of the input vector
		# Ytr = N x 1 array of labels for each input
		if(len(Xtr.shape) != 2 && len(Ytr.shape) !=1)
		{
			print("wrong shape for training")
			return
		}
		
		
	
	
	









cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'

# Cleaning up variables to prevent loading data multiple times (which may cause memory issue)

try:
	del X_train, y_train
	del X_test, y_test
	print('Clear previously loaded data.')
except:
	pass




X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# Xtr = 50000 x 32 x 32 x 3 as float matrix of 0-255


# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)


# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
	idxs = np.flatnonzero(y_train == y)
	idxs = np.random.choice(idxs, samples_per_class, replace=False)
	for i, idx in enumerate(idxs):
		plt_idx = i * num_classes + y + 1
		plt.subplot(samples_per_class, num_classes, plt_idx)
		plt.imshow(X_train[idx].astype('uint8'))
		plt.axis('off')
		if i == 0:
			plt.title(cls)


plt.show()



# Subsample the data for more efficient code execution in this exercise
num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)




from cs231n.classifiers import KNearestNeighbor

# Create a kNN classifier instance. 
# Remember that training a kNN classifier is a noop: 
# the Classifier simply remembers the data and does no further processing 
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)


# ex 
# tmp = np.array([[ [[1,10],[2,11],[3,12]],[[4,13],[5,14],[6,15]],[[7,16],[8,17],[9,18]]] ])
# (1, 3, 3, 2)
#tmp[0,:,:,0]

#tmp2 = tmp.reshape(1, 3*3*2)

# tmp2 1 x 18 = array([[ 1, 10,  2, 11,  3, 12,  4, 13,  5, 14,  6, 15,  7, 16,  8, 17, 9, 18]])















































