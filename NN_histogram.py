#!/usr/bin/env python
# coding: utf-8
"""

Classifying histograms with a Neural Network.  

Histograms are represented as numpy arrays where each elem is number of Points in each [Yi, Yi + dy]. We have 111 Features and ont output, so it's a binary classification.


Why Kears ? 
	Sklearn NN classifier module is poor compared to what could be done with API like Theano. 
	Keras is well documented and very user friendly. In fact, if you are already familiar with sklearn classifiers, you wont be disorientated.
	Keras allows you to save models and load them. Models are stored as JSON and YAML files which is perfect. 
	Keras exemples in officiel GitHub are amazing and very helpful. 

to explore : http://www.chioka.in/why-is-keras-running-so-slow/


"""


TEST_SIZE           = 10 
POURCENTAGE_TRAIN   = 0.15
HIDDEN_LAYER_SIZE   = 55 
NB_FEATURES         = 111
NB_EPOCH            = 100

from sklearn import cross_validation 
import matplotlib.pyplot as plt
import keras                                 #pip install keras in sudo    
from keras.models import Sequential, load_model, model_from_yaml
from keras.layers import Dense
from keras.utils import plot_model

#from sklearn.model_selection import KFold       
import numpy as np
import pandas as pd 
import numpy as np

class Data: 
	def __init__(self, path): 
		self.file_name     = path
		self.loaded_frame  = pd.read_csv(self.file_name, sep=',',header=None) 
		self.frame_numpied = self.loaded_frame.values 
		self.dim_x         = (self.frame_numpied.shape[0], self.frame_numpied.shape[1] - 1)
		self.dim_y         = (self.frame_numpied.shape[0], 1)
		self.X             = self.frame_numpied[ :self.dim_x[0], :self.dim_x[1]]
		self.y             = self.frame_numpied[ :self.dim_y[0], 111]
		self.X_app, self.X_test, self.y_app, self.y_test = self.cross_validate()

	def cross_validate(self): 
		""" needs to change to this https://stackoverflow.com/questions/25889637/how-to-use-k-fold-cross-validation-in-a-neural-network """
		return cross_validation.train_test_split(self.X, self.y, test_size =TEST_SIZE, train_size=POURCENTAGE_TRAIN, random_state=0)

class NeuralNetwork:
	def __init__ (self, x, y): 
		self.X_train            = x
		self.Y_train            = y 
		self.NN                 = self.build()
		self.stored_perfo_data  = None

	def build(self): 
		Network = Sequential() 
		Network.add(Dense(output_dim = HIDDEN_LAYER_SIZE, init='uniform', activation='relu', input_dim=NB_FEATURES)) #first hidden layer
		Network.add(Dense(output_dim = 1, init='uniform', activation='sigmoid')) #output neuron 
		Network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #compile NN
			#optimizer : how to tune wieghts  (adam, SGD..)
			#loss      : loss function to optimize  (exemple : sum(y - ŷ)^^2)
			#metrics   : accuracy
		return Network

	def fit_to_training(self):
		self.stored_perfo_data = self.NN.fit(self.X_train, self.Y_train, validation_split=0.33, batch_size = 10, nb_epoch = NB_EPOCH )
		self.store_NN()
		#self.NN.predict(np.transpose(self.simulate()))
		self.visualize_network()
		self.visualize_perfo()

	def simulate(self): 
		return np.random.randint(low=0, high=5, size=111, dtype='l').T

	def visualize_perfo(self):
		# list all data in history
		print(self.stored_perfo_data.history.keys())
		# summarize history for accuracy
		plt.plot(self.stored_perfo_data.history['acc'])
		plt.plot(self.stored_perfo_data.history['val_acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()
		# summarize history for loss
		plt.plot(self.stored_perfo_data.history['loss'])
		plt.plot(self.stored_perfo_data.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()

	def visualize_network(self):
		fig = plt.figure(figsize=(12, 12))
		self.draw_neural_net(fig.gca(), .1, .9, .1, .9, [22, 7, 1])
		plot_model(self.NN, to_file='model.png')
		plt.show()

	def store_NN(self): 
		self.NN.save('network_weights.h5')
		model_yaml = self.NN.to_yaml()
		with open("model.yaml", "w") as yaml_file:
			yaml_file.write(model_yaml)
		# serialize weights to HDF5
		#model.save_weights("model.h5")
		print("Saved model to disk")

	def draw_neural_net(self, ax, left, right, bottom, top, layer_sizes):
		n_layers = len(layer_sizes)
		v_spacing = (top - bottom)/float(max(layer_sizes))
		h_spacing = (right - left)/float(len(layer_sizes) - 1)
		# Nodes
		for n, layer_size in enumerate(layer_sizes):
			layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
			for m in range(layer_size):
				circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
									color='w', ec='k', zorder=4)
				ax.add_artist(circle)
		# Edges
		for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
			layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
			layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
			for m in range(layer_size_a):
				for o in range(layer_size_b):
					line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
									[layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
					ax.add_artist(line)

def load_NeuralNetwork_model(): 
	yaml_file = open('model.yaml', 'r')
	loaded_model_yaml = yaml_file.read()
	yaml_file.close()
	loaded_model = model_from_yaml(loaded_model_yaml)
	# load weights into new model
	loaded_model.load_weights("network_weights.h5")
	print("Loaded model from disk")



if __name__ == "__main__": 
	#NeuralNetwork(Data('dataset.csv').X, Data('dataset.csv').y).fit_to_training()
	load_NeuralNetwork_model()







