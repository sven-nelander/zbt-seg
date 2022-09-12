import numpy as np;
import os;

from keras.models import Sequential, load_model;
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, InputLayer;
from keras.models import Model;
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, concatenate, Conv2DTranspose;

from keras import optimizers;
from keras import backend as K;

##########
## File ##
##########

def save(modelPath):
	model.save(modelPath);

def load(modelPath):
	customObjects = {
		"weightedBinaryCrossentropy"     : weightedBinaryCrossentropy,
		"weightedCategoricalCrossentropy": weightedCategoricalCrossentropy,
	};

	if os.path.isfile(modelPath):
		return load_model(modelPath, custom_objects=customObjects);
	return None;

####################
## Loss Functions ##
####################

def weightedBinaryCrossentropy(y_true, y_pred):
	targets = K.clip(K.sign(y_true), 0, 1);
	weights = K.abs(y_true);

	losses  = K.binary_crossentropy(targets, y_pred);

	return K.mean(losses * weights, axis=-1);

def weightedCategoricalCrossentropy(y_true, y_pred):
	W = y_true[:, :, 0];
	T = y_true[:, :, 1:];

	#y_pred /= y_pred.sum(axis=1, keepdims=True);
	y_pred = K.clip(y_pred, 10e-8, 1.0 - 10e-8);

	L = -K.sum(T * K.log(y_pred), axis=2);

	#return K.mean(L * W, axis=-1);
	return K.mean(L * W, axis=-1);

#######################
## Model Definitions ##
#######################

def compile(model, binary=True):
	#opt = optimizers.SGD(lr=0.0001);
	opt = optimizers.Adam(lr=0.0001);

	if binary:
		model.compile(loss=weightedBinaryCrossentropy, optimizer='Adam', metrics=['binary_accuracy']);
	else:
		model.compile(loss=weightedCategoricalCrossentropy, optimizer=opt);

	return model;

def classificationNet(inputShape, numLayers, numFilters):
	model = Sequential();

	for i in range(0, numLayers):
		model.add(Conv2D(filters=numFilters * 2 ** i, kernel_size=3, padding='same', activation='relu', input_shape=inputShape));
		model.add(Conv2D(filters=numFilters * 2 ** i, kernel_size=3, padding='same', activation='relu'));
		model.add(MaxPooling2D((2,2), strides=(2,2)));

	model.add(Dense(1, activation='sigmoid'));
	
	return compile(model);

def multiClassificationNet(input_shape, num_layers, num_filters, num_classes):
	model = Sequential();
	model.add(InputLayer(input_shape));

	for i in range(0, num_layers):
		model.add(Conv2D(filters=num_filters * 2 ** i, kernel_size=3, padding='same', activation='selu'));
		model.add(Conv2D(filters=num_filters * 2 ** i, kernel_size=3, padding='same', activation='selu'));
		model.add(Dropout(0.1));
		#model.add(Conv2D(filters=num_filters * 2 ** i, kernel_size=3, padding='same', activation='relu'));
		model.add(MaxPooling2D((2,2), strides=(2,2)));

	model.add(Flatten());
	#model.add(Dense(124, activation='relu'));
	#model.add(Dropout(0.2));
	#model.add(Dense(256, activation='relu'));
	model.add(Dense(num_classes, activation='softmax'));
	
	opt = optimizers.Adam(lr=0.0001);
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['categorical_accuracy']);

	return model;

def createVGGNet(inputShape, numLayers, numFilters, numFinalFilters):
	model = Sequential();

	for i in range(0, numLayers):
		model.add(Conv2D(filters=numFilters * 2 ** i, kernel_size=3, padding='same', activation='relu', input_shape=inputShape));
		model.add(Conv2D(filters=numFilters * 2 ** i, kernel_size=3, padding='same', activation='relu'));
		model.add(MaxPooling2D((2,2), strides=(2,2)));

	model.add(Conv2D(filters=numFinalFilters, kernel_size=7, padding='same', activation='relu'));
	model.add(Conv2D(filters=1, kernel_size=1, padding='same', activation='relu'));

	model.add(UpSampling2D(size=(2**numLayers, 2**numLayers)));

	model.add(Flatten());
	model.add(Activation(activation='sigmoid'));
	
	return compile(model);

def createUNET(inputShape, numLayers, numFilters, classes = 1, kernel_size=(3, 3)):
	inputs = Input(inputShape);
	downLayers = [];

	previousLayer = inputs;
	for n in range(numLayers - 1):
		conv1 			= Conv2D(numFilters * 2 ** n, kernel_size, activation='relu', padding='same')(previousLayer);
		conv2 			= Conv2D(numFilters * 2 ** n, kernel_size, activation='relu', padding='same')(conv1);
		previousLayer 	= MaxPooling2D(pool_size=(2, 2))(conv2);
		
		downLayers.append(conv2);

	convM1 = Conv2D(numFilters * 2 ** (numLayers - 1), kernel_size, activation='relu', padding='same')(previousLayer);
	convM2 = Conv2D(numFilters * 2 ** (numLayers - 1), kernel_size, activation='relu', padding='same')(convM1);

	previousLayer = convM2;
	for n in range(numLayers - 2, -1, -1):
		up 			  = concatenate([Conv2DTranspose(numFilters * 2 ** n, (3, 3), strides=(2, 2), padding='same')(previousLayer), downLayers[n]], axis=3);
		conv 		  = Conv2D(numFilters * 2 ** n, kernel_size, activation='relu', padding='same')(up);
		previousLayer = Conv2D(numFilters * 2 ** n, kernel_size, activation='relu', padding='same')(conv);

	convF = Conv2D(classes, (1, 1), activation='linear')(previousLayer);
	
	if classes == 1:
		flatten = Flatten()(convF);
		output  = Activation(activation='sigmoid')(flatten);
	else:
		flatten = Reshape((inputShape[0] * inputShape[1], classes))(convF);
		output  = Activation(activation='softmax')(flatten);
		
	model = Model(inputs=[inputs], outputs=[output]);

	return compile(model, classes == 1);

