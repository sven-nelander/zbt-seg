import numpy as np;
import scipy.misc;
import os;

import utils;

def input2output(file):
	return utils.getFileStem(file) + ".png";

def filter(file, model, inputPath):
	sample = utils.loadPaddedImage(inputPath + file, "L");
	image  = sample["image"];

	imageSize = (image.shape[0], image.shape[1], 1);# image.shape[2]);
	numPixels = image.size;

	images = np.array(image);
	images = images.reshape(1, imageSize[0], imageSize[1], imageSize[2]);

	prediction = model.predict(images, batch_size=1, verbose=1);

	prediction = prediction.reshape(imageSize[0], imageSize[1]);

	prediction = utils.cropImage(prediction, sample["paddings"]);

	return prediction;

def filterImage(sample, model, classes=1, verbose=1):
	image  = sample["image"];

	numChannels = 1 if len(image.shape) < 3 else image.shape[2];

	imageSize = (image.shape[0], image.shape[1], numChannels);
	numPixels = image.size;

	images = np.array(image);
	images = images.reshape(1, imageSize[0], imageSize[1], imageSize[2]);

	prediction = model.predict(images, batch_size=1, verbose=verbose);

	prediction = prediction.reshape(imageSize[0], imageSize[1], classes);

	prediction = utils.cropImage(prediction, sample["paddings"]);

	return prediction;

def filterImages(samples, model, classes=1):
	baseImage  = samples[0]["image"];

	numChannels = 1 if len(baseImage.shape) < 3 else baseImage.shape[2];

	imageSize = (baseImage.shape[0], baseImage.shape[1], numChannels);
	numPixels = baseImage.size;

	images = np.zeros(shape=(len(samples), imageSize[0], imageSize[1], imageSize[2]));
	for i, sample in enumerate(samples):
		images[i, :, :, :] = sample["image"];

	predictions = model.predict(images, batch_size=len(samples), verbose=0);

	predictions = predictions.reshape(len(samples), imageSize[0], imageSize[1], classes);

	predictionsList = [];
	for i in range(len(samples)):
		predictionsList.append(utils.cropImage(predictions[i, :, :, :], samples[0]["paddings"]));

	return predictionsList;

def filterAndSave(file, model, inputPath, outputPath):
	filteredImage = filter(file, model, inputPath);
		
	scipy.misc.toimage(filteredImage, cmin=0, cmax=1).save(outputPath + input2output(file));

def filterAndSaveAll(inputs, model, inputPath, outputPath):
	for input in inputs:
		filterAndSave(input["file"], model, inputPath, outputPath);