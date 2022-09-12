'''

Image analysis for zebrafish injected with GFP-positive glioblastoma cells imaged using the IncuCyte S3 instrument at 10x objective. The program does the following things order:
* Train a convolutional neural network given training data (unless model already exist)
* Create pixel masks for each class
* Classify images as either valid, invalid or dead
* Measure tumor area and tumor size and output all measurements into an csv file. 

Intermediate results, such as trained model or segmented images, are saved and does not have to be recomputed if there is an error. Delete or move files to force a recomputation. 

If an already trained neural network is provided, the program expect input data in the following format:
* Folders for each plate in the same parental folder. Plate folder names can not include underscore characters. 
* Each plate folder is expected to contain images with name format: "{channel}_{row}{col:d}_1_{day:02d}d{hour:02d}h{minute:02d}m.{fileEnding}";
** Where channel is either "gray" for phase contrast or "green" for GFP. 
** Images are in .tif format
* Segmented images will be in the format {plate_id}_{row}{col}/{day:02d}d{hour:02d}h{minute:02d}m.png. Each folder represent a unique well, and each well contains timestamped images of that well. 

The absolute path to the folder containing raw data as well as the path were all results are saved have to be specified under the "input" section. 

The program requires python 3.5 and tensorflow 1.12 to work.

Segmented images have the following color-class mapping:
- Green: Tumor cells
- Red: Fish, excluding dead or out of focus fish. Also excluding the eye and egg yolk. 
- Yellow: Dead fish
- Blue: Egg yolk
- Turquoise: Fish eye
- Pink: Out of focus or blurry fish
- Orange: Outside well
- Purple: Well background

'''

import numpy as np;
import pandas as pd;

import os;
import parse;
import datetime;

from functools import partial;

import json;

import split;
import utils;
import models;
import filter;

from keras import callbacks;
from keras.preprocessing.image import ImageDataGenerator;

import scipy.misc;

from matplotlib import pyplot as plt;
from pathos.multiprocessing import ProcessingPool as Pool;

import cv2;

###########
## Input ##
###########

## Path to RAW data
# The program will search the following folder for "rawPlatePattern" and for files with "rawPattern" within these folders.
rawPath = "RAW/";
if rawPath is None:
	raise Exception("Need path to raw data. Set variable in main.py.");

## Path where all processed images are saved
basePath = None;
if basePath is None:
	raise Exception("Need path to output folder. Set variable in main.py");

## Filename pattern for processed files. 
# The program saves each well accoding to the folder pattern each filled with inidvidual timepoints 
folderPattern  = "{plate}_{row}{col:02d}";
filePattern    = "{day:02d}d{hour:02d}h{minute:02d}m.{fileEnding}";

## Filename pattern for plates and individual files
rawPlatePattern = "{plate}";
rawPattern      = "{channel}_{row}{col:d}_1_{day:02d}d{hour:02d}h{minute:02d}m.{fileEnding}";

## Filename pattern for images related to training the neural network
rawPaintedPattern  = "{channel}_{row}{col:d}_1_{day:02d}d{hour:02d}h{minute:02d}m.{fileEnding}";
paintedPattern     = "{channel}_{row}{col:d}_1_{day:02d}d{hour:02d}h{minute:02d}m_{type}.{fileEnding}";
weightPattern      = "{channel}_{batch:d}_{row}{col:d}_1_{day:02d}d{hour:02d}h{minute:02d}m_{type}.{fileEnding}";

## List of folders containing training images
paintedPaths = ["Painted20190219/", "Painted20190228/", "Painted20190301/", "Painted20190329/", "Painted20190409/", "Painted20190930/", "Painted20191001/"];

filterPath    = basePath + "Filtered/";			# Folder where segmented images are stored
weightPath   = basePath + "WeightMap/";			# Folder where pixel-weights are stored (for training)

## Path to neural network
netPath = "Model/model.h5";

## Nerual network parameters (training only)
numLayers = 7;
numFilters = 32;
epochs = 2000;

imageSize = 512;		## Images are analyzed as patches with this pixel size

## Weight matrix (training only)
weights = [
	{"weight":  1  , "operation": "addition", "type": "weightMap_Balance", "optional": False, "padParams": {"mode": "edge"}},
];

## Class matrix. Match colors with each channel. 
classes = [
	{"name": "painted", "index": 0, "color": 0x00FF00},		# Tumor : green
	{"name": "painted", "index": 1, "color": 0xFF0000},		# Fish : Red
	{"name": "painted", "index": 2, "color": 0xFFFF00},		# Dead fish: Yellow
	{"name": "painted", "index": 3, "color": 0x0000FF},		# Egg yolk: Blue
	{"name": "painted", "index": 4, "color": 0x00FFFF},		# Eye: Turquoise 
	{"name": "painted", "index": 5, "color": 0xFF00FF},		# Bubble, OOF: Pink
	{"name": "painted", "index": 6, "color": 0xFF6A00},		# Well edge: Orange
	{"name": "painted", "index": 7, "color": 0xB200FF},		# Well background: Purple
];

################
## Initialize ##
################
utils.defineOutput(filterPath);
utils.defineOutput(weightPath);

######################
## Helper functions ##
######################
rawPaths = [rawPath];

def loadRawTrainingImage(original):
	I = loadPaintedImage(original["gray"], original["green"]);

	return {"image": I, "paddings": [0, 0]};

def loadPaintedImage(inputGray, inputGreen):
	gray  = cv2.imread(inputGray ["path"] + "/" + inputGray ["file"], -1) / 2**8 ;
	green = cv2.imread(inputGreen["path"] + "/" + inputGreen["file"], -1) / 2**16;

	return np.dstack((gray, green));

def loadRawImage(inputGray, inputGreen):
	targetSize = (1200, 1200);

	gray  = cv2.imread(inputGray ["path"] + "/" + inputGray ["file"], -1);
	green = cv2.imread(inputGreen["path"] + "/" + inputGreen["file"], -1);

	if gray is None or green is None:
		return None;

	gray  = gray  / 2**8;
	green = green / 2**16; 

	gray  = scipy.misc.imresize(gray , targetSize, interp="bilinear", mode='F');
	green = scipy.misc.imresize(green, targetSize, interp="bilinear", mode='F');

	return np.dstack((gray, green));

def sizeFilter(input):
	I = cv2.imread(input["path"] + "/" + input["file"], -1);

	return I.shape[0] == 512 and I.shape[1] == 512;

def loadPainted():
	inputs = [];

	for i, paintedPath in enumerate(paintedPaths):
		t_inputs = utils.getInputFiles(paintedPattern, "TrainingSet/" + paintedPath, lambda input: input["type"] == "painted" and input["fileEnding"] != "pdn" and sizeFilter(input));
		
		for n, input in enumerate(t_inputs):
			t_inputs[n]["batch"] = i;

		inputs = inputs + t_inputs;

	return inputs;

def loadRawImages(channel):
	inputs = [];

	print("Channel", channel);
	for i, paintedPath in enumerate(paintedPaths):
		t_inputs = utils.getInputFiles(rawPaintedPattern, "TrainingSet/" + paintedPath, lambda input: input["channel"] in channel and input["fileEnding"] == "tif" and sizeFilter(input));

		for n, input in enumerate(t_inputs):
			t_inputs[n]["batch"] = i;

		inputs = inputs + t_inputs;

	return inputs;

if __name__ == '__main__':
	#################################
	## Generate balanced weightmap ##
	#################################

	'''
	Generate pixel-based weight map to ensure correct class balance. 

	Can be commeneted out if training is not necessary
	'''

	inputs  = loadPainted();
	
	numScores = {};
	total     = 0;
	for input in inputs:
		mask   = utils.loadImage(input["path"] + input["file"], "HEX");

		for row in classes:
			color = row["color"];
			index = row["index"];

			if index not in numScores.keys():
				numScores[index] = 0;
			
			numScores[index] += np.sum(mask == color); 
			total    	 	 += np.sum(mask == color);

	for input in inputs:
		mask      = utils.loadImage(input["path"] + input["file"], "HEX");
		weightMap = np.zeros(mask.shape);

		for row in classes:
			color = row["color"];
			index = row["index"];

			# TODO: Check for numScores == 0 / too small
			weightMap[mask == color] = np.min(list(numScores.values())) / numScores[index];

			#print(np.min(list(numScores.values())) / numScores[index] * 255);

		weightMap[weightMap*255 < 1] = 1.0 / 255;

		meta = {**input, **{"type": "weightMap_Balance"}};
		scipy.misc.toimage(weightMap, cmin=0, cmax=1).save(weightPath + weightPattern.format(**meta));
	
	###################
	## Train Network ##
	###################

	'''
	Train neural network. 

	Can be commented out if training is not necessary
	'''

	print("Generate / load neural network");

	## Load neural network if it already exists
	neuralNet = models.load(netPath);
	
	## Train neural network if it does not already exist
	if not neuralNet:
		print("Neural network not found, training new network...");

		## Assemble training data
		grayInputs  = loadRawImages(["grey" , "gray"]);
		greenInputs = loadRawImages(["green"]);

		originalInputs = utils.mergeInputs(["row", "col", "day", "hour", "minute", "batch"], {"gray": grayInputs, "green": greenInputs});

		for i, inp in enumerate(originalInputs):
			originalInputs[i]["row"]    = originalInputs[i]["gray"]["row"];
			originalInputs[i]["col"]    = originalInputs[i]["gray"]["col"];
			originalInputs[i]["day"]    = originalInputs[i]["gray"]["day"];
			originalInputs[i]["hour"]   = originalInputs[i]["gray"]["hour"];
			originalInputs[i]["minute"] = originalInputs[i]["gray"]["minute"];
			originalInputs[i]["batch"]  = originalInputs[i]["gray"]["batch"];
		
		paintedInputs  = loadPainted();

		balancedInputs = utils.getInputFiles(weightPattern, weightPath, lambda input: input["type"] == "weightMap_Balance");
		
		inputs = utils.mergeInputs(["row", "col", "day", "hour", "minute", "batch"], {"painted": paintedInputs, "original": originalInputs, "weightMap_Balance": balancedInputs});

		print(len(inputs), len(originalInputs), len(balancedInputs), len(paintedInputs));
		
		## 
		trainingSet = utils.loadWeightedTrainingSamples(inputs, classes, weights, numChannels=2, customLoad=loadRawTrainingImage);

		numSamples = trainingSet["inputs"].shape[0];
		print(numSamples);

		dataArgs = dict(
			rotation_range=180,
	        width_shift_range=0.2,
	        height_shift_range=0.2,
	        shear_range=0.2,
	        zoom_range=0.1,
	        horizontal_flip=True,
	        vertical_flip=True,
	        fill_mode='reflect');

		traingen  = ImageDataGenerator(**dataArgs);
		targetgen = ImageDataGenerator(**{**dataArgs, "fill_mode": "constant", "cval": 0});

		trainingSet["imageSize"] = (imageSize, imageSize, 2);

		# Data augmentation method
		def datagen(traingen, targetgen, batchSize, classes):
			seed = np.random.randint(0, 2**32);

			traingen  = traingen .flow(trainingSet["inputs" ], seed=seed, batch_size=batchSize);
			targetgen = targetgen.flow(trainingSet["targets"], seed=seed, batch_size=batchSize);

			for X, Y in zip(traingen, targetgen):				
				yield X, Y.reshape(batchSize, -1, len(classes) + 1);

		if not neuralNet:
			neuralNet = models.createUNET(trainingSet['imageSize'], numLayers, numFilters, classes=len(classes), kernel_size=(3, 3));

		batchSize  = 5;
		savePeriod = 100;

		saveCheckpoint    = callbacks.ModelCheckpoint(netPath, verbose=0, period=savePeriod, save_best_only=True, monitor="loss", mode="min");

		neuralNet.fit_generator(datagen(traingen, targetgen, batchSize, classes), steps_per_epoch=(numSamples / batchSize), epochs=epochs, verbose=1, callbacks=[saveCheckpoint]);

		neuralNet = models.load(netPath);
	
	###################
	## Filter Images ##
	###################

	'''
	Segment all images. Files already segmented will be skipped. 
	'''

	print("Apply neural network filter");

	# Get all folders mathcing the rawPlatePattern 
	parentInputs = utils.getInputFiles(rawPlatePattern, rawPath);

	print("Applying to: " + str(len(parentInputs)) + " plates");

	# Loop over all folders
	for parentInput in parentInputs:
		# Get all images in folder as a list
		inputs = utils.getInputFiles(rawPattern, parentInput["path"] + parentInput["file"], lambda input: input["channel"] in ["grey", "gray"]);

		print("Applying to: " + str(len(inputs)) + " images");

		for input in inputs:
			# Define save path of segemnted image
			savePath = filterPath + folderPattern.format(**{**parentInput, **input}) + "/";
			utils.defineOutput(savePath);
			saveFile = savePath + filePattern.format(**{**input, "fileEnding": "png"});

			# If the file does not already exist, segment the image
			if not os.path.isfile(saveFile):
				# Match phase contrast and green GFP channel 
				greenInput = {"file": rawPattern.format(**{**input, "channel": "green"}), "path": input["path"]};
				
				R = loadRawImage(input, greenInput);

				if R is None:
					continue;

				## Segment image in patches, stitch together tiled image, and save segmented image
				FS = [];
				for I, p in split.splitImage(R, imageSize, 128):				
					I = {"image": I, "paddings": [[0, 0], [0, 0]]};

					F = filter.filterImage(I, neuralNet, len(classes), verbose=0);

					FS.append({"offset": p, "image": F});

				F = split.fusePatches(FS, classes);
				L = utils.layered2rgb(F, classes);
							
				scipy.misc.toimage(L, cmin=0, cmax=1).save(saveFile);
	
	#####################################
	## Image classification heuristics ##
	#####################################
	
	'''
	The following functions depends on pixel values hihgly related to the size of the image. Change values to match input with different sizes. 
	'''

	from skimage.transform import rescale;
	from shutil import copyfile;

	def loadRescaledFilteredImage(file, classes):
		M = utils.loadImage(file, type="HEX");
		M = utils.hex2layered(M, classes);

		for i, m in enumerate(M):
			M[i] = rescale(M[i], 2, anti_aliasing=False);

		return M;

	## If image is ok (true) or contains substantial image artifacts (false)
	def isImageOK(I):
		OOF = I[5];
		F   = I[1] + I[2] + I[3] + I[4];

		NOOF = np.sum(OOF);
		NF   = np.sum(F  );

		ret, labels = cv2.connectedComponents(((F + OOF) * 255).astype("uint8"));

		numFish = 0;
		for n in range(1, ret):
			soof = np.sum(OOF[labels == n]);

			s = np.sum(labels == n);

			if s > 10000 and soof / (1 + s - soof) < 1:
				numFish += 1;

		return NOOF / (1 + NF) < 1 and NF > 100000 and NF < 300000 and numFish == 1;

	## If the fish is alive (true) or dead (false)
	def isImageDead(I):
		D = I[2];
		F = I[1] + I[3] + I[4];

		NF   = np.sum(F  );
		ND = np.sum(D);

		return ND > 10000 and ND / (1 + NF) > 0.5;
	
	#############
	## Analyze ##
	#############

	'''
	Convert segemnted images to csv with image class (valid, invalid, dead), tumor size (tumor area and integrated intensity) and survival.
	'''

	## Estimate time of death given series of images
	def findDeathEvent(df):
		df = df.sort_values(by="Time");

		tau        = len(df);
		minError   = np.inf;
		isCensored = True;

		for n in range(len(df)):
			if n == len(df) - 1:
				e = np.sum(df["IsDead"]);
			else:
				e = np.sum(df.iloc[:(n + 1)]["IsDead"]) + np.sum(1 - df.iloc[(n + 1):]["IsDead"]);

			if e <= minError:
				minError = e;
				tau      = df.at[n, "Time"] if n == len(df) - 1 else df.at[n + 1, "Time"];

				isCensored = True if n == len(df) - 1 else False;

		return tau, isCensored;

	def loopInputs(parentInput):
		rows = [];

		inputs = utils.getInputFiles(filePattern, parentInput["path"] + parentInput["file"] + "/");
		IS = [];

		NB = None;
		B = None;

		def loadRawImage(parentInput):
			for rp in [rawPath]:
				path      = rp + rawPlatePattern.format(**parentInput) + "/";
				imagePath = path + rawPattern.format(**{**parentInput, **input, "channel": "green", "fileEnding": "tif"});

				I  = cv2.imread(imagePath, -1);
				
				if I is not None:
					return I;

			return None;
		
		## Create background image
		for input in inputs:
			M = loadRescaledFilteredImage(input["path"] + input["file"], classes);

			I  = loadRawImage(parentInput);
			I = I / 2**16 if I is not None else I;

			IS.append(I);

			if I is None:
				continue;

			if B is None:
				B  = np.zeros(shape=I.shape);
				NB = np.zeros(shape=I.shape);

			isDead = isImageDead(M);
			if isDead or I is None:
				continue;

			RM = np.ones(shape=I.shape);
			RM[M[0] == 1] = 0;
			RM[M[3] == 1] = 0;

			B  = B  + RM * I;
			NB = NB + RM;
		if NB is None:
			return [];

		NB[NB == 0] = 1;
		B = B / NB;

		# Subtract background signal
		for i, I in enumerate(IS):
			if I is not None:
				IS[i] = IS[i] - B;

		for input, I in zip(inputs, IS):
			M = loadRescaledFilteredImage(input["path"] + input["file"], classes);
			
			isOK   = isImageOK(M);
			isDead = isImageDead(M);

			print(input["file"], isOK, isDead, I is None);

			if I is None:
				continue;

			#I  = cv2.imread(path + rawPattern.format(**{**parentInput, **input, "channel": "green", "fileEnding": "tif"}), -1) / 2**16;
			#I2 = cv2.imread(path + rawPattern.format(**{**parentInput, **input, "channel": "gray", "fileEnding": "tif"}), -1) / 2**8;

			I[I > 0.9]   = 0;
			I[M[0] == 0] = 0;

			row = {
				"Plate": parentInput["plate"],
				"Row": parentInput["row"],
				"Column": parentInput["col"],
				"Time": (input["day"] * 24 * 3600 + input["hour"] * 3600 + input["minute"] * 60) / 3600,
				"NumTumorPixels": np.sum(M[0][:]),
				"TumorIntegratedIntensity": np.sum(I[:]),
				"IsDead": isDead,
				"IsOK": isOK
			};

			print(row);

			rows.append(row);

		if len(rows) > 1:
			deathTau, isCensored = findDeathEvent(pd.DataFrame(rows));

			for i, row in enumerate(rows):
				rows[i]["Tau"]        = deathTau;
				rows[i]["IsCensored"] = isCensored;

		return rows;

	parentInputs = utils.getInputFiles(folderPattern, filterPath);

	## 
	with Pool(10) as pool:
		rows = pool.map(loopInputs, parentInputs);

	rows = [item for sublist in rows for item in sublist];

	df = pd.DataFrame(rows);
	df.to_csv(basePath + "estimates.csv", sep="\t", index=False);



