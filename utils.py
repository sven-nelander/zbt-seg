import numpy as np;
import scipy.misc;
import os;
import parse;
import csv;

from matplotlib import pyplot as plt;
from matplotlib.colors import rgb2hex;

##########
## Math ##
##########

def trailingZeros(n):
	s = str(bin(n));

	return len(s) - len(s.rstrip('0'));

def findNext2Power(n, d):
	for i in range(0, 2**d):
		if trailingZeros(n + i) >= d:
			return i;
	return 0;

def findNext2Pad(image, p):
	pad = [];

	for d, num in enumerate(image.shape):
		if d < 2:
			extra = findNext2Power(num, p);

			before = int(np.floor(extra / 2));
			after  = int(np.ceil(extra / 2));

			pad.append((before, after));
		else:
			pad.append((0, 0));

	return pad;

###########
## Files ##
###########

def defineOutput(output):
	if not os.path.isdir(output):
		os.makedirs(output);

def parseString(pattern, string):
	res = parse.compile(pattern).parse(string);

	return res.named if res else {};

def extractMetaData(pattern, path, files):
	metaParser = parse.compile(pattern);
	
	inputs = [];
	for file in files:
		res = metaParser.parse(file);

		if res:
			input = res.named;
			input["file"] = file;
			input['path'] = path;

			inputs.append(input);

	return inputs;

def getInputFiles(pattern, inputPath, filter=None, getMeta=None):
	metaParser = parse.compile(pattern);

	allFiles  = os.listdir(inputPath);

	inputs = extractMetaData(pattern, inputPath, allFiles);
	
	if getMeta is not None:
		inputs = getMeta(inputs);

	inputs = [input for input in inputs if filter == None or filter(input)];

	return inputs;

def mergeInputs(keys, inputs):
	# Populate list with first key
	initKey, initList = inputs.popitem();
	
	mergedInputs = [{initKey: elm} for elm in initList];

	# Merge
	for n in range(len(mergedInputs)):
		A = [mergedInputs[n][initKey][key] for key in keys];

		for typeKey in inputs:
			if typeKey != initKey: 
				for q in range(len(inputs[typeKey])):
					B = [inputs[typeKey][q][key] for key in keys];

					if A == B:
						mergedInputs[n][typeKey] = inputs[typeKey][q];

	# Remove partial merges
	fullInputs = [];
	for mergedInput in mergedInputs:
		isComplete = True;

		for typeKey in inputs:
			if typeKey not in mergedInput:
				isComplete = False;

		if isComplete:
			fullInputs.append(mergedInput);

	return fullInputs;



def saveCSV(path, name, titles, table, delimiter=","):
	with open(path + name, 'w', newline='') as fp:
		writer = csv.writer(fp);

		writer.writerows([titles]);
		writer.writerows(table);


def groupAndSort(inputs, groupKeys=[], sortKey=None):
	groupedInput = {};

	for input in inputs:
		groupKey = "";

		for key in groupKeys:
			groupKey += str(input[key]);

		if groupKey not in groupedInput.keys():
			groupedInput[groupKey] = {"inputs": []};

			for key in groupKeys:
				groupedInput[groupKey][key] = input[key];

		groupedInput[groupKey]["inputs"].append(input);

	if sortKey:
		for group in groupedInput:
			sortTargets = [input[sortKey] for input in groupedInput[group]["inputs"]];

			order = [i[0] for i in sorted(enumerate(sortTargets), key=lambda x:x[1])];
			
			groupedInput[group]["inputs"] = [groupedInput[group]["inputs"][order[i]] for i, input in enumerate(groupedInput[group]["inputs"])];

	if not groupKeys:
		groupedInput = groupedInput[""];

	return groupedInput;

def filterDone(inputs, outputPath, input2output):
	doneFiles = os.listdir(outputPath);

	newInput = [];
	for input in inputs:
		if input2output(input["file"]) not in doneFiles:
			newInput.append(input);

	return newInput;

def filterDoneInput(inputs, outputPath, input2output):
	doneFiles = os.listdir(outputPath);

	newInput = [];
	for input in inputs:
		if input2output(input) not in doneFiles:
			newInput.append(input);

	return newInput;

def getFileStem(file):
	parts = file.split('.');
	
	return parts[0];


#################
## Image Stuff ##
#################
def floodFill(image, mask, pixels, id, x, y):
	if not mask[x][y] or image[x][y] == id:
		return;

	imageWidth  = image.shape[0];
	imageHeight = image.shape[1];

	pixels.append((x, y));
	image[x][y] = id;

	if x - 1 >= 0:
		floodFill(image, mask, pixels, id, x - 1, y);
	if x + 1 < imageWidth:
		floodFill(image, mask, pixels, id, x + 1, y);
	if y - 1 >= 0:
		floodFill(image, mask, pixels, id, x, y - 1);
	if y + 1 < imageHeight:
		floodFill(image, mask, pixels, id, x, y + 1);

############
## Images ##
############

def hex2class(I, classes):
	T = np.zeros(I.shape);

	for cl in classes:	
		T[I == cl["color"]] = cl["index"];

	return T;

def class2hex(I, classes):
	B = np.argmax(I, axis=2);
		
	P = np.zeros(shape=B.shape, dtype=np.uint32);
	for cl in classes:
		# TODO: Don't assume classes is ordered by index vale

		P[B == cl["index"]] = classes[cl["index"]]["color"];

	return P;

def hex2layered(I, classes):
	filtered = [];
	for row in classes:
		filtered.append(I == row["color"]);

	return filtered;

def hex2rgb(I):
	RGB = np.zeros(shape=(I.shape[0], I.shape[1], 3));
		
	RGB[:, :, 0] = (I // (2**16)) & 0x0000FF;
	RGB[:, :, 1] = (I // (2**8 )) & 0x0000FF;
	RGB[:, :, 2] = (I // (2**0 )) & 0x0000FF;

	return RGB.astype('float32') / 255;

def layered2rgb(I, classes):
	best = np.argmax(I, axis=2);
		
	P = np.zeros(shape=best.shape, dtype=np.uint32);
	for row in classes:
		P[best == row["index"]] = classes[row["index"]]["color"];

	RGB = np.zeros(shape=(best.shape[0], best.shape[1], 3));
	
	RGB[:, :, 0] = P // (2**16) & 0x0000FF;
	RGB[:, :, 1] = P // (2**8 ) & 0x0000FF;
	RGB[:, :, 2] = P // (2**0 ) & 0x0000FF;

	RGB = RGB.astype('float32') / 255;

	return RGB;

def layered2hex(I, classes):
	best = np.argmax(I, axis=2);
		
	P = np.zeros(shape=best.shape, dtype=np.uint32);
	for row in classes:
		P[best == row["index"]] = classes[row["index"]]["color"];

	return P;

def cropImage(image, paddings):
	for dim, padding in enumerate(paddings):
		if dim < len(image.shape):
			delete = [];

			for n in range(padding[0]):
				delete.append(n);
			for n in range(padding[1]):
				delete.append(image.shape[dim] - n - 1);
			
			image = np.delete(image, delete, dim);
	return image;

def loadImage(fileName, type="L"):
	if type == "L":
		image = scipy.misc.imread(fileName, True, 'L');
	elif type == "RGB":
		image = scipy.misc.imread(fileName, False);
	elif type == "HEX":
		image = scipy.misc.imread(fileName, False, 'RGB');

		R = np.left_shift(image[:, :, 0].astype('uint32'), 16);
		G = np.left_shift(image[:, :, 1].astype('uint32'), 8);
		B = np.left_shift(image[:, :, 2].astype('uint32'), 0);

		H = np.bitwise_or(np.bitwise_or(R, G), B);

		return H;

	return image.astype('float32') / 255;


def loadPaddedImage(fileName, type="L", padParams={"mode": "reflect"}):
	image = loadImage(fileName, type);

	paddings = findNext2Pad(image, 4);

	return {"image": np.lib.pad(image, paddings, **padParams), "paddings": paddings};

def padImage(I, d=4, padParams={"mode": "reflect"}):
	paddings = findNext2Pad(I, d);

	return {"image": np.lib.pad(I, paddings, **padParams), "paddings": paddings};

def loadWeightedTrainingSample(inputs, classes, weights, numChannels=1, customLoad=None):
	type = "RGB" if numChannels > 1 else "L";

	if customLoad is not None:
		input = customLoad(inputs["original"]);
	elif "image-binary" not in inputs["original"] or inputs["original"]["image-binary"] is None:
		input   = loadPaddedImage(inputs["original"]["path"] + inputs["original"]["file"], type=type);
	else:
		image = inputs["original"]["image-binary"];

		paddings = findNext2Pad(image, 4);

		input = {"image": np.lib.pad(image, paddings, **{"mode": "reflect"}), "paddings": paddings};

	truths = {};
	for cl in classes:
		I = loadPaddedImage(inputs[cl["name"]]["path"] + inputs[cl["name"]]["file"], "HEX")["image"];
		T = np.zeros(I.shape);
		T[I == cl["color"]] = 1;
		truths[cl["index"]] = T;

	summedWeights = np.zeros(shape=(input["image"].shape[0], input["image"].shape[1]));

	for weight in weights:
		if weight["optional"] == False or os.path.isfile(inputs[weight["type"]]["path"] + inputs[weight["type"]]["file"]):
			weightImage = loadPaddedImage(inputs[weight["type"]]["path"] + inputs[weight["type"]]["file"], padParams=weight["padParams"]);

			weightImage["image"] = 1 - weightImage["image"] if (weight["weight"] < 0) else weightImage["image"];

			if   weight["operation"] == "addition":
				summedWeights += abs(weight["weight"]) * weightImage["image"];
			elif weight["operation"] == "multiplication":
				summedWeights *= abs(weight["weight"]) * weightImage["image"];

	return {'input': input["image"], 'targets': truths, 'weight': summedWeights};

def loadWeightedTrainingSamples(inputs, classes, weights, numChannels=1, customLoad=None):
	samples = [];
	targets = [];

	for input in inputs:
		sample = loadWeightedTrainingSample(input, classes, weights, numChannels, customLoad);

		imageSize = (sample['input'].shape[0], sample['input'].shape[1], numChannels);
		numPixels = sample['input'].shape[0] * sample['input'].shape[1];

		if len(classes) == 1:
			target = np.zeros((sample['input'].shape[0], sample['input'].shape[1]));

			st = sample["targets"][classes[0]["name"]];

			target[st == 0] = -sample["weight"][st == 0];
			target[st == 1] =  sample["weight"][st == 1];

			samples.append(sample['input' ].reshape(imageSize[0], imageSize[1], imageSize[2]));
			targets.append(target.reshape(numPixels));
		else:
			target = np.zeros(shape=(sample['input'].shape[0], sample['input'].shape[1], len(classes) + 1));
			
			target[:, :, 0] = sample["weight"];
			for cl in classes:
				target[:, :, cl["index"] + 1] = sample["targets"][cl["index"]]

			samples.append(sample['input' ].reshape(imageSize[0], imageSize[1], imageSize[2]));
			targets.append(target);

	samples = np.array(samples);
	targets = np.array(targets);
	
	return {'inputs': samples, 'targets': targets, 'imageSize': imageSize, 'numPixels': numPixels};



