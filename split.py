from PIL import Image;
import numpy as np;
import utils;

from functools import partial;

import matplotlib.pyplot as plt;

##########
## Fuse ##
##########
def index2edgeDistance(y, x, width, height):
	return 1 + np.sqrt((width / 2)**2 + (height / 2)**2) -  np.sqrt((x - width / 2) ** 2 + (y - height / 2)**2);

def fusePatches(patches, classes):
	SX = [p["offset"][0] for p in patches];
	SY = [p["offset"][1] for p in patches];
	EX = [p["offset"][2] for p in patches];
	EY = [p["offset"][3] for p in patches];

	W = np.max(EX) - np.min(SX);
	H = np.max(EY) - np.min(SY);

	I = np.zeros(shape=(H, W, len(classes)));
	N = np.zeros(shape=(H, W, len(classes)));

	for patch in patches:
		offset = patch["offset"];

		WW = patch["offset"][2] - patch["offset"][0];
		HH = patch["offset"][3] - patch["offset"][1];

		wf = lambda y, x, z: index2edgeDistance(y, x, width=WW, height=HH);

		weights = np.fromfunction(wf, shape=(HH, WW, len(classes)));

		I[offset[1]:offset[3], offset[0]:offset[2], :] += weights * patch["image"];
		N[offset[1]:offset[3], offset[0]:offset[2], :] += weights;

	N[N == 0] = 1;

	return I / N;

###########
## Split ##
###########
def splitImage(I, splitSize, padding):
	width  = I.shape[1];
	height = I.shape[0];

	if I.ndim < 3:
		I = I[:, :, np.newaxis];

	y = 0;
	while y < height:
		ey = int(y + splitSize);

		x = 0;
		while x < width:
			ex = int(x + splitSize);

			ley = np.clip(ey, 0, height);
			lex = np.clip(ex, 0, width );

			lsy = ley - splitSize;
			lsx = lex - splitSize;

			C = I[lsy:ley, lsx:lex, :];

			yield C, (lsx, lsy, lex, ley);

			x = int(ex - padding);
		y = int(ey - padding);