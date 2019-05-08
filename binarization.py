import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import skimage
from skimage import segmentation, io, filters, morphology

from segmentBackground import segBack

### Parameters ###
EROSION_COUNT = 24

def binarizeImage(image, E_C):
	image = segBack(image, E_C)

	thresh = filters.threshold_minimum(image)
	print("Threshold value: {}".format(thresh))
	binary = image > thresh

	##### Combination of Dilation and Erosion #######
												#
	binary = morphology.binary_dilation(binary)     #
	binary = morphology.binary_dilation(binary)     #
	binary = morphology.binary_erosion(binary)      #
	# binary = morphology.binary_dilation(binary)   #
	binary = morphology.binary_erosion(binary)      #
	# binary = morphology.binary_erosion(binary)    #
	# binary = morphology.binary_erosion(binary)    #
	# binary = morphology.binary_erosion(binary)    #
												#
	#################################################
	
	return binary
	
if __name__ == "__main__":

	###Get file from command line ###
	if len(sys.argv) < 2:
		print("ERROR --- Please give the filename in /image-data that you wish to binarize")
		exit()
	fileName = "../image-data/" + sys.argv[1]
	image = io.imread(fileName)


	original_image = image
	plt.figure("Compare", figsize=(15,8))

	plt.subplot(121)
	plt.imshow(original_image, cmap="gray")

	binary = binarizeImage(image,  EROSION_COUNT)

	plt.subplot(122)
	plt.imshow(binary, cmap="gray")

	# fig.tight_layout()
	plt.show()