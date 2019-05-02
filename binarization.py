import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import segmentation, io, filters, morphology

image = io.imread("../image-data/P123-Fg001-R-C01-R01-fused.jpg")
original_image = image

image = segmentation.inverse_gaussian_gradient(image, alpha=200, sigma=1)
mask = segmentation.flood(image, (int(original_image.shape[0] / 2), int(original_image.shape[1] / 2)), tolerance=0.5)
mask = morphology.closing(mask, selem=morphology.disk(4))

original_image[mask == False] = 0

image = original_image

#thresh = filters.threshold_otsu(image)
binary = image > 50

binary = morphology.binary_dilation(binary)
binary = morphology.binary_dilation(binary)
binary = morphology.binary_erosion(binary)
binary = morphology.binary_dilation(binary)
binary = morphology.binary_erosion(binary)


plt.imshow(binary, cmap="gray")
plt.show()