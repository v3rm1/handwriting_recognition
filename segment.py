import numpy as np
import skimage
from skimage import segmentation, morphology, filters, exposure

img = skimage.io.imread('test-images/normal-fragment-2.jpg')
img2 = np.copy(img)
preprocessed = segmentation.inverse_gaussian_gradient(img, alpha=200, sigma=1)
preprocessed = morphology.closing(preprocessed, selem=morphology.disk(2))
mask = segmentation.flood(preprocessed, (int(img.shape[0] / 2), int(img.shape[1] / 2)), tolerance=0.5)
mask = morphology.closing(mask, selem=morphology.disk(4))
# contour = segmentation.find_boundaries(mask)
img2[mask == False] = 0
segmented = segmentation.morphological_chan_vese(img2, 15, init_level_set=mask)
img2 = np.copy(img)
img2[segmented == False] = 255
threshold = 50 # filters.threshold_otsu(img2)
binary = img2 < threshold
skimage.io.imshow(binary)
skimage.io.show()