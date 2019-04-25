import numpy as np
import skimage
from skimage import segmentation, morphology

img = skimage.io.imread('test-images/normal-fragment.jpg')
initial_levelset = segmentation.circle_level_set(img.shape, radius=50)
preprocessed = segmentation.inverse_gaussian_gradient(img, alpha=200, sigma=1)
mask = segmentation.flood(preprocessed, (int(img.shape[0] / 2), int(img.shape[1] / 2)), tolerance=0.5)
mask = morphology.closing(mask, selem=morphology.disk(4))
# contour = segmentation.find_boundaries(mask)
img[mask == False] = 0
# segmented = segmentation.morphological_chan_vese(img, 10, init_level_set=mask)

skimage.io.imshow(img)
skimage.io.show()