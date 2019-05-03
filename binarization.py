import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import skimage
from skimage import segmentation, io, filters, morphology

### Parameters ###
EROSION_COUNT = 24


image = io.imread("../image-data/P123-Fg001-R-C01-R01-fused.jpg")
original_image = image
plt.figure("Compare", figsize=(15,8))

plt.subplot(121)
plt.imshow(original_image, cmap="gray")

image = segmentation.inverse_gaussian_gradient(image, alpha=200, sigma=1)
mask = segmentation.flood(image, (int(original_image.shape[0] / 2), int(original_image.shape[1] / 2)), tolerance=0.5)
mask = morphology.closing(mask, selem=morphology.disk(4))



while (EROSION_COUNT > 0):
    mask = morphology.erosion(mask)
    EROSION_COUNT -= 1
    


original_image[mask == False] = 255

image = original_image

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


plt.subplot(122)
plt.imshow(binary, cmap="gray")

# fig.tight_layout()
plt.show()


def key_press(self, event):
    if (event.key_press == 'q'):
        plt.close('all')