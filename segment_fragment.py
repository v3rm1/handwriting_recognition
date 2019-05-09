import sys
import numpy as np
import skimage
from skimage import segmentation, morphology

# Global parameters
EROSION_COUNT = 24


def segment_fragment(image, erosion_count):
    original_image = image

    image = segmentation.inverse_gaussian_gradient(image, alpha=200, sigma=1)
    mask = segmentation.flood(image, (int(
        original_image.shape[0] / 2), int(original_image.shape[1] / 2)), tolerance=0.5)
    mask = morphology.closing(mask, selem=morphology.disk(4))

    while (erosion_count > 0):
        mask = morphology.erosion(mask)
        erosion_count -= 1

    original_image[mask == False] = 255

    image = original_image
    return image


if __name__ == "__main__":

    # Get file from command line ###
    if len(sys.argv) < 2:
        print("ERROR --- Please give the filename in /image-data that you wish to segment")
        exit()
    fileName = "../image-data/" + sys.argv[1]
    img = skimage.io.imread(fileName)

    img = segment_fragment(img, EROSION_COUNT)

    skimage.io.imshow(img)
    skimage.io.show()
