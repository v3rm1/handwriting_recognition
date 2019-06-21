import sys, os
import numpy as np
import skimage
from skimage import segmentation, morphology

def segment_fragment(image):
    original_image = image

    img2 = np.copy(image)
    image = segmentation.inverse_gaussian_gradient(image, alpha=200, sigma=1)
    mask = segmentation.flood(image, (int(
        original_image.shape[0] / 2), int(original_image.shape[1] / 2)), tolerance=0.5)
    mask = morphology.closing(mask, selem=morphology.disk(4))

    img2[mask == False] = 0
    segmented = segmentation.morphological_chan_vese(img2, 15, init_level_set=mask)
    img2 = np.copy(original_image)
    img2[segmented == False] = 255

    return img2


if __name__ == "__main__":
    # Get file from command line
    if len(sys.argv) < 2:
        print("[ERROR] Please specify the file you wish to segment.")
        exit()

    file_name = os.path.abspath(sys.argv[1])
    img = skimage.io.imread(file_name)

    img = segment_fragment(img)

    skimage.io.imshow(img)
    skimage.io.show()
