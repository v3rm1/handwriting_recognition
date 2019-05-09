import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import skimage
from skimage import segmentation, io, filters, morphology
from segment_fragment import segment_fragment


def binarize_image(image):
    image = segment_fragment(image)

    binary = image > 45

    # binary = morphology.binary_dilation(binary, selem=morphology.square(5))
    # binary = morphology.binary_erosion(binary, selem=morphology.square(5))

    return binary


if __name__ == "__main__":
    # Get file from command line.
    if len(sys.argv) < 2:
        print("[ERROR] Please specify the file you wish to segment.")
        exit()

    file_name = os.path.abspath(sys.argv[1])
    image = io.imread(file_name)

    original_image = image
    plt.figure("Comparison")

    plt.subplot(121)
    plt.imshow(original_image, cmap="gray")

    binary = binarize_image(image)

    plt.subplot(122)
    plt.imshow(binary, cmap="gray")

    # fig.tight_layout()
    plt.show()
