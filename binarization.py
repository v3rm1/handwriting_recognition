import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import skimage
from skimage import segmentation, io, filters, morphology
from segment_fragment import segment_fragment

# Global parameters
EROSION_COUNT = 1


def binarize_image(image, erosion_count):
    image = segment_fragment(image, erosion_count)

    binary = image > 50

    # Combination of Dilation and Erosion
    binary = morphology.binary_dilation(binary)
    binary = morphology.binary_erosion(binary)

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

    binary = binarize_image(image, EROSION_COUNT)

    plt.subplot(122)
    plt.imshow(binary, cmap="gray")

    # fig.tight_layout()
    plt.show()
