import numpy as np
from line_segmenter.main import line_segmentation
from binarization import binarize_image
import matplotlib.pyplot as plt
import sys
import skimage
from skimage import io
import cv2 as cv


def run(im):
    binarized = binarize_image(im)
    cv_binarized = skimage.img_as_ubyte(binarized)
    lines = line_segmentation(cv_binarized)
    characters = []
    contours_viz = [np.copy(l) for l in lines]
    for line in lines:
        contours, _something = cv.findContours(line, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        # viz = cv.drawContours(line, contours, -1, (200, 55, 255), 3)
        for c in contours:
            (x, y, w, h) = cv.boundingRect(c)
            cv2.rectangle(line, (x,y), (x+w,y+h), (0,255,0), 2)
        contours_viz.append(viz)
    return lines


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("[ERROR] Please specify the file you wish to process.")
        exit()

    im = io.imread(sys.argv[1])
    original = np.copy(im)
    lines = run(im)

    plt.figure("Comparison")
    plt.subplot(221)
    plt.imshow(original, cmap="gray")
    plt.subplot(222)
    plt.imshow(lines[2], cmap="gray")
    plt.subplot(223)
    plt.imshow(lines[3], cmap="gray")
    plt.subplot(224)
    plt.imshow(lines[4], cmap="gray")
    plt.tight_layout()
    plt.show()
