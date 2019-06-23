import numpy as np
from line_segmenter.main import line_segmentation
from binarization import binarize_image
import matplotlib.pyplot as plt
import sys
import preprocessing
import skimage
from skimage import io
import cv2 as cv
from cnn_char_pred import char_to_text, load_model_file


model_file_path = './models/kerasSun_23_Jun_19__191628.h5'
label_dict_path = './models/label_list.txt'

def run(im):
    # binarized = binarize_image(im)
    binarized = preprocessing.main()
    cv_binarized = skimage.img_as_ubyte(binarized)
    
    lines = line_segmentation(cv_binarized)
    lines = [skimage.img_as_float(i) for i in lines]
    # lines = [skimage.img_as_float(cv_binarized)]
    bounding_boxes = []
    for i, line in enumerate(lines):
        # Find the contours
        contours = skimage.measure.find_contours(line, 0.8)
        for contour in contours:
            # Get the contour's bounding box (x, y, w, h)
            bound = (int(contour[:, 0].min()),
                     int(contour[:, 1].min()),
                     int(contour[:, 0].max() - contour[:, 0].min()),
                     int(contour[:, 1].max() - contour[:, 1].min()),
                     i)
            # Only keep it if it's large enough
            if bound[2] > 5 and bound[3] > 5:
                bounding_boxes.append(bound)
        
    bounding_boxes = sorted(bounding_boxes, key=lambda x: x[0])
    merged = []
    done = []
    # Attempt to merge the bounding boxes
    for i1, bound in enumerate(bounding_boxes):
        if i1 in done:
            continue
        for i2, other in enumerate(bounding_boxes):
            if (other[0] >= bound[0] and other[0] <= bound[0] + bound[2]) and (other[1] >= bound[1] and other[1] <= bound[1] + bound[3]):
                merged.append((
                    min(bound[0], other[0]),
                    min(bound[1], other[1]),
                    bound[2] + other[2],
                    bound[3] + other[3],
                    min(bound[4], other[4])
                ))
                done.append(i1)
                done.append(i2)

    boxes = merged + bounding_boxes
    boxes = [box for box in boxes if box[2] < 80 and box[3] < 80]
    windows = np.zeros((len(boxes), 70, 70))
    for i, box in enumerate(boxes):
        windows[i, :, :] = skimage.transform.resize(binarized[box[0]:box[0]+box[2], box[1]:box[1]+box[3]], (70, 70))
    line_ns = [box[4] for box in boxes]
    # Draw the rectangles
    # for rect in boxes:
    #     lines[0][rect[0]:rect[0]+rect[2], rect[1]] = 0
    #     lines[0][rect[0]:rect[0]+rect[2], rect[1]+rect[3]] = 0
    #     lines[0][rect[0], rect[1]:rect[1]+rect[3]] = 0
    #     lines[0][rect[0]+rect[2], rect[1]:rect[1]+rect[3]] = 0
    
    return (windows, line_ns)

        # _fig, ax = plt.subplots()
        # ax.imshow(line, interpolation='nearest', cmap=plt.cm.gray)
        # for n, _contour in enumerate(contours):
        #     ax.plot(contours[n][:, 1], contours[n][:, 0], linewidth=2)
        # plt.show()

    '''
    characters = []
    contours_viz = []
    for line in lines:
        contours, _something = cv.findContours(line, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # viz = cv.drawContours(line, contours, -1, (200, 55, 255), 3)
        viz = np.copy(line)
        for c in contours:
            x, y, w, h = cv.boundingRect(c)
            cv.rectangle(viz, (x,y), (x+w,y+h), (255,255,0), 2)
        # contours_viz.append(viz)
        viz = skimage.img_as_float(viz)
        contours_viz.append(viz)
    return contours_viz
    '''


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("[ERROR] Please specify the file you wish to process.")
        exit()

    im = io.imread(sys.argv[1])
    original = np.copy(im)
    print("ORIGINAL:", original.shape)
    boxes, linenumbers = run(im)
    print("Found {} character segments!!".format(len(boxes)))

    
    model = load_model_file(model_file_path)

    for char_index in range(len(boxes)):
        img = boxes[char_index]
        img = skimage.img_as_ubyte(img)
        img = cv.merge((img, img, img))
        img = np.reshape(img, (-1, 70, 70, 3))
        char_to_text(model, img, label_dict_path)

    # lt.figure("Comparison")
    '''
    plt.subplot(221)
    plt.imshow(original, cmap="gray")
    plt.subplot(222)
    plt.imshow(lines[2], cmap="gray")
    plt.subplot(223)
    plt.imshow(lines[3], cmap="gray")
    plt.subplot(224)
    plt.imshow(lines[4], cmap="gray")
    '''
    '''plt.imshow(lines[0], cmap="gray")
    plt.tight_layout()
    plt.show()'''
