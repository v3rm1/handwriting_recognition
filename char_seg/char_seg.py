import cv2
from os import path, makedirs
import numpy as np

def edge_detect(file_name, thresh_min, thresh_max):
    opdir = './segmented/' + file_name.split('.')[0]
    opdir1 = './segmented_greys/' + file_name.split('.')[0]

    if not path.exists(opdir):
            makedirs(opdir)
    if not path.exists(opdir1):
            makedirs(opdir1)
    if not path.exists('./segmentation_map'):
        makedirs('./segmentation_map')


    seg_map_dir = './segmentation_map/map_' + str(thresh_min) + '_' + str(thresh_max) + '_' + file_name
    image = cv2.imread(file_name)
    # grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    # gray = cv2.filter2D(gray, 0, kernel)
    # cv2.imwrite('./segmentation_map/gray_sharpened.jpg', gray)
    # gray = cv2.dilate(gray)


    #binarize 
    ret,thresh = cv2.threshold(gray,thresh_min,thresh_max,cv2.THRESH_BINARY_INV)
    
    #find contours
    ctrs, hier = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, 
    cv2.CHAIN_APPROX_SIMPLE)

    #sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        print(Ctr)

        # Getting ROI
        roi = gray[y:y+h, x:x+w]

        # show ROI
        #cv2.imwrite('roi_imgs.png', roi)
        cv2.imshow('character'+str(i), roi)
        roi_resized = cv2.resize(roi, (70, 70))
        (thresh, im_bw) = cv2.threshold(roi_resized, 128, 255, cv2.THRESH_OTSU)
        (cv2.countNonZero(roi_resized)/roi_resized.size)
        if cv2.countNonZero(roi_resized)/roi_resized.size < 1:
            # print(cv2.countNonZero(roi_resized)/roi_resized.size)
            cv2.imwrite(path.join(opdir, 'character' + str(i) + '.jpg'), im_bw)
            cv2.rectangle(im_bw,(x,y),( x + w, y + h ),(90,0,255),2)
        else:
            cv2.imwrite(path.join(opdir1, 'character' + str(i) + '.jpg'), im_bw)
            cv2.rectangle(im_bw,(x,y),( x + w, y + h ),(90,0,255),2)
        # cv2.waitKey(0)

    cv2.imwrite(seg_map_dir, image)
    # cv2.waitKey(0)
    # cv2.imshow('marked areas',image)

if __name__ == '__main__':
	# for thresh_min in range(0, 140, 5):
	# 	for thresh_max in range(145, 255, 5):
	# 		print(thresh_min, thresh_max)
	# 		edge_detect('scroll_frag.jpg', thresh_min, thresh_max)
    edge_detect('scroll_frag.jpg', 200, 255)