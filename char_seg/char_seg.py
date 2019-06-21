import cv2

def edge_detect(file_name, tresh_min, tresh_max):
    image = cv2.imread(file_name)
    # grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    

    #binarize 
    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    
    #find contours
    ctrs, hier = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, 
    cv2.CHAIN_APPROX_SIMPLE)

    #sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

        # Getting ROI
        roi = image[y:y+h, x:x+w]

        # show ROI
        cv2.imwrite('roi_imgs.png', roi)
        # cv2.imshow('charachter'+str(i), roi)
        cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
        # cv2.waitKey(0)

    cv2.imwrite('chr_seg_map_'+file_name, image)
    # cv2.waitKey(0)
    cv2.imshow('marked areas',image)

if __name__ == '__main__':
  edge_detect('scroll_frag.jpg', 10, 100)