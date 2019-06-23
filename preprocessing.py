"""
Usage: 'preprocessing.py [image-path]'
to preprocess every image use the following bash script (assuming all the images are in '../data'):
'for f in ../data/*; do python region_growing.py $f; done'
Before use: create an 'output' folder, resulting images will be stored there.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from skimage import segmentation, morphology, filters

data_path           = sys.argv[1]
tolerance_seed      = 80  #80
tolerance_grow      = 90  #90

def regiongrow(data_path):
    image = plt.imread(data_path)
    image = np.array(image)

    ### TRY
    image = filters.gaussian(image, sigma=2)
    image *= 255
    ###

    seed, val = find_seed(image)
    print("Seed in X: {}, Y: {}\nSeed pixel value: {:.2f}".format(seed[0],seed[1], val))
    seg = np.zeros(image.shape, dtype=np.bool)
    checked = np.zeros(seg.shape, dtype=np.bool)

    seg[seed] = True
    checked[seed] = True
    need_check = get_neighboors(seed, checked, image.shape)

    while (len(need_check) > 0):
        pt = need_check.pop()
        # Already checked?
        if (checked[pt]): continue
        checked[pt] = True

        # If this doesn't work try dynamically change val with the current pt
        if (image[pt] > val - tolerance_grow and image[pt] < val + tolerance_grow):
            seg[pt] = True
            need_check += get_neighboors(pt, checked, image.shape)

    return seg


def find_seed(image, submatrix_dim=16, lmat=9, stride=30):
    x_dim, y_dim = np.shape(image)
    center = (x_dim // 2, y_dim // 2)
    sample_matrices = []
    sm_dim = submatrix_dim
    for i in range(-lmat,lmat+1):
        for j in range(-lmat,lmat+1):
            matrix = np.zeros((sm_dim,sm_dim))
            start_x = center[0] + (i * stride)
            start_y = center[1] + (j * stride)
            for y in range(sm_dim):
                for x in range(sm_dim):
                    matrix[x,y] = image[start_x + x, start_y + y]
            sample_matrices.append(matrix)
    max_pools = []
    for mat in sample_matrices:
        max_pools.append(np.mean(mat))
    thresh = np.max(max_pools)
    print("Threshold value to find seed: {:.2f}".format(thresh))
    seed_found = False
    seed = center
    while (not seed_found):
        # print("Seed value: {}".format(image[seed]))
        if (image[seed] < (thresh - tolerance_seed) or image[seed] > (thresh + tolerance_seed)):
            seed = random_distance(seed[0],seed[1],biasx=2,biasy=8,unit=4)
        else:
            if(check_neighbours(image, seed, thresh)):
                break
            else:
                seed = random_distance(seed[0],seed[1],biasx=2,biasy=8,unit=4)
    return seed, image[seed]


def random_distance(x,y, biasx=0, biasy=0, unit=1):
    x = x + np.random.randint(0,2*unit+1) - unit + biasx
    y = y + np.random.randint(0,2*unit+1) - unit + biasy
    return (x,y)

def check_neighbours(image, seed, thresh):
    x, y = seed[0], seed[1]
    found = True
    for j in [-1,0,1]:
        for i in [-1,0,1]:
            if (image[(x+i,y+j)] < (thresh - tolerance_seed) or image[(x+i,y+j)] > (thresh + tolerance_seed)):
                # print("Bad neighboor value: {}".format(image[(x+i,y+j)]))
                found = False
    return found

def get_neighboors(pt, checked, dims):
    nbhd = []

    if (pt[0] > 0) and not checked[pt[0]-1, pt[1]]:
        nbhd.append((pt[0]-1, pt[1]))
    if (pt[1] > 0) and not checked[pt[0], pt[1]-1]:
        nbhd.append((pt[0], pt[1]-1))
    if (pt[0] < dims[0]-1) and not checked[pt[0]+1, pt[1]]:
        nbhd.append((pt[0]+1, pt[1]))
    if (pt[1] < dims[1]-1) and not checked[pt[0], pt[1]+1]:
        nbhd.append((pt[0], pt[1]+1))

    return nbhd

def show_comparison(image1, image2):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,9))
    ax[0].imshow(image1, cmap="gray")
    ax[0].axis("off")
    ax[1].imshow(image2, cmap="gray")
    ax[1].axis("off")
    fig.tight_layout()
    plt.show()

def save_comparison(image1, image2, name):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,9))
    ax[0].imshow(image1, cmap="gray")
    ax[0].axis("off")
    ax[1].imshow(image2, cmap="gray")
    ax[1].axis("off")
    fig.tight_layout()
    plt.savefig('output/{}.jpg'.format(name))

def save_image(image, name):
    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.savefig('output/{}.jpg'.format(name))



def main():
    name = sys.argv[1].split('/')[-1].split('.')[0]
    print("Segmenting image: {}".format(name))
    image_og = plt.imread(data_path)
    seg = regiongrow(data_path)

    counter = 0
    for i in seg:
        for j in i:
            if (j == True):
                counter += 1
    print("Segmented area dimension: {}".format(counter))

    print("Processing...")
    seg = 255 * seg
    seg = morphology.dilation(seg, selem=morphology.disk(6))
    seg = morphology.area_closing(seg, area_threshold=3000)
    seg = morphology.dilation(seg, selem=morphology.disk(4))
    seg = morphology.area_closing(seg, area_threshold=400)
    seg = morphology.opening(seg, selem=morphology.disk(16))
    seg = morphology.binary_erosion(seg, selem=morphology.disk(4))


    seg = (1/255) * seg
    res = np.copy(image_og)
    res[seg == False] = 255

    res = filters.gaussian(res, sigma=1.5)
    thresh = filters.threshold_sauvola(res)
    res = (res > thresh)*1
    res = morphology.area_closing(res, area_threshold=100) #64
    res = morphology.binary_erosion(res)

    # Apply eroded mask again to delete borders from Sauvola
    seg = morphology.binary_erosion(seg, selem=morphology.disk(24))
    res[seg == False] = 255

    # show_comparison(image, seg)
    # save_comparison(image_og, res, name)
    # save_image(res, name)

    # print("Done.\n")
    return res