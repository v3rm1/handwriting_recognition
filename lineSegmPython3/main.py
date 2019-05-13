import cv2
import numpy as np
from sys import argv
from lib import sauvola, linelocalization, pathfinder
from time import time as timer

###custom parameters###
n_strips = 10

def draw_line(im, path):
    for p in path:
        im[p[0], p[1]] = 0


def draw_map(im, map):
    for m in map:
        im[m[0], m[1]] = 255


def print_path(path):
    print('\t# path: ' + str(path[::-1]))


def save(filename, imbw):
    imbw_filename = str.replace(filename, '.', '_bw.')
    imbw_filename = str.replace(imbw_filename, 'data', 'data/bw')
    print('Saving image "' + imbw_filename + '"..\n')
    cv2.imwrite(imbw_filename, imbw)
    #immap_filename = str.replace(imbw_filename, '_bw', '_map')
    #cv2.imwrite(immap_filename, immap)
    
def getClosest(a, B):
    best_b = None
    best_diff = 999999
    for b in B:
        diff = abs(a - b)
        if diff < best_diff:
            best_b = b
            best_diff = diff
    return best_b
    
def getStarts(prevLines, lines):
    if prevLines == None:
        return lines
    starts = [None] * len(lines)
    for i in range(0, len(lines)):
        p = getClosest(lines[i], prevLines)
        l = getClosest(p, lines)
        if l == lines[i]:
            starts[i] = p
        else:
            starts[i] = lines[i]
    return starts  
    
def connectPaths(totalPaths, paths):
    for p in paths:
        connection = False
        for i in range(0,len(totalPaths)):
            if totalPaths[i][0] == p[-1]:
                totalPaths[i] = p + totalPaths[i]
                connection = True
        if connection == False:
            totalPaths.append(p)
    return totalPaths

######################
# ------ MAIN ------ #
######################


begin = timer()

filenames = argv
filenames.pop(0)

print('\n############################')
print('##    Line Segmentation   ##')
print('############################\n')

for filename in filenames:
    print('Reading image "' + filename + '"..')
    im = cv2.imread(filename, 0)

    print('- Thresholding image..')
    #imbw = sauvola.binarize(im, [20, 20], 128, 0.3)
	
	### format the (already) binarized image "im"
    imbw = np.array(255 * (im >= 45), 'uint8')
	
    ### modified algorithm to process things in strips, and concatenate those strips###
    im_list= np.array_split(imbw, n_strips, 1)
    im_length = imbw.shape[1]
    strip_length= im_length//n_strips
    
    prev_lines = None
    total_paths= None
    for i in range(0, n_strips):
        lines = linelocalization.localize(im_list[i])
                    
        if lines != []:
            starts = getStarts(prev_lines, lines)
            paths = [None] * len(lines)
            #yeah they have to be even because A* takes steps of 2
            startX = 2 * (strip_length*i // 2)
            endX = 2 * (strip_length*(i+1) // 2)
            for i in range(0, len(lines)):
                path, map = pathfinder.search(imbw, 'A', starts[i], lines[i], startX, endX)
                paths[i] = path
            if total_paths == None:
                total_paths = paths[:]
            else:
                total_paths = connectPaths(total_paths, paths)    
            prev_lines = lines[:]
            
    #extend paths to the edges of the image
    for i in range(0, len(total_paths)):
        if total_paths[i][-1][1] != 0:
            path, map = pathfinder.search(imbw, 'A', total_paths[i][-1][0], total_paths[i][-1][0], 0, total_paths[i][-1][1])
            total_paths[i].extend(path)
        if total_paths[i][0][1] < im_length - 1:
            path, map = pathfinder.search(imbw, 'A', total_paths[i][0][0], total_paths[i][0][0], total_paths[i][0][1], im_length - 1)
            total_paths[i] = path + total_paths[i]
    
    for path in total_paths:
        draw_line(imbw, path)

    save(filename, imbw)

print(' - Elapsed time: ' + str((timer() - begin)) + ' s\n')