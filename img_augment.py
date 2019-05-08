"""
Created Date: Apr 30, 2019

Created By: varunravivarma
-------------------------------------------------------------------------------

img_augment.py:
"""
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math
from PIL import ImageOps, Image
import os
from skimage.util import random_noise

INPUT_ROOT = 'labelled'

EXCLUDED_DIRS = {'.DS_Store'}

OUTPUT_ROOT = 'train_aug'

def _standardize_img(input_root=INPUT_ROOT, output_root=OUTPUT_ROOT, excluded_dirs=EXCLUDED_DIRS, output_size=(70, 70)):
    """
    
    Arguments:
    
    Returns:
    """
    for dirs in os.listdir(input_root):
        if dirs not in excluded_dirs:
            char_name = dirs
            if not os.path.exists(os.path.join(output_root, char_name)):
                os.makedirs(os.path.join(output_root, char_name))
            print(char_name)
            i = 1
            _invert_img(os.path.join(input_root, dirs))
            for filename in os.listdir(os.path.join(input_root, dirs)):
                if filename.endswith('jpg'):
                    image_string=tf.read_file(os.path.join(input_root, char_name, filename))
                    img = tf.image.decode_jpeg(image_string, channels=3)
                    padded_img = tf.image.resize_image_with_crop_or_pad(img, output_size[0], output_size[1])
                    op_filename = char_name + '_' + str(i) + '_pad.jpg'
                    op_path = os.path.join(OUTPUT_ROOT, char_name, op_filename)
                    mpimg.imsave(op_path, padded_img.numpy())
                i = i + 1
            _invert_img(os.path.join(input_root, dirs))
    return


def _translate(img, op_path, filename, translate=5):
    """
    Image translation or offset.
    Arguments:
    
    Returns:
    """
    for i in range(-translate, translate, 2):
        if i != 0:
            trans_mat = [i, i]
            trans_img = tf.contrib.image.translate(img, trans_mat)
            op_filename = filename.split('.')[0] + '_translate_' + str(i) + '.jpg'
            op_full_path = os.path.join(op_path, op_filename)
            mpimg.imsave(op_full_path, trans_img.numpy())
    return

def _rotate(img, op_path, filename, max_rot=15):
    """
    Image rotation, creates rotated images in steps of 5 degrees from 0 to max rotation.
    Arguments:
    
    Returns:
    """
    for rot_angle in range(-max_rot, max_rot, 5):
        rot_img = tf.contrib.image.rotate(img, math.radians(rot_angle))
        op_filename = filename.split('.')[0] + '_rot' + str(rot_angle) + '.jpg'
        op_full_path = os.path.join(op_path, op_filename)
        mpimg.imsave(op_full_path, rot_img.numpy())
    return

def _invert_img(directory):
    """
    Inverts all images in a directory b->w and w->b.
    Arguments:
    
    Returns:
    """
    for file in os.listdir(directory):
        if file.endswith('jpeg') or file.endswith('jpg'):
            image = Image.open(os.path.join(directory, file))
            inv_img = ImageOps.invert(image)
            inv_img.save(os.path.join(directory, file))
    return




def main():
    tf.enable_eager_execution()
    _standardize_img()
    print("Finished standardizing with black padding.")
    for dirs in os.listdir(OUTPUT_ROOT):
        if dirs not in EXCLUDED_DIRS and dirs.endswith('png') == False:
            for filename in os.listdir(os.path.join(OUTPUT_ROOT, dirs)):
                image_string=tf.read_file(os.path.join(OUTPUT_ROOT, dirs, filename))
                img = tf.image.decode_jpeg(image_string, channels=3)
                op_path = os.path.join(OUTPUT_ROOT, dirs)
                _translate(img, op_path, filename=filename, translate=6)
                _rotate(img, op_path, filename=filename, max_rot=15)
            _invert_img(op_path)


    print("Finished translation and rotation.")


if __name__ == '__main__':
    main()
