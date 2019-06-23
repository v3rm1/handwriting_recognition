from tensorflow import keras
import pandas as pd
import numpy as np
import os
import cv2
from time import strftime
import sys


habbakuk_char_map = {'Alef' : ')', 
            'Ayin' : '(', 
            'Bet' : 'b', 
            'Dalet' : 'd', 
            'Gimel' : 'g', 
            'He' : 'x', 
            'Het' : 'h', 
            'Kaf' : 'k', 
            'Kaf-final' : '\\', 
            'Lamed' : 'l', 
            'Mem' : '{', 
            'Mem-medial' : 'm', 
            'Nun-final' : '}', 
            'Nun-medial' : 'n', 
            'Pe' : 'p', 
            'Pe-final' : 'v', 
            'Qof' : 'q', 
            'Resh' : 'r', 
            'Samekh' : 's', 
            'Shin' : '$', 
            'Taw' : 't', 
            'Tet' : '+', 
            'Tsadi-final' : 'j', 
            'Tsadi-medial' : 'c', 
            'Waw' : 'w', 
            'Yod' : 'y', 
            'Zayin' : 'z'}



def load_model_file(model_file_path):
    if os.path.exists(model_file_path):
        print("Loading model from {}.".format(model_file_path))
        model = keras.models.load_model(model_file_path)
        return model
    else:
        print("Model file not found.")
        return 0

def predict_chars(model, char_imgs, label_dict_path, file_name=sys.argv[1]):
    pred_text = []
    label_dict = {}

    with open(label_dict_path) as f:
        for line in f:
            (k, v) = line.split()
            k = int(k.split(':')[0])
            label_dict[k] = v

    # model = load_model_file(model_file_path)

    for img in char_imgs:
            input_img = img.reshape(-1, 70, 70, 3)
            pred = model.predict_classes(input_img)
            pred_text.append(label_dict[pred[0]])
    
    return pred_text
    
def convert_to_habbakuk(pred, habbakuk_dict=habbakuk_char_map, img_file_name=sys.argv[1], output_path='./output_txt_files'):
    img_file_name = img_file_name.split('/')[-1]
    
    habbakuk_convert = []
    for char_pos in range(len(pred)-1, -1, -1):
        habbakuk_convert.append(habbakuk_dict[pred[char_pos]])
    
    habbakuk_str = ''.join(habbakuk_convert)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    save_file = os.path.join(output_path, img_file_name.split('.')[0] + '.txt')

    with open(save_file, "a") as f:
        f.write(habbakuk_str)
    f.close()
    return save_file

def english_representation(pred, img_file_name=sys.argv[1], output_path='./output_txt_files_english'):
    img_file_name = img_file_name.split('/')[-1]
    
    char_list = []
    for char_pos in range(len(pred)-1, -1, -1):
        char_list.append(pred[char_pos])
    
    english_str = '  '.join(char_list)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    save_file_english = os.path.join(output_path, img_file_name.split('.')[0] + '.txt')

    with open(save_file_english, "a") as f:
        f.write(english_str)
    f.close()
    return save_file_english

def char_to_text(model, image, label_dict_path):
    
    
    predictions = predict_chars(model, image, label_dict_path)
    eng_rep = english_representation(predictions)
    habbakuk_pred_path = convert_to_habbakuk(predictions)
    return habbakuk_pred_path