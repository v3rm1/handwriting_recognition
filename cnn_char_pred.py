from tensorflow import keras
import pandas as pd
import numpy as np
import os
import cv2
from time import strftime


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

# TODO: Remove this method since stream comes from segmenter
def test_char_stream(test_file_path='./test'):
    files = []
    if os.path.exists(test_file_path):
        for file in os.listdir(test_file_path):
            f_path = os.path.join('./test', file)
            files.append(f_path)
        return files
    else:
        print("Test file path not found.")
        return 0

# TODO: Add image file names and segmented character as inputs to this function
def predict_chars(model, img_file_names, label_dict_path):
    pred_text = []
    label_dict = {}

    with open(label_dict_path) as f:
        for line in f:
            (k, v) = line.split()
            k = int(k.split(':')[0])
            label_dict[k] = v

    # model = load_model_file(model_file_path)

    for img in img_file_names:
            input_img = cv2.imread(img)
            input_img = input_img.reshape(-1, 70, 70, 3)
            pred = model.predict_classes(input_img)
            pred_text.append(label_dict[pred[0]])
    
    print(pred_text)
    return pred_text
    
def convert_to_habbakuk(pred, habbakuk_dict=habbakuk_char_map, img_file_name='test.jpg', output_path='./output_txt_files'):
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
    return 0

def main():
    model_file_path = './models/kerasThu_20_Jun_19__120212.h5'
    img_file_names = test_char_stream()
    label_dict_path = './models/label_list.txt'
    model = load_model_file(model_file_path)
    predictions = predict_chars(model, img_file_names, label_dict_path)
    # 
    habbakuk_pred = convert_to_habbakuk(predictions)
    print(habbakuk_pred)
    return

   
if __name__ == "__main__":
    main()